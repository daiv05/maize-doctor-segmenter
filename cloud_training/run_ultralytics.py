#!/usr/bin/env python3
"""Authorized cloud runner for smoke, full train, resume, validation and test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml

from src.evaluation.segmentation_downstream import ROW_COLUMNS, evaluate_downstream
from src.evaluation.split_instance_counts import count_split_instances
from src.training.checkpoint_promotion import expected_best_checkpoint_sha256
from src.training.segmentation_experiments import materialize_source_balanced_dataset
from src.training.segmentation_preflight import (
    EXPECTED_INITIAL_WEIGHTS_SHA256,
    verify_cloud_training_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def project_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default))
    return value if value.is_absolute() else ROOT / value


DATASET = project_path(
    "LEAF_SEGMENTATION_DATASET",
    "data/leaf_detection/detector_dataset",
)
OUTPUTS = project_path("LEAF_SEGMENTATION_OUTPUT", "outputs/leaf_detection")
CLOUD_DIR = project_path("CLOUD_TRAINING_DIR", "cloud_training")
MODEL_NAME = os.getenv("SEGMENTATION_MODEL", "yolo26n-seg.pt")
DEVICE = os.getenv("SEGMENTATION_DEVICE", "0")
DEVICE_INDEX = int(DEVICE.split(",", maxsplit=1)[0])
REQUESTED_EVALUATION_SPLIT = "test"
EXPECTED_TEST_FINGERPRINT = (
    "046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51"
)
EXPECTED_FASTER_COCO_EVAL = "1.7.2"
ALLOWED_EXPERIMENT_SEEDS = {7, 42, 1337}
EVALUATION_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BOX_METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
MASK_METRIC_KEYS = (
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config inválida: {path}")
    return payload


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Atomically write the stable per-image downstream schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        fieldnames: list[str] = [str(column) for column in ROW_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a frozen YAML without ever replacing an earlier decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"No se sobrescribe configuración congelada: {path}") from exc


def verified_weights(
    manifest_name: str = "weights_manifest.json",
    *,
    expected_filename: str | None = None,
) -> Path:
    manifest_path = OUTPUTS / "cloud_preflight" / manifest_name
    if not manifest_path.is_file():
        raise RuntimeError(f"Falta manifiesto de pesos verificados: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("resolved") or not manifest.get("path"):
        raise RuntimeError(
            f"{manifest_path} no resolvió pesos (resolved={manifest.get('resolved')!r}); "
            "vuelva a ejecutar el preflight cloud antes de entrenar"
        )
    path = Path(str(manifest["path"]))
    digest = sha256(path) if path.is_file() else None
    if digest is None or digest != manifest["sha256"]:
        raise RuntimeError(
            f"Los pesos no coinciden con {manifest_path.name}: "
            f"{path} sha256={digest}, esperado {manifest['sha256']}"
        )
    if expected_filename is not None and path.name != expected_filename:
        raise RuntimeError(
            f"Pesos inesperados: {path.name!r} != {expected_filename!r}"
        )
    frozen = EXPECTED_INITIAL_WEIGHTS_SHA256.get(path.name)
    if frozen is not None and digest != frozen:
        raise RuntimeError(
            f"Procedencia de {path.name} fuera del contrato congelado: "
            f"{digest} != {frozen}"
        )
    return path


def resolve_initial_model(profile: str) -> tuple[str, dict[str, Any]]:
    """Resolve only explicit, auditable initialization profiles."""
    if profile == "pretrained_yolo26n":
        path = verified_weights()
        return str(path), checkpoint_record(path)
    if profile == "scratch_yolo26n":
        return "yolo26n-seg.yaml", {
            "path": "yolo26n-seg.yaml",
            "initialization": "scratch",
            "weights_sha256": None,
        }
    if profile == "pretrained_yolo26s":
        path = verified_weights(
            "weights_manifest_yolo26s.json",
            expected_filename="yolo26s-seg.pt",
        )
        return str(path), checkpoint_record(path)
    raise RuntimeError(f"initialization_profile no permitido: {profile!r}")


def base_gate() -> dict[str, Any]:
    payload = verify_cloud_training_payload(DATASET)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA dejó de estar disponible")
    free = shutil.disk_usage(OUTPUTS if OUTPUTS.exists() else ROOT).free
    if free < 10 * 1024**3:
        raise RuntimeError("Se requieren al menos 10 GiB libres para resultados")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    probe = OUTPUTS / ".persistence_probe"
    probe.write_text("persistent-output-check\n", encoding="utf-8")
    probe.unlink()
    return payload


def initialize_cuda_and_reset_peak_memory_stats(device_index: int) -> torch.device:
    """Select and initialize one valid CUDA device before touching allocator stats."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA dejó de estar disponible")
    device_count = torch.cuda.device_count()
    if not 0 <= device_index < device_count:
        raise RuntimeError(
            f"Índice CUDA fuera de rango: {device_index}; dispositivos={device_count}"
        )

    torch.cuda.set_device(device_index)
    torch.cuda.init()
    if not torch.cuda.is_initialized():
        raise RuntimeError("PyTorch no inicializó CUDA")
    current_device = torch.cuda.current_device()
    if current_device != device_index:
        raise RuntimeError(
            f"Dispositivo CUDA activo inesperado: {current_device} != {device_index}"
        )

    cuda_device = torch.device("cuda", device_index)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    return cuda_device


def timestamped_manifest_path(directory: Path, stamp: datetime | None = None) -> Path:
    """Manifest path that never collides with a previous resume record."""
    moment = stamp if stamp is not None else datetime.now(timezone.utc)
    return directory / f"resume_manifest_{moment.strftime('%Y%m%dT%H%M%S%fZ')}.json"


def metrics_dict(result: Any) -> dict[str, Any]:
    values = getattr(result, "results_dict", {})
    return {str(key): float(value) for key, value in values.items()}


def to_serializable(value: Any) -> Any:
    """Convert tensors/arrays into JSON-safe values for the run summary."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return str(value)


def resolve_trainer(model: Any, result: Any) -> Any:
    """Prefer the model's trainer; metrics objects usually lack one."""
    return getattr(model, "trainer", None) or getattr(result, "trainer", None)


def numeric_values(value: Any) -> list[float]:
    """Flatten numeric tensor/list/mapping values while rejecting booleans."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, numbers.Real):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in numeric_values(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in numeric_values(item)]
    return []


def require_finite_numeric(field: str, value: Any) -> list[float]:
    values = numeric_values(value)
    if not values:
        raise RuntimeError(f"{field} no contiene valores numéricos")
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"{field} contiene NaN o infinito: {values}")
    return values


def selected_positive_batch(trainer: Any) -> int:
    candidates = (
        getattr(trainer, "batch_size", None),
        getattr(getattr(trainer, "args", None), "batch", None),
    )
    for value in candidates:
        if (
            isinstance(value, numbers.Integral)
            and not isinstance(value, bool)
            and int(value) > 0
        ):
            return int(value)
    raise RuntimeError(f"Batch efectivo inválido: {candidates!r}")


def expected_run_directory(config: dict[str, Any]) -> Path:
    name = config.get("name")
    if not isinstance(name, str) or not name.strip() or Path(name).name != name:
        raise RuntimeError(f"name de run inválido: {name!r}")
    return (OUTPUTS / "segmenter" / name).resolve()


def checkpoint_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Checkpoint obligatorio ausente: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def package_trace() -> dict[str, Any]:
    manifest_path = CLOUD_DIR / "package_manifest.json"
    if not manifest_path.is_file():
        return {"manifest": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest": str(manifest_path),
        "package_version": manifest.get("package_version"),
        "git": manifest.get("git"),
        "parent_fingerprint": manifest.get("parent_fingerprint"),
        "split_fingerprints": manifest.get("split_fingerprints"),
    }


def _is_vendored(distribution: Any) -> bool:
    """Indica si una distribución vive dentro de un árbol ``_vendor``.

    setuptools publica metadatos de sus dependencias vendorizadas. Se vuelven visibles
    para ``importlib.metadata`` en cuanto algo lo importa, así que contarlas haría que
    el inventario cambiara sin que se instale nada.
    """
    try:
        location = Path(str(distribution.locate_file("")))
    except Exception:
        return False
    return "_vendor" in location.parts


def installed_distribution_snapshot() -> dict[str, Any]:
    """Hash the installed distribution inventory to detect runtime installs."""
    rows = sorted(
        {
            (
                str(distribution.metadata.get("Name", "")).strip().lower(),
                str(distribution.version),
            )
            for distribution in metadata.distributions()
            if str(distribution.metadata.get("Name", "")).strip()
            and not _is_vendored(distribution)
        }
    )
    encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
    try:
        faster_coco_eval = metadata.version("faster-coco-eval")
    except metadata.PackageNotFoundError:
        faster_coco_eval = None
    return {
        "distribution_count": len(rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "faster_coco_eval": faster_coco_eval,
        "names": [name for name, _ in rows],
    }


def require_evaluated_split(requested_split: str, evaluated_split: str) -> None:
    """Block a summary when Ultralytics did not evaluate the requested split."""
    if requested_split != evaluated_split:
        raise RuntimeError(
            "Ultralytics evaluó un split distinto al solicitado: "
            f"requested_split={requested_split!r}, "
            f"evaluated_split={evaluated_split!r}"
        )


def _validator_argument(model: Any, name: str) -> Any:
    args = getattr(model, "args", None)
    if isinstance(args, dict):
        return args.get(name)
    return getattr(args, name, None)


def capture_validation_observation(
    validator: Any,
    observation: dict[str, Any],
) -> None:
    """Capture the effective split and dataset after Ultralytics builds its loader."""
    evaluated_split = str(_validator_argument(validator, "split") or "")
    data = getattr(validator, "data", {})
    split_path = data.get(evaluated_split) if isinstance(data, dict) else None
    dataset = getattr(getattr(validator, "dataloader", None), "dataset", None)
    labels = getattr(dataset, "labels", None)
    if dataset is None or not isinstance(labels, list):
        raise RuntimeError("Ultralytics no expuso el dataset efectivo de validación")
    observation.update(
        {
            "evaluated_split": evaluated_split,
            "resolved_split_path": str(Path(str(split_path)).resolve()),
            "image_count": len(dataset),
            "instance_count": sum(
                len(label.get("cls", ()))
                for label in labels
                if isinstance(label, dict)
            ),
        }
    )


def _metric_groups(metrics: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    missing = [
        key
        for key in (*BOX_METRIC_KEYS, *MASK_METRIC_KEYS)
        if key not in metrics
    ]
    if missing:
        raise RuntimeError(f"Faltan métricas oficiales de cajas/máscaras: {missing}")
    require_finite_numeric("evaluation_metrics", metrics)
    return (
        {key: metrics[key] for key in BOX_METRIC_KEYS},
        {key: metrics[key] for key in MASK_METRIC_KEYS},
    )


def validate_test_evaluation_inputs(
    checkpoint: Path,
    dataset_gate: dict[str, Any],
) -> dict[str, Any]:
    """Validate the retained test split and exact immutable checkpoint."""
    dataset_yaml_path = (DATASET / "dataset.yaml").resolve()
    dataset_yaml = load_yaml(dataset_yaml_path)
    configured_test = dataset_yaml.get("test")
    if configured_test != "images/test":
        raise RuntimeError(
            "dataset.yaml no fija test: images/test; "
            f"actual={configured_test!r}"
        )
    serialized_dataset_yaml = json.dumps(
        dataset_yaml,
        sort_keys=True,
        ensure_ascii=False,
    ).lower()
    if "pilot" in serialized_dataset_yaml:
        raise RuntimeError("El piloto aparece en dataset.yaml")
    resolved_test = (dataset_yaml_path.parent / configured_test).resolve()
    expected_test = (DATASET / "images" / "test").resolve()
    if resolved_test != expected_test or resolved_test.parts[-2:] != ("images", "test"):
        raise RuntimeError(
            f"Ruta test resuelta incorrectamente: {resolved_test} != {expected_test}"
        )
    label_dir = (DATASET / "labels" / "test").resolve()
    if not resolved_test.is_dir() or not label_dir.is_dir():
        raise RuntimeError("Faltan las rutas canónicas images/test o labels/test")
    images = sorted(
        path
        for path in resolved_test.iterdir()
        if path.is_file() and path.suffix.lower() in EVALUATION_IMAGE_EXTENSIONS
    )
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        raise RuntimeError("La correspondencia images/test ↔ labels/test no es exacta")
    counts = count_split_instances(DATASET, REQUESTED_EVALUATION_SPLIT)
    locked_images = int(dataset_gate["image_counts"][REQUESTED_EVALUATION_SPLIT])
    locked_annotations = int(dataset_gate["mask_counts"][REQUESTED_EVALUATION_SPLIT])
    if len(images) != locked_images or counts.image_count != locked_images:
        raise RuntimeError(
            f"Conteo test inválido: {len(images)}/{counts.image_count} != {locked_images}"
        )
    if counts.annotation_count != locked_annotations:
        raise RuntimeError(
            "Anotaciones test inválidas: "
            f"{counts.annotation_count} != {locked_annotations}"
        )
    actual_test_fingerprint = str(
        dataset_gate.get("split_fingerprints", {}).get("test", "")
    )
    if actual_test_fingerprint != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(
            "Fingerprint test inválido: "
            f"{actual_test_fingerprint} != {EXPECTED_TEST_FINGERPRINT}"
        )

    checkpoint = checkpoint.resolve()
    runs_root = (OUTPUTS / "segmenter").resolve()
    try:
        relative = checkpoint.relative_to(runs_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Checkpoint de evaluación fuera de {runs_root}: {checkpoint}"
        ) from exc
    if len(relative.parts) != 3 or relative.parts[1:] != ("weights", "best.pt"):
        raise RuntimeError(
            "El checkpoint de evaluación debe ser <run>/weights/best.pt; "
            f"recibido {relative.as_posix()}"
        )
    run_name = relative.parts[0]
    checkpoint_before = checkpoint_record(checkpoint)
    expected_checkpoint_sha256 = expected_best_checkpoint_sha256(OUTPUTS.parent)
    if checkpoint_before["sha256"] != expected_checkpoint_sha256:
        raise RuntimeError(
            "SHA-256 de best.pt inesperado: "
            f"{checkpoint_before['sha256']} != {expected_checkpoint_sha256}"
        )

    evaluation_root = (OUTPUTS / "segmenter_evaluation").resolve()
    expected_save_dir = evaluation_root / "yolo26n_seg_test"
    prediction_dir = evaluation_root / "yolo26n_seg_test_predictions"
    summary_path = evaluation_root / "test_summary.json"
    previous = [
        path
        for path in (expected_save_dir, prediction_dir, summary_path)
        if path.exists()
    ]
    if previous:
        if os.getenv("FORCE_INTERNAL_TEST_RERUN") != "1":
            raise RuntimeError(
                "No se reutilizan resultados test existentes: "
                f"{[str(path) for path in previous]}. Repetir el test sobre la misma "
                "configuración invalida su valor metodológico; sólo con una decisión "
                "formal registrada: FORCE_INTERNAL_TEST_RERUN=1"
            )
        archive_root = evaluation_root / "superseded"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in previous:
            shutil.move(str(path), str(archive_root / f"{path.name}.{stamp}"))
        print(
            f"FORCE_INTERNAL_TEST_RERUN=1: {len(previous)} artefacto(s) previo(s) "
            f"archivados en {archive_root}",
            flush=True,
        )
    return {
        "requested_split": REQUESTED_EVALUATION_SPLIT,
        "run_name": run_name,
        "dataset_yaml": str(dataset_yaml_path),
        "resolved_split_path": str(resolved_test),
        "image_count": len(images),
        "instance_count": counts.annotation_count,
        "annotation_count": counts.annotation_count,
        "loader_instance_count": counts.loader_instance_count,
        "loader_deduplicated_annotations": counts.deduplicated_annotations,
        "loader_deduplicated_images": list(counts.deduplicated_images),
        "test_fingerprint": actual_test_fingerprint,
        "pilot_used": False,
        "checkpoint": checkpoint_before,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "expected_save_dir": str(expected_save_dir),
        "prediction_dir": str(prediction_dir),
        "summary_path": str(summary_path),
    }


def require_confirmation(mode: str) -> None:
    variable = (
        "CONFIRM_SEGMENTATION_SMOKE_TRAINING"
        if mode == "smoke"
        else "CONFIRM_SEGMENTATION_TRAINING"
    )
    if os.getenv(variable) != "1":
        raise RuntimeError(f"{mode} no autorizado: use {variable}=1")


def train_mode(mode: str, config_path: Path) -> None:
    from ultralytics import YOLO

    payload = base_gate()
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    if config.pop("task", None) != "segment":
        raise RuntimeError("La configuración debe declarar task=segment")
    experiment_id = config.pop("experiment_id", None)
    initialization_profile = str(
        config.pop("initialization_profile", "pretrained_yolo26n")
    )
    sampling_profile = str(config.pop("sampling_profile", "canonical"))
    seed = config.get("seed")
    if config.get("deterministic") is not True:
        raise RuntimeError("La configuración debe conservar deterministic=true")
    if experiment_id is None and seed != 42:
        raise RuntimeError("El baseline debe conservar seed=42")
    if experiment_id is not None and seed not in ALLOWED_EXPERIMENT_SEEDS:
        raise RuntimeError(
            f"Semilla de experimento no permitida: {seed!r}; "
            f"use {sorted(ALLOWED_EXPERIMENT_SEEDS)}"
        )
    requested_batch = config.get("batch")
    if mode == "smoke":
        if requested_batch != -1:
            raise RuntimeError("El smoke debe usar batch=-1 para medir AutoBatch")
        if config.get("epochs") != 1:
            raise RuntimeError("El smoke debe ejecutar exactamente una época")
    elif (
        not isinstance(requested_batch, numbers.Integral)
        or isinstance(requested_batch, bool)
        or int(requested_batch) <= 0
    ):
        raise RuntimeError(
            "El entrenamiento completo requiere un batch entero positivo congelado"
        )
    elif (
        not isinstance(config.get("epochs"), numbers.Integral)
        or isinstance(config.get("epochs"), bool)
        or int(config["epochs"]) <= 1
    ):
        raise RuntimeError("El entrenamiento completo requiere más de una época")
    model_path, initial_model = resolve_initial_model(initialization_profile)
    config["model"] = model_path
    config["data"] = str(DATASET / "dataset.yaml")
    config["device"] = DEVICE
    config["project"] = str(OUTPUTS / "segmenter")
    config["exist_ok"] = False
    expected_save_dir = expected_run_directory(config)
    if expected_save_dir.exists():
        raise RuntimeError(
            f"El run solicitado ya existe; no se permite baseline2 implícito: "
            f"{expected_save_dir}"
        )
    sampling_metadata: dict[str, object] = {
        "profile": "canonical",
        "dataset_yaml": str((DATASET / "dataset.yaml").resolve()),
        "test_included": False,
        "pilot_included": False,
    }
    if sampling_profile == "source_balanced_corn":
        sampling_metadata = materialize_source_balanced_dataset(
            DATASET,
            OUTPUTS / "segmenter" / "experiment_inputs" / str(config["name"]),
        )
        config["data"] = str(sampling_metadata["dataset_yaml"])
    elif sampling_profile != "canonical":
        raise RuntimeError(f"sampling_profile no permitido: {sampling_profile!r}")
    model_path = config.pop("model")
    started = time.monotonic()
    cuda_device = initialize_cuda_and_reset_peak_memory_stats(DEVICE_INDEX)
    if mode == "smoke":
        summary_path = OUTPUTS / "segmenter" / "smoke_summary.json"
    elif experiment_id is not None:
        summary_path = (
            OUTPUTS / "segmenter" / "experiment_summaries" / f"{config['name']}.json"
        )
    else:
        summary_path = OUTPUTS / "segmenter" / "training_summary.json"
    summary: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "expected_save_dir": str(expected_save_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprints": payload,
        "experiment_id": experiment_id,
        "initialization_profile": initialization_profile,
        "sampling": sampling_metadata,
        "initial_weights": initial_model,
        "source": package_trace(),
    }
    final_path = OUTPUTS / "segmenter" / "configs" / "train_yolo26n_seg.final.yaml"
    active_manifest_path = (
        OUTPUTS / "segmenter" / "active_run_manifest.json"
        if experiment_id is None
        else OUTPUTS
        / "segmenter"
        / "experiment_manifests"
        / f"{config['name']}.json"
    )
    if mode == "smoke" and final_path.exists():
        raise RuntimeError(
            f"Ya existe una configuración final; no se sobrescribe: {final_path}"
        )
    if mode == "train":
        if active_manifest_path.exists():
            raise RuntimeError(
                f"Ya existe identidad de run activa: {active_manifest_path}"
            )
        write(
            active_manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "run_id": config["name"],
                "experiment_id": experiment_id,
                "save_dir": str(expected_save_dir),
                "expected_last_checkpoint": str(
                    expected_save_dir / "weights" / "last.pt"
                ),
                "expected_best_checkpoint": str(
                    expected_save_dir / "weights" / "best.pt"
                ),
                "config": str(config_path),
                "config_sha256": summary["config_sha256"],
                "dataset_fingerprints": payload,
                "initial_weights": summary["initial_weights"],
                "initialization_profile": initialization_profile,
                "sampling": sampling_metadata,
                "started_utc": summary["started_utc"],
                "source": summary["source"],
            },
        )
    try:
        model = YOLO(model_path)
        result = model.train(**config)
        trainer = resolve_trainer(model, result)
        if trainer is None:
            raise RuntimeError("Ultralytics no expuso el trainer efectivo")
        save_dir = Path(
            str(getattr(trainer, "save_dir", None) or getattr(result, "save_dir", ""))
        ).resolve()
        if save_dir != expected_save_dir:
            raise RuntimeError(
                f"Directorio de run inesperado: {save_dir} != {expected_save_dir}"
            )
        selected_batch = selected_positive_batch(trainer)
        loss = to_serializable(getattr(trainer, "loss_items", None))
        require_finite_numeric("loss", loss)
        metrics = metrics_dict(result)
        require_finite_numeric("metrics", metrics)
        trainer_device = str(getattr(trainer, "device", ""))
        peak_vram = torch.cuda.max_memory_allocated(cuda_device)
        if not trainer_device.startswith("cuda") or peak_vram <= 0:
            raise RuntimeError(
                "No se verificó uso efectivo de GPU: "
                f"device={trainer_device!r}, peak_vram={peak_vram}"
            )
        last = checkpoint_record(save_dir / "weights" / "last.pt")
        checkpoints = {"last": last}
        if mode == "train":
            checkpoints["best"] = checkpoint_record(save_dir / "weights" / "best.pt")
        effective_config = {
            "task": "segment",
            "experiment_id": experiment_id,
            "initialization_profile": initialization_profile,
            "sampling_profile": sampling_profile,
            "model": model_path,
            **config,
            "batch": selected_batch,
        }
        effective_config_path = save_dir / "doctor_maiz_effective_config.yaml"
        write_yaml_exclusive(effective_config_path, effective_config)
        summary.update(
            {
                "status": "passed",
                "duration_seconds": time.monotonic() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "save_dir": str(save_dir),
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "metrics": metrics,
                "loss": loss,
                "selected_batch": selected_batch,
                "checkpoints": checkpoints,
                "checkpoint": (
                    checkpoints["last"]["path"]
                    if mode == "smoke"
                    else checkpoints["best"]["path"]
                ),
                "checkpoint_sha256": (
                    checkpoints["last"]["sha256"]
                    if mode == "smoke"
                    else checkpoints["best"]["sha256"]
                ),
                "effective_config": str(effective_config_path),
                "effective_config_sha256": sha256(effective_config_path),
                "errors": [],
            }
        )
        if mode == "smoke":
            final = load_yaml(CLOUD_DIR / "configs" / "train_yolo26n_seg.yaml")
            final["model"] = model_path
            final["initialization_profile"] = initialization_profile
            final["data"] = str(DATASET / "dataset.yaml")
            final["device"] = DEVICE
            final["project"] = str(OUTPUTS / "segmenter")
            final["batch"] = selected_batch
            write_yaml_exclusive(final_path, final)
            summary["final_config"] = str(final_path)
            summary["final_config_sha256"] = sha256(final_path)
        else:
            active_manifest = json.loads(
                active_manifest_path.read_text(encoding="utf-8")
            )
            active_manifest.update(
                {
                    "status": "completed",
                    "completed_utc": summary["completed_utc"],
                    "selected_batch": selected_batch,
                    "trainer_device": trainer_device,
                    "checkpoints": checkpoints,
                    "effective_config": str(effective_config_path),
                    "effective_config_sha256": summary["effective_config_sha256"],
                }
            )
            write(active_manifest_path, active_manifest)
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "duration_seconds": time.monotonic() - started,
                "errors": [repr(exc)],
            }
        )
        if mode == "train" and active_manifest_path.is_file():
            active_manifest = json.loads(
                active_manifest_path.read_text(encoding="utf-8")
            )
            active_manifest.update(
                {
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                }
            )
            write(active_manifest_path, active_manifest)
        write(summary_path, summary)
        raise
    write(summary_path, summary)


def resume_mode(
    checkpoint: Path,
    reason: str,
    active_manifest: Path | None = None,
) -> None:
    from ultralytics import YOLO

    payload = base_gate()
    baseline_active_path = (OUTPUTS / "segmenter" / "active_run_manifest.json").resolve()
    experiment_manifest_root = (
        OUTPUTS / "segmenter" / "experiment_manifests"
    ).resolve()
    active_path = (
        active_manifest.resolve()
        if active_manifest is not None
        else baseline_active_path
    )
    if active_path != baseline_active_path and active_path.parent != experiment_manifest_root:
        raise RuntimeError(f"Manifiesto activo fuera del área permitida: {active_path}")
    if not active_path.is_file():
        raise RuntimeError(f"Falta identidad activa: {active_path}")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    if active.get("status") not in {"running", "failed"}:
        raise RuntimeError(f"Run no reanudable: {active.get('status')!r}")
    checkpoint = checkpoint.resolve()
    expected_checkpoint = Path(
        str(active.get("expected_last_checkpoint", ""))
    ).resolve()
    if checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"Checkpoint fuera del run activo: {checkpoint} != {expected_checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = Path(str(active.get("config", "")))
    if (
        not config_path.is_file()
        or sha256(config_path) != active.get("config_sha256")
    ):
        raise RuntimeError("La configuración original del run cambió")
    if active.get("dataset_fingerprints") != payload:
        raise RuntimeError("El dataset ya no coincide con el run interrumpido")
    model = YOLO(str(checkpoint))
    checkpoint_payload = getattr(model, "ckpt", None)
    epoch = (
        checkpoint_payload.get("epoch")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    experiment_id = active.get("experiment_id")
    run_id = str(active.get("run_id", ""))
    if experiment_id is not None and active_path.parent != experiment_manifest_root:
        raise RuntimeError("Un experimento debe usar experiment_manifests/")
    if experiment_id is None and active_path != baseline_active_path:
        raise RuntimeError("El baseline debe usar active_run_manifest.json")
    if not run_id:
        raise RuntimeError("El manifiesto activo no declara run_id")
    manifest = {
        "status": "authorized",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": epoch,
        "original_configuration": str(config_path),
        "original_configuration_sha256": active["config_sha256"],
        "run_id": run_id,
        "experiment_id": experiment_id,
        "active_manifest": str(active_path),
        "save_dir": active.get("save_dir"),
        "dataset_fingerprints": payload,
        "utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    # Un manifiesto por reanudación (histórico completo) y una copia con el
    # nombre estable que consultan los consumidores existentes.
    resume_root = OUTPUTS / "segmenter"
    if experiment_id is not None:
        history_root = resume_root / "experiment_resume_history" / run_id
        stable_resume_path = (
            resume_root / "experiment_resume_manifests" / f"{run_id}.json"
        )
        training_summary_path = (
            resume_root / "experiment_summaries" / f"{run_id}.json"
        )
    else:
        history_root = resume_root
        stable_resume_path = resume_root / "resume_manifest.json"
        training_summary_path = resume_root / "training_summary.json"
    history_path = timestamped_manifest_path(history_root)
    write(history_path, manifest)
    write(stable_resume_path, manifest)
    try:
        cuda_device = initialize_cuda_and_reset_peak_memory_stats(DEVICE_INDEX)
        result = model.train(resume=True)
        trainer = resolve_trainer(model, result)
        if trainer is None:
            raise RuntimeError("Ultralytics no expuso el trainer al reanudar")
        save_dir = Path(str(getattr(trainer, "save_dir", ""))).resolve()
        if save_dir != Path(str(active["save_dir"])).resolve():
            raise RuntimeError("La reanudación cambió el directorio del run")
        selected_batch = selected_positive_batch(trainer)
        loss = to_serializable(getattr(trainer, "loss_items", None))
        require_finite_numeric("loss", loss)
        metrics = metrics_dict(result)
        require_finite_numeric("metrics", metrics)
        trainer_device = str(getattr(trainer, "device", ""))
        peak_vram = torch.cuda.max_memory_allocated(cuda_device)
        if not trainer_device.startswith("cuda") or peak_vram <= 0:
            raise RuntimeError(
                "No se verificó uso efectivo de GPU durante la reanudación"
            )
        checkpoints = {
            "last": checkpoint_record(save_dir / "weights" / "last.pt"),
            "best": checkpoint_record(save_dir / "weights" / "best.pt"),
        }
        manifest.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "selected_batch": selected_batch,
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "metrics": metrics,
                "loss": loss,
                "checkpoints": checkpoints,
            }
        )
        active.update(
            {
                "status": "completed",
                "completed_utc": manifest["completed_utc"],
                "selected_batch": selected_batch,
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "checkpoints": checkpoints,
                "resumed": True,
            }
        )
        training_summary = (
            json.loads(training_summary_path.read_text(encoding="utf-8"))
            if training_summary_path.is_file()
            else {}
        )
        training_summary.update(
            {
                "status": "passed",
                "mode": "train",
                "resumed": True,
                "completed_utc": manifest["completed_utc"],
                "save_dir": str(save_dir),
                "selected_batch": selected_batch,
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "metrics": metrics,
                "loss": loss,
                "checkpoints": checkpoints,
                "errors": [],
            }
        )
        write(active_path, active)
        write(training_summary_path, training_summary)
        write(history_path, manifest)
        write(stable_resume_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        write(history_path, manifest)
        write(stable_resume_path, manifest)
        raise


def evaluate_mode(checkpoint: Path, config_path: Path, split: str) -> None:
    environment_before = installed_distribution_snapshot()
    if environment_before["faster_coco_eval"] != EXPECTED_FASTER_COCO_EVAL:
        raise RuntimeError(
            "faster-coco-eval no está fijado antes de evaluar: "
            f"{environment_before['faster_coco_eval']!r} "
            f"!= {EXPECTED_FASTER_COCO_EVAL!r}"
        )
    from ultralytics import YOLO

    dataset_gate = base_gate()
    if split != REQUESTED_EVALUATION_SPLIT:
        raise ValueError(
            "La evaluación final exige --split test explícito; "
            f"recibido={split!r}"
        )
    contract = validate_test_evaluation_inputs(checkpoint, dataset_gate)
    config = load_yaml(config_path)
    if config.get("split") != REQUESTED_EVALUATION_SPLIT:
        raise RuntimeError(
            "La configuración de evaluación no fija split=test: "
            f"{config.get('split')!r}"
        )
    config.pop("task", None)
    config["data"] = str(DATASET / "dataset.yaml")
    config["device"] = DEVICE
    config["project"] = str(OUTPUTS / "segmenter_evaluation")
    config["split"] = REQUESTED_EVALUATION_SPLIT
    config["name"] = "yolo26n_seg_test"
    config["exist_ok"] = False
    model = YOLO(str(checkpoint))
    validation_observation: dict[str, Any] = {}
    model.add_callback(
        "on_val_start",
        lambda validator: capture_validation_observation(
            validator,
            validation_observation,
        ),
    )
    result = model.val(
        **config,
        plots=True,
        save_json=True,
        save_txt=True,
        save_conf=False,
    )
    evaluated_split = str(validation_observation.get("evaluated_split", ""))
    require_evaluated_split(REQUESTED_EVALUATION_SPLIT, evaluated_split)
    evaluated_split_path = Path(
        str(validation_observation.get("resolved_split_path", ""))
    ).resolve()
    expected_split_path = Path(str(contract["resolved_split_path"])).resolve()
    if evaluated_split_path != expected_split_path:
        raise RuntimeError(
            "Ultralytics resolvió una ruta distinta para test: "
            f"{evaluated_split_path} != {expected_split_path}"
        )
    evaluated_image_count = validation_observation.get("image_count")
    evaluated_instance_count = validation_observation.get("instance_count")
    expected_loader_instances = int(contract["loader_instance_count"])
    if (
        evaluated_image_count != contract["image_count"]
        or evaluated_instance_count != expected_loader_instances
    ):
        raise RuntimeError(
            "Ultralytics no evaluó los conteos test derivados del dataset: "
            f"images={evaluated_image_count}/{contract['image_count']}, "
            f"instances={evaluated_instance_count}/{expected_loader_instances} "
            f"(anotaciones={contract['annotation_count']}, "
            f"colapsadas por bbox repetido={contract['loader_deduplicated_annotations']})"
        )
    actual_save_dir = Path(
        str(
            getattr(result, "save_dir", None)
            or getattr(getattr(model, "validator", None), "save_dir", "")
        )
    ).resolve()
    expected_save_dir = Path(str(contract["expected_save_dir"])).resolve()
    if actual_save_dir != expected_save_dir:
        raise RuntimeError(
            f"Directorio test inesperado: {actual_save_dir} != {expected_save_dir}"
        )
    prediction_labels_dir = actual_save_dir / "labels"
    if not prediction_labels_dir.is_dir():
        raise RuntimeError(
            "Ultralytics no guardó las etiquetas YOLO-seg necesarias para las "
            f"métricas downstream: {prediction_labels_dir}"
        )
    downstream_rows, downstream_summary = evaluate_downstream(
        dataset_root=DATASET,
        prediction_root=prediction_labels_dir,
        split=REQUESTED_EVALUATION_SPLIT,
    )
    write_csv(actual_save_dir / "downstream_per_image.csv", downstream_rows)
    write(actual_save_dir / "downstream_summary.json", downstream_summary)
    metrics = metrics_dict(result)
    box_metrics, mask_metrics = _metric_groups(metrics)

    checkpoint_after = checkpoint_record(checkpoint.resolve())
    if (
        checkpoint_after != contract["checkpoint"]
        or checkpoint.stat().st_mtime_ns != contract["checkpoint_mtime_ns"]
    ):
        raise RuntimeError("best.pt cambió durante la evaluación")
    environment_after = installed_distribution_snapshot()
    if environment_after != environment_before:
        added = sorted(set(environment_after["names"]) - set(environment_before["names"]))
        removed = sorted(set(environment_before["names"]) - set(environment_after["names"]))
        raise RuntimeError(
            "El entorno Python cambió durante la evaluación: "
            f"añadidas={added}, eliminadas={removed}, "
            f"before={environment_before['distribution_count']}, "
            f"after={environment_after['distribution_count']}"
        )

    sample_dir = DATASET / "images" / REQUESTED_EVALUATION_SPLIT
    samples = [str(path) for path in sorted(sample_dir.iterdir())[:12]]
    model.predict(
        source=samples,
        imgsz=int(config["imgsz"]),
        device=config["device"],
        save=True,
        project=str(OUTPUTS / "segmenter_evaluation"),
        name="yolo26n_seg_test_predictions",
        exist_ok=False,
        verbose=False,
    )
    write(
        Path(str(contract["summary_path"])),
        {
            "status": "passed",
            "requested_split": REQUESTED_EVALUATION_SPLIT,
            "run_name": contract["run_name"],
            "evaluated_split": evaluated_split,
            "split": REQUESTED_EVALUATION_SPLIT,
            "image_count": contract["image_count"],
            "instance_count": contract["instance_count"],
            "annotation_count": contract["annotation_count"],
            "loader_instance_count": contract["loader_instance_count"],
            "loader_deduplicated_annotations": contract["loader_deduplicated_annotations"],
            "loader_deduplicated_images": contract["loader_deduplicated_images"],
            "evaluated_image_count": evaluated_image_count,
            "evaluated_instance_count": evaluated_instance_count,
            "test_fingerprint": contract["test_fingerprint"],
            "dataset_yaml": contract["dataset_yaml"],
            "resolved_split_path": contract["resolved_split_path"],
            "save_dir": str(actual_save_dir),
            "prediction_labels_dir": str(prediction_labels_dir),
            "downstream_summary": downstream_summary,
            "checkpoint": checkpoint_after["path"],
            "checkpoint_sha256": checkpoint_after["sha256"],
            "checkpoint_size_bytes": checkpoint_after["size_bytes"],
            "metrics": metrics,
            "box_metrics": box_metrics,
            "mask_metrics": mask_metrics,
            "pilot_used": False,
            "prediction_samples": len(samples),
            "faster_coco_eval": EXPECTED_FASTER_COCO_EVAL,
            "environment_before": environment_before,
            "environment_after": environment_after,
            "environment_modified": False,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("smoke", "train", "resume", "evaluate"))
    result.add_argument("--config", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--active-manifest", type=Path)
    result.add_argument("--split", choices=("test",))
    result.add_argument("--reason", default="manual_authorized_resume")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode in {"smoke", "train", "resume"}:
        require_confirmation(args.mode)
    if args.mode in {"smoke", "train"}:
        if args.config is None:
            raise SystemExit("--config es obligatorio")
        train_mode(args.mode, args.config)
    elif args.mode == "resume":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint es obligatorio")
        resume_mode(args.checkpoint, args.reason, args.active_manifest)
    else:
        if args.checkpoint is None or args.config is None or args.split is None:
            raise SystemExit("--checkpoint, --config y --split son obligatorios")
        evaluate_mode(args.checkpoint, args.config, args.split)


if __name__ == "__main__":
    main()

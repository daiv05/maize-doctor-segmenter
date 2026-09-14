"""Read-only preflight for the frozen maize-leaf segmentation training dataset."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw

from src.data.loader import load_and_normalize_image
from src.data.segmentation_audit import (
    IMAGE_EXTENSIONS,
    parse_yolo_segmentation_line,
    sha256_file,
)
from src.data.segmentation_split import SPLITS, verify_parent_dataset

EXPECTED_IMAGES = {"train": 809, "val": 173, "test": 173}
EXPECTED_MASKS = {"train": 858, "val": 183, "test": 183}
EXPECTED_PARENT_FINGERPRINT = (
    "7a4a5c083fc64b067df12bcc95ec976d5a7e3b8a585d0a090b6b3940af4d7d5c"
)
EXPECTED_SPLIT_FINGERPRINTS = {
    "train": "06035eed94b920b9c7ad600d76eec132b93ade78ace1edb7d6a48340085d29ba",
    "val": "3c7bf7aba8a9f29b409c61bad4d9e9d59a3387915592f181ad3950ac8374e720",
    "test": "046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51",
}
EXPECTED_COMBINED_FINGERPRINT = (
    "96833e43a46c959f0d5c86615b1d1ea6deecb139063eea9c877986a61084c0e1"
)
EXPECTED_SEED = 42
EXPECTED_INITIAL_WEIGHTS_SHA256 = {
    "yolo26n-seg.pt": "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f",
}
EXPECTED_IMAGE_TOTAL = 1155
EXPECTED_MASK_TOTAL = 1224
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _candidate_package_version() -> str:
    requirement = (
        _PROJECT_ROOT / "cloud_training" / "requirements" / "ultralytics.in"
    )
    matches = [
        line.strip().removeprefix("ultralytics==")
        for line in requirement.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("ultralytics==")
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(f"Versión de Ultralytics inválida en {requirement}")
    return matches[0]


CANDIDATE_PACKAGE_VERSION = _candidate_package_version()
CANDIDATE_MODEL = "yolo26n-seg"
WEIGHTS_FILENAME = f"{CANDIDATE_MODEL}.pt"
CONFIG_FILENAME = f"{CANDIDATE_MODEL}.yaml"
ALLOWED_STATUSES = {
    "ready_for_training",
    "ready_for_remote_training",
    "blocked_by_missing_dependency",
    "blocked_by_missing_weights",
    "blocked_by_no_gpu",
    "blocked_by_model_incompatibility",
    "blocked_by_dataset_change",
    "blocked_by_validation_error",
}


class PreflightError(RuntimeError):
    """Raised for a hard preflight gate."""


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _split_digest(rows: Sequence[Mapping[str, str]], root: Path, split: str) -> str:
    digest = hashlib.sha256()
    selected = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["filename"].casefold(),
    )
    for row in selected:
        image_digest = sha256_file(root / row["materialized_image_path"])
        label_digest = sha256_file(root / row["materialized_label_path"])
        if image_digest != row["image_sha256"] or label_digest != row["label_sha256"]:
            raise PreflightError(f"Contenido materializado modificado: {row['filename']}")
        digest.update(row["filename"].encode())
        digest.update(b"\0")
        digest.update(image_digest.encode())
        digest.update(b"\0")
        digest.update(label_digest.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_training_locks(dataset_root: Path) -> dict[str, Any]:
    """Verify parent and split locks plus every materialized content fingerprint."""
    try:
        parent = verify_parent_dataset(dataset_root)
    except Exception as exc:
        raise PreflightError(str(exc)) from exc
    split_lock_path = dataset_root / "manifests" / "split_lock.json"
    if not split_lock_path.is_file():
        raise PreflightError(f"Falta split_lock.json: {split_lock_path}")
    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    if split_lock.get("status") != "ready_for_training_preflight":
        raise PreflightError(
            "split_lock.status debe ser ready_for_training_preflight; "
            f"actual={split_lock.get('status')!r}"
        )
    manifest_path = dataset_root / "manifests" / "split_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = {
        split: _split_digest(rows, dataset_root, split) for split in SPLITS
    }
    expected = {
        split: str(split_lock.get(f"{split}_fingerprint", "")) for split in SPLITS
    }
    changed = {
        split: {"expected": expected[split], "actual": actual[split]}
        for split in SPLITS
        if actual[split] != expected[split]
    }
    if changed:
        raise PreflightError(f"Fingerprint de split modificado: {changed}")
    return {
        "dataset_lock_status": parent["status"],
        "split_lock_status": split_lock["status"],
        "parent_fingerprint": parent["global_fingerprint"]["sha256"],
        "split_fingerprints": actual,
        "verified_manifest_rows": len(rows),
        "passed": True,
    }


def verify_cloud_training_payload(dataset_root: Path) -> dict[str, Any]:
    """Verify a cloud payload that intentionally excludes the frozen ``all/`` tree."""
    manifests = dataset_root / "manifests"
    parent = json.loads((manifests / "dataset_lock.json").read_text(encoding="utf-8"))
    split_lock = json.loads((manifests / "split_lock.json").read_text(encoding="utf-8"))
    if parent.get("status") != "ready_for_split_generation":
        raise PreflightError("dataset_lock no está listo")
    if split_lock.get("status") != "ready_for_training_preflight":
        raise PreflightError("split_lock no está listo")
    parent_fingerprint = str(parent.get("global_fingerprint", {}).get("sha256", ""))
    if parent_fingerprint != EXPECTED_PARENT_FINGERPRINT:
        raise PreflightError(
            "Fingerprint padre fuera del contrato congelado: "
            f"{parent_fingerprint} != {EXPECTED_PARENT_FINGERPRINT}"
        )
    if split_lock.get("parent_dataset_fingerprint") != EXPECTED_PARENT_FINGERPRINT:
        raise PreflightError("Fingerprint padre inválido en split_lock")
    if split_lock.get("seed") != EXPECTED_SEED:
        raise PreflightError(f"Seed inválido en split_lock: {split_lock.get('seed')!r}")
    for field, expected_value in (
        ("combined_fingerprint", EXPECTED_COMBINED_FINGERPRINT),
        ("actual_counts", EXPECTED_IMAGES),
        ("mask_counts", EXPECTED_MASKS),
        ("image_count", EXPECTED_IMAGE_TOTAL),
        ("mask_count", EXPECTED_MASK_TOTAL),
        ("parent_content_equivalent", True),
        ("training_performed", False),
        ("cross_split_duplicate_count", 0),
        ("cross_split_group_leakage_count", 0),
        ("cross_split_roboflow_variant_count", 0),
        ("cross_split_perceptual_count", 0),
        ("pilot_leakage_count", 0),
    ):
        if split_lock.get(field) != expected_value:
            raise PreflightError(
                f"split_lock.{field}={split_lock.get(field)!r} != {expected_value!r}"
            )
    frozen_fingerprints = json.loads(
        (manifests / "split_fingerprints.json").read_text(encoding="utf-8")
    )
    expected_fingerprint_manifest = {
        "parent_dataset_fingerprint": EXPECTED_PARENT_FINGERPRINT,
        "combined_fingerprint": EXPECTED_COMBINED_FINGERPRINT,
        **{
            f"{split}_fingerprint": fingerprint
            for split, fingerprint in EXPECTED_SPLIT_FINGERPRINTS.items()
        },
    }
    for field, expected_value in expected_fingerprint_manifest.items():
        if frozen_fingerprints.get(field) != expected_value:
            raise PreflightError(
                f"split_fingerprints.{field} fuera del contrato congelado"
            )
    with (manifests / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    row_counts = dict(Counter(row.get("split", "") for row in rows))
    if row_counts != EXPECTED_IMAGES or len(rows) != EXPECTED_IMAGE_TOTAL:
        raise PreflightError(
            f"Conteos del manifiesto modificados: {row_counts}, total={len(rows)}"
        )
    actual = {
        split: _split_digest(rows, dataset_root, split) for split in SPLITS
    }
    lock_fingerprints = {
        split: str(split_lock.get(f"{split}_fingerprint", "")) for split in SPLITS
    }
    if lock_fingerprints != EXPECTED_SPLIT_FINGERPRINTS:
        raise PreflightError(
            "Los fingerprints declarados no coinciden con el contrato congelado"
        )
    if actual != EXPECTED_SPLIT_FINGERPRINTS:
        raise PreflightError(
            "Contenido de splits modificado: "
            f"{actual} != {EXPECTED_SPLIT_FINGERPRINTS}"
        )
    return {
        "passed": True,
        "parent_fingerprint": parent_fingerprint,
        "split_fingerprints": actual,
        "combined_fingerprint": EXPECTED_COMBINED_FINGERPRINT,
        "seed": EXPECTED_SEED,
        "image_counts": row_counts,
        "mask_counts": EXPECTED_MASKS,
        "verified_manifest_rows": len(rows),
    }


def _nvidia_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "executable": executable,
        "driver_available": False,
        "driver_version": None,
        "gpu_name": None,
        "vram_total_bytes": None,
        "vram_free_bytes": None,
        "error": None,
    }
    if not executable:
        result["error"] = "nvidia-smi not found"
        return result
    command = [
        executable,
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        result["error"] = (completed.stderr or completed.stdout).strip()
        return result
    first = completed.stdout.strip().splitlines()[0].split(",")
    if len(first) >= 4:
        result.update(
            {
                "driver_available": True,
                "gpu_name": first[0].strip(),
                "vram_total_bytes": int(float(first[1].strip()) * 1024 * 1024),
                "vram_free_bytes": int(float(first[2].strip()) * 1024 * 1024),
                "driver_version": first[3].strip(),
            }
        )
    return result


def _memory_report() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "ram_total_bytes": values.get("MemTotal"),
        "ram_available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def audit_environment(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Collect environment, dependency and hardware facts without mutation.

    ``torch`` se importa aquí y no a nivel de módulo para que los targets de sólo
    lectura del Makefile no exijan la pila CUDA completa.
    """
    import torch

    nvidia = _nvidia_report()
    cuda_available = torch.cuda.is_available()
    disk = shutil.disk_usage(project_root)
    environment = {
        "operating_system": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "virtual_environment_active": sys.prefix != sys.base_prefix,
        "virtual_environment": sys.prefix if sys.prefix != sys.base_prefix else None,
        "pip": _version("pip"),
    }
    dependencies = {
        "torch": torch.__version__,
        "torchvision": _version("torchvision"),
        "ultralytics": _version("ultralytics"),
        "ultralytics_status": (
            "installed" if _version("ultralytics") else "missing_dependency"
        ),
        "candidate_install_command": (
            f'python -m pip install "ultralytics=={CANDIDATE_PACKAGE_VERSION}"'
        ),
        "installation_performed": False,
        "conflict_warning": (
            "Revisar el resolver antes de autorizar: el entorno usa "
            f"torch={torch.__version__}, torchvision={_version('torchvision')} y "
            f"CUDA compilada={torch.version.cuda}; pip podría sustituirlos."
        ),
    }
    hardware = {
        "cuda_compiled_in_torch": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "torch_gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "nvidia": nvidia,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "cpu_count": os.cpu_count(),
        **_memory_report(),
    }
    return environment, dependencies, hardware


def audit_candidate_model(project_root: Path) -> dict[str, Any]:
    """Audit only local Ultralytics/model evidence; never trigger a download."""
    installed = _version("ultralytics")
    weight_matches = sorted(project_root.rglob(WEIGHTS_FILENAME))
    config_matches = sorted(project_root.rglob(CONFIG_FILENAME))
    result: dict[str, Any] = {
        "candidate": CANDIDATE_MODEL,
        "weights_filename": WEIGHTS_FILENAME,
        "configuration_filename": CONFIG_FILENAME,
        "task": "segment",
        "ultralytics_version": installed,
        "recognized_by_installed_version": False,
        "training_compatible": "not_locally_verifiable",
        "export_compatible": "not_locally_verifiable",
        "configuration_available_locally": bool(config_matches),
        "configuration_paths": [str(path) for path in config_matches],
        "weights_available_locally": bool(weight_matches),
        "weights_paths": [str(path) for path in weight_matches],
        "weights_download_required": not bool(weight_matches),
        "weights_downloaded": False,
        "license": (
            "Ultralytics AGPL-3.0 or Enterprise; verify the selected distribution "
            "terms before authorized training"
        ),
        "license_locally_verified": False,
        "supported_segmentation_alternatives": [],
        "recommended_alternative": None,
        "forward_pass": {
            "executed": False,
            "status": "blocked_by_missing_dependency"
            if not installed
            else "blocked_by_missing_weights",
            "memory_used_bytes": None,
            "segmentation_output_verified": False,
        },
    }
    if not installed:
        result["compatibility_status"] = "blocked_by_missing_dependency"
        result["reason"] = (
            "Ultralytics no está instalado; no se puede afirmar soporte de "
            "YOLO26n-seg ni enumerar alternativas de esa versión."
        )
        return result

    spec = importlib.util.find_spec("ultralytics")
    package_root = Path(spec.origin).parent if spec and spec.origin else None
    bundled_configs = (
        sorted(package_root.rglob(CONFIG_FILENAME)) if package_root else []
    )
    alternatives = (
        sorted({path.stem for path in package_root.rglob("*[ns]-seg.yaml")})
        if package_root
        else []
    )
    result["configuration_paths"] = [
        *result["configuration_paths"],
        *(str(path) for path in bundled_configs),
    ]
    result["configuration_available_locally"] = bool(result["configuration_paths"])
    result["recognized_by_installed_version"] = bool(bundled_configs)
    result["supported_segmentation_alternatives"] = alternatives
    result["recommended_alternative"] = next(
        (name for name in alternatives if name.endswith("n-seg")), None
    )
    if not bundled_configs:
        result["compatibility_status"] = "blocked_by_model_incompatibility"
        result["reason"] = f"{CONFIG_FILENAME} no existe en la versión instalada"
    elif not weight_matches:
        result["compatibility_status"] = "blocked_by_missing_weights"
        result["reason"] = f"{WEIGHTS_FILENAME} no está disponible localmente"
        result["training_compatible"] = True
        result["export_compatible"] = True
    else:
        result["compatibility_status"] = "locally_available"
        result["reason"] = "Configuración y pesos existen localmente"
        result["training_compatible"] = True
        result["export_compatible"] = True
        result["forward_pass"]["status"] = "not_executed_by_static_audit"
    try:
        result["license"] = metadata.metadata("ultralytics").get(
            "License", result["license"]
        )
        result["license_locally_verified"] = True
    except metadata.PackageNotFoundError:
        pass
    return result


def _load_dataset_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"dataset.yaml inválido: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreflightError("dataset.yaml debe ser un mapping")
    return payload


def validate_segmentation_dataset(dataset_root: Path) -> dict[str, Any]:
    """Fully validate portable paths, pairs, polygons, classes and expected counts."""
    errors: list[str] = []
    yaml_path = dataset_root / "dataset.yaml"
    try:
        config = _load_dataset_yaml(yaml_path)
    except PreflightError as exc:
        return {"passed": False, "errors": [str(exc)]}
    if config.get("names") != {0: "maize_leaf"}:
        errors.append(f"names inválido: {config.get('names')!r}")
    base = Path(str(config.get("path", ".")))
    if base.is_absolute() or ".." in base.parts:
        errors.append("path debe ser relativo y portable")
    counts: dict[str, int] = {}
    masks: dict[str, int] = {}
    bbox_mixed = 0
    class_counts: Counter[int] = Counter()
    for split in SPLITS:
        relative = Path(str(config.get(split, "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            errors.append(f"Ruta {split} no es relativa/portable: {relative}")
            continue
        image_dir = dataset_root / base / relative
        label_dir = dataset_root / base / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"Rutas inexistentes para {split}")
            continue
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        counts[split] = len(images)
        if len(images) != EXPECTED_IMAGES[split]:
            errors.append(f"{split}: imágenes={len(images)} != {EXPECTED_IMAGES[split]}")
        if {path.stem for path in images} != {path.stem for path in labels}:
            errors.append(f"{split}: correspondencia imagen/TXT inválida")
        mask_count = 0
        for label in labels:
            lines = label.read_text(encoding="utf-8").splitlines()
            if not lines or any(not line.strip() for line in lines):
                errors.append(f"TXT vacío o con línea vacía: {label}")
                continue
            for line_number, line in enumerate(lines, start=1):
                parsed = parse_yolo_segmentation_line(line)
                if parsed.annotation_format == "yolo_bbox":
                    bbox_mixed += 1
                if not parsed.valid:
                    issues = ",".join(str(row["issue_type"]) for row in parsed.issues)
                    errors.append(f"{label}:{line_number}: {issues}")
                if parsed.class_id is not None:
                    class_counts[parsed.class_id] += 1
                mask_count += 1
        masks[split] = mask_count
        if mask_count != EXPECTED_MASKS[split]:
            errors.append(f"{split}: máscaras={mask_count} != {EXPECTED_MASKS[split]}")
    if set(class_counts) != {0}:
        errors.append(f"Clases encontradas: {dict(class_counts)}")
    if bbox_mixed:
        errors.append(f"Bounding boxes mezclados: {bbox_mixed}")
    return {
        "passed": not errors,
        "dataset_yaml": str(yaml_path),
        "portable_paths": not any("relativ" in error for error in errors),
        "image_counts": counts,
        "mask_counts": masks,
        "class_counts": dict(class_counts),
        "bbox_mixed_count": bbox_mixed,
        "errors": errors,
    }


def _letterbox(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    scale = min(size / image.width, size / image.height)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized_image = image.resize(resized_size, Image.Resampling.BILINEAR)
    resized_mask = mask.resize(resized_size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    mask_canvas = Image.new("L", (size, size), 0)
    offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
    canvas.paste(resized_image, offset)
    mask_canvas.paste(resized_mask, offset)
    return canvas, mask_canvas


def run_loader_smoke_test(
    dataset_root: Path,
    preview_root: Path,
    *,
    image_size: int = 640,
    seed: int = 42,
) -> dict[str, Any]:
    """Load 4/2/2 samples, rasterize masks and create one finite tensor batch."""
    import torch

    manifest = dataset_root / "manifests" / "split_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requested = {"train": 4, "val": 2, "test": 2}
    selected: list[dict[str, str]] = []
    for split in SPLITS:
        candidates = [row for row in rows if row["split"] == split]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['filename']}".encode()
            ).hexdigest()
        )
        selected.extend(candidates[: requested[split]])
    if preview_root.exists():
        shutil.rmtree(preview_root)
    preview_root.mkdir(parents=True)
    image_tensors: list[torch.Tensor] = []
    mask_tensors: list[torch.Tensor] = []
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        image_path = dataset_root / row["materialized_image_path"]
        label_path = dataset_root / row["materialized_label_path"]
        image = load_and_normalize_image(str(image_path))
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        instance_count = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_segmentation_line(line)
            if not parsed.valid:
                raise PreflightError(f"Smoke label inválido: {label_path}")
            points = [
                (round(x * image.width), round(y * image.height)) for x, y in parsed.points
            ]
            draw.polygon(points, fill=255)
            instance_count += 1
        if not mask.getbbox():
            raise PreflightError(f"Máscara no rasterizable: {label_path}")
        batch_image, batch_mask = _letterbox(image, mask, image_size)
        image_array = np.asarray(batch_image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(batch_mask, dtype=np.float32) > 0).astype(np.float32)
        image_tensors.append(torch.from_numpy(image_array).permute(2, 0, 1))
        mask_tensors.append(torch.from_numpy(mask_array).unsqueeze(0))
        overlay = batch_image.convert("RGBA")
        color = Image.new("RGBA", overlay.size, (20, 255, 80, 0))
        color.putalpha(batch_mask.point(lambda value: 90 if value else 0))
        preview = Image.alpha_composite(overlay, color).convert("RGB")
        preview_path = preview_root / f"{index:02d}_{row['split']}_{row['filename']}"
        preview.save(preview_path, quality=90)
        samples.append(
            {
                "split": row["split"],
                "filename": row["filename"],
                "original_shape": [image.height, image.width, 3],
                "instances": instance_count,
                "rasterized_pixels": int(mask_array.sum()),
                "preview": str(preview_path),
            }
        )
    images = torch.stack(image_tensors)
    masks = torch.stack(mask_tensors)
    finite = bool(torch.isfinite(images).all() and torch.isfinite(masks).all())
    return {
        "passed": finite and len(samples) == 8,
        "seed": seed,
        "image_size": image_size,
        "sample_counts": dict(Counter(row["split"] for row in selected)),
        "samples": samples,
        "image_batch_shape": list(images.shape),
        "mask_batch_shape": list(masks.shape),
        "image_dtype": str(images.dtype),
        "mask_dtype": str(masks.dtype),
        "finite": finite,
        "nan_count": int(torch.isnan(images).sum() + torch.isnan(masks).sum()),
        "infinite_count": int(torch.isinf(images).sum() + torch.isinf(masks).sum()),
        "optimizer_steps": 0,
        "epochs_run": 0,
    }


def recommended_configuration(hardware: Mapping[str, Any]) -> dict[str, Any]:
    """Return CPU smoke and VRAM-autotuned remote proposals; execute neither."""
    workers = min(8, max(1, int(hardware.get("cpu_count") or 1) // 2))
    common = {
        "model": WEIGHTS_FILENAME,
        "data": "data/leaf_detection/detector_dataset/dataset.yaml",
        "task": "segment",
        "imgsz": 640,
        "epochs": 150,
        "patience": 30,
        "optimizer": "auto",
        "seed": 42,
        "deterministic": True,
        "project": "outputs/leaf_detection/segmentation_training",
    }
    return {
        "local_conservative": {
            **common,
            "batch": 1,
            "device": "cpu",
            "workers": min(2, workers),
            "cache": False,
            "name": "yolo26n_seg_local_smoke_seed42",
            "purpose": "smoke_only_no_training_recommended_without_cuda",
        },
        "remote_recommended": {
            **common,
            "batch": -1,
            "batch_reason": "Ultralytics AutoBatch; determine from remote VRAM",
            "device": 0,
            "workers": workers,
            "cache": "disk",
            "name": "yolo26n_seg_full_seed42",
            "minimum_recommended_vram_gib": 12,
        },
    }


def training_command(config: Mapping[str, Any]) -> str:
    """Render a guarded command; this function never executes it."""
    values = config["remote_recommended"]
    arguments = " ".join(
        (
            f"model={values['model']}",
            f"data={values['data']}",
            "task=segment",
            f"imgsz={values['imgsz']}",
            f"batch={values['batch']}",
            f"epochs={values['epochs']}",
            f"patience={values['patience']}",
            "seed=42",
            "deterministic=True",
            f"device={values['device']}",
            f"workers={values['workers']}",
            f"cache={values['cache']}",
            f"optimizer={values['optimizer']}",
            f"project={values['project']}",
            f"name={values['name']}",
        )
    )
    return (
        'test "${CONFIRM_SEGMENTATION_TRAINING:-0}" = "1" || '
        '{ echo "Entrenamiento bloqueado: use CONFIRM_SEGMENTATION_TRAINING=1"; exit 2; }\n'
        f"yolo segment train {arguments}\n"
    )


def require_training_confirmation(environment: Mapping[str, str] | None = None) -> None:
    """Reusable guard for any future segmentation training entry point."""
    values = environment if environment is not None else os.environ
    if values.get("CONFIRM_SEGMENTATION_TRAINING") != "1":
        raise PreflightError(
            "Entrenamiento bloqueado: use CONFIRM_SEGMENTATION_TRAINING=1"
        )


def _memory_estimate(smoke: Mapping[str, Any], hardware: Mapping[str, Any]) -> dict[str, Any]:
    images = int(np.prod(smoke["image_batch_shape"])) * 4
    masks = int(np.prod(smoke["mask_batch_shape"])) * 4
    per_sample = (images + masks) // int(smoke["image_batch_shape"][0])
    return {
        "smoke_input_tensor_bytes": images + masks,
        "estimated_input_bytes_per_sample": per_sample,
        "estimated_input_bytes_batch_16": per_sample * 16,
        "model_activation_memory": "not_measured_model_unavailable",
        "local_vram_total_bytes": hardware["nvidia"]["vram_total_bytes"],
        "local_vram_free_bytes": hardware["nvidia"]["vram_free_bytes"],
        "batch_decision": (
            "No se fija batch de entrenamiento local sin GPU; local=1 sólo para smoke, "
            "remoto=-1 para AutoBatch según VRAM."
        ),
    }


def run_segmentation_preflight(
    *,
    project_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run all read-only gates and persist the requested preflight evidence."""
    output_root.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    try:
        locks = verify_training_locks(dataset_root)
    except PreflightError as exc:
        locks = {"passed": False, "error": str(exc)}
        blockers.append("dataset_change")
    environment, dependencies, hardware = audit_environment(project_root)
    dataset = validate_segmentation_dataset(dataset_root)
    if not dataset["passed"]:
        blockers.append("dataset_validation")
    model = audit_candidate_model(project_root)
    if dependencies["ultralytics"] is None:
        blockers.append("missing_dependency")
    if model["compatibility_status"] == "blocked_by_model_incompatibility":
        blockers.append("model_incompatibility")
    if model["weights_download_required"]:
        blockers.append("missing_weights")
    if not hardware["cuda_available"]:
        blockers.append("no_gpu")
    smoke = (
        run_loader_smoke_test(dataset_root, output_root / "previews")
        if locks.get("passed") and dataset["passed"]
        else {"passed": False, "skipped": "dataset gate failed", "epochs_run": 0}
    )
    if not smoke["passed"]:
        blockers.append("loader_validation")
    config = recommended_configuration(hardware)
    command = training_command(config)
    memory = _memory_estimate(smoke, hardware) if smoke["passed"] else {}
    if "dataset_change" in blockers:
        status = "blocked_by_dataset_change"
    elif "dataset_validation" in blockers or "loader_validation" in blockers:
        status = "blocked_by_validation_error"
    elif "missing_dependency" in blockers:
        status = "blocked_by_missing_dependency"
    elif "model_incompatibility" in blockers:
        status = "blocked_by_model_incompatibility"
    elif "missing_weights" in blockers:
        status = "blocked_by_missing_weights"
    elif "no_gpu" in blockers:
        status = "ready_for_remote_training"
    else:
        status = "ready_for_training"
    if status not in ALLOWED_STATUSES:
        raise RuntimeError(f"Estado inesperado: {status}")
    safety = {
        "training_started": False,
        "models_built": 0,
        "forward_passes": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "epochs_run": 0,
        "checkpoints_written": 0,
        "weights_downloaded": 0,
        "internet_used": False,
        "dependencies_installed": 0,
    }
    summary = {
        "schema_version": 1,
        "status": status,
        "remote_readiness": (
            "blocked_until_dependency_and_weights_are_authorized"
            if "missing_dependency" in blockers
            else "ready_for_remote_training"
        ),
        "blockers": blockers,
        "locks_verified": bool(locks.get("passed")),
        "dataset_valid": bool(dataset["passed"]),
        "loader_smoke_valid": bool(smoke["passed"]),
        "model_forward_executed": False,
        "safety": safety,
    }
    _json(output_root / "summary.json", summary)
    _json(output_root / "environment.json", environment)
    _json(output_root / "dependency_report.json", dependencies)
    _json(output_root / "hardware_report.json", hardware)
    _json(output_root / "dataset_check.json", {**dataset, "locks": locks})
    _json(output_root / "model_compatibility.json", model)
    _json(output_root / "loader_smoke_test.json", smoke)
    _json(output_root / "memory_estimate.json", memory)
    (output_root / "recommended_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_root / "training_command.txt").write_text(command, encoding="utf-8")
    return summary

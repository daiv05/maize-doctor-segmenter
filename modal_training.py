"""Plano de control Modal para el segmentador de hojas de maíz.

El código viaja en la imagen y el dataset vive en un Volume sembrado una sola vez desde
Hugging Face, de modo que iterar código no obligue a resubir gigabytes. La garantía del
dataset no la da el transporte sino ``verify_cloud_training_payload``, que recalcula los
fingerprints congelados sobre el árbol montado antes de cada operación.

    modal run modal_training.py::seed_dataset
    modal run modal_training.py::verify_dataset
    modal run modal_training.py::preflight

Las funciones de entrenamiento exigen el argumento literal ``--confirm true``.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "doctor-maiz-leaf-segmentation"
REPO_ANCHOR = "/root"
DATASET_VOLUME_NAME = "doctor-maiz-leaf-segmentation-data"
OUTPUTS_VOLUME_NAME = "doctor-maiz-leaf-segmentation-outputs"
DATASET_MOUNT_PATH = "/data"
OUTPUTS_MOUNT_PATH = "/outputs"
DATASET_MOUNT = Path(DATASET_MOUNT_PATH)
OUTPUTS_MOUNT = Path(OUTPUTS_MOUNT_PATH)
DATASET_ROOT = DATASET_MOUNT / "leaf_detection" / "detector_dataset"
SEGMENTATION_OUTPUT_ROOT = OUTPUTS_MOUNT / "leaf_detection"
MODAL_RUNTIME_ROOT = SEGMENTATION_OUTPUT_ROOT / "modal_runtime"
INITIAL_WEIGHTS_NAME = "yolo26n-seg.pt"
PERSISTENT_INITIAL_WEIGHTS = (
    SEGMENTATION_OUTPUT_ROOT / "initial_weights" / INITIAL_WEIGHTS_NAME
)
HF_DATASET_REPO = os.getenv(
    "HF_SEGMENTATION_DATASET_REPO",
    "daiv05/corn-leaf-instance-segmentation",
)
EXPECTED_PARENT_FINGERPRINT = "7a4a5c083fc64b067df12bcc95ec976d5a7e3b8a585d0a090b6b3940af4d7d5c"
EXPECTED_TEST_FINGERPRINT = "046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51"
DATASET_MARKER = MODAL_RUNTIME_ROOT / "dataset_verification.json"

BASE_IMAGE_TAG = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
BASE_IMAGE_DIGEST = "sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"
BASE_IMAGE = f"{BASE_IMAGE_TAG}@{BASE_IMAGE_DIGEST}"
EXPECTED_PYTHON = "3.11"
EXPECTED_TORCH = "2.6.0"
EXPECTED_TORCHVISION = "0.21.0"
EXPECTED_ULTRALYTICS = "8.4.104"
EXPECTED_FASTER_COCO_EVAL = "1.7.2"
EXPECTED_CUDA = "12.4"
EXPECTED_CUDA_LOCAL = "cu124"
IMAGE_SYSTEM_PACKAGES = (
    "bash",
    "coreutils",
    "git",
    "libgl1",
    "libglib2.0-0",
    "make",
    "procps",
    "tar",
)
IMAGE_PYTHON_PACKAGES = (
    "filelock==3.18.0",
    f"faster-coco-eval=={EXPECTED_FASTER_COCO_EVAL}",
    "huggingface-hub==1.30.0",
    "matplotlib==3.10.3",
    "numpy==1.26.4",
    "nvidia-ml-py==12.575.51",
    "opencv-python==4.11.0.86",
    "pandas==2.3.1",
    "pillow==11.2.1",
    "polars==1.31.0",
    "psutil==7.0.0",
    "python-dotenv==1.1.1",
    "pyyaml==6.0.2",
    "requests==2.32.4",
    "tqdm==4.67.1",
    "ultralytics-thop==2.0.18",
    f"ultralytics=={EXPECTED_ULTRALYTICS}",
)
IMAGE_BUILD_LOCK_PATH = "/opt/doctor_maiz_modal_image.lock"
IMAGE_BUILD_LOCK = Path(IMAGE_BUILD_LOCK_PATH)
IMAGE_RECIPE = {
    "base_image": BASE_IMAGE,
    "base_image_tag": BASE_IMAGE_TAG,
    "base_image_digest": BASE_IMAGE_DIGEST,
    "python": EXPECTED_PYTHON,
    "torch": EXPECTED_TORCH,
    "torchvision": EXPECTED_TORCHVISION,
    "cuda": EXPECTED_CUDA,
    "cuda_local": EXPECTED_CUDA_LOCAL,
    "system_packages": IMAGE_SYSTEM_PACKAGES,
    "python_packages": IMAGE_PYTHON_PACKAGES,
}
IMAGE_RECIPE_SHA256 = hashlib.sha256(
    json.dumps(IMAGE_RECIPE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

ALLOWED_GPUS = ("A10", "L4", "A100")
REQUESTED_GPU = os.getenv("DOCTOR_MAIZ_MODAL_GPU", "A10").strip().upper()
if REQUESTED_GPU not in ALLOWED_GPUS:
    raise ValueError(
        f"DOCTOR_MAIZ_MODAL_GPU={REQUESTED_GPU!r} no permitido; use uno de {ALLOWED_GPUS}"
    )
MINIMUM_VRAM_BYTES = 12 * 1024**3

TRAINABLE_EXPERIMENTS = {
    "d01_mosaic0_seed42": "d01_mosaic0_seed42.yaml",
    "d02_imgsz512_seed42": "d02_imgsz512_seed42.yaml",
    "d03_source_balanced_seed42": "d03_source_balanced_seed42.yaml",
    "d05_scratch_seed42": "d05_scratch_seed42.yaml",
    "d06_copy_paste_seed42": "d06_copy_paste_seed42.yaml",
    "e01_baseline_seed7": "e01_baseline_seed7.yaml",
    "e01_baseline_seed1337": "e01_baseline_seed1337.yaml",
}


def _validate_image_versions(
    actual: dict[str, str | None],
    python_version: tuple[int, ...],
) -> None:
    """Comprueba que la imagen resuelta coincide exactamente con la receta declarada."""
    from packaging.version import InvalidVersion, Version

    expected_python = tuple(int(part) for part in EXPECTED_PYTHON.split("."))
    if python_version[:2] != expected_python:
        raise RuntimeError(
            f"python={actual.get('python')!r}; versión esperada {EXPECTED_PYTHON}.x"
        )

    expected_distributions = {
        "faster-coco-eval": EXPECTED_FASTER_COCO_EVAL,
        "torch": EXPECTED_TORCH,
        "torchvision": EXPECTED_TORCHVISION,
        "ultralytics": EXPECTED_ULTRALYTICS,
    }
    for name, expected in expected_distributions.items():
        installed = actual.get(name)
        if not installed:
            raise RuntimeError(f"{name} no reportó una versión; esperado {expected}")
        try:
            installed_release = Version(installed).release
        except InvalidVersion as exc:
            raise RuntimeError(f"{name}={installed!r} no es una versión válida") from exc
        expected_release = Version(expected).release
        if installed_release != expected_release:
            raise RuntimeError(
                f"{name}={installed!r} tiene release {installed_release}; "
                f"esperado {expected_release}"
            )

    for name in ("torch_import", "torchvision_import"):
        imported = actual.get(name)
        if not imported:
            raise RuntimeError(f"{name} no reportó una versión importada")
        try:
            local = Version(imported).local
        except InvalidVersion as exc:
            raise RuntimeError(f"{name}={imported!r} no es una versión válida") from exc
        if local != EXPECTED_CUDA_LOCAL:
            raise RuntimeError(
                f"{name}={imported!r} no contiene el sufijo local "
                f"+{EXPECTED_CUDA_LOCAL}"
            )

    if actual.get("torch_cuda") != EXPECTED_CUDA:
        raise RuntimeError(
            f"torch.version.cuda={actual.get('torch_cuda')!r}; "
            f"esperado {EXPECTED_CUDA!r}"
        )


def _validate_modal_image_versions() -> None:
    """Valida la imagen durante su construcción, antes de cualquier ejecución."""
    from importlib import metadata

    import torch
    import torchvision

    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": metadata.version("torch"),
        "torch_import": str(torch.__version__),
        "torchvision": metadata.version("torchvision"),
        "torchvision_import": str(torchvision.__version__),
        "ultralytics": metadata.version("ultralytics"),
        "faster-coco-eval": metadata.version("faster-coco-eval"),
        "torch_cuda": torch.version.cuda,
    }
    print("Modal image version check:", flush=True)
    for name, version in actual.items():
        print(f"  {name}: {version}", flush=True)
    _validate_image_versions(actual, tuple(sys.version_info[:3]))


modal_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install(*IMAGE_SYSTEM_PACKAGES)
    .pip_install(*IMAGE_PYTHON_PACKAGES)
    .run_commands(
        "python -m pip check",
        f"python -m pip freeze | LC_ALL=C sort > {IMAGE_BUILD_LOCK_PATH}",
    )
    .run_function(_validate_modal_image_versions)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "YOLO_AUTOINSTALL": "false",
            "YOLO_OFFLINE": "true",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "PROJECT_DATA_ROOT": DATASET_MOUNT_PATH,
            "OUTPUT_ROOT": OUTPUTS_MOUNT_PATH,
            "HF_SEGMENTATION_DATASET_REPO": HF_DATASET_REPO,
        }
    )
    .add_local_file("Makefile", f"{REPO_ANCHOR}/Makefile", copy=True)
    .add_local_file("pyproject.toml", f"{REPO_ANCHOR}/pyproject.toml", copy=True)
    .add_local_dir("config", f"{REPO_ANCHOR}/config", copy=True)
    .add_local_dir("cloud_training", f"{REPO_ANCHOR}/cloud_training", copy=True)
    .add_local_python_source("src", "scripts")
)

app = modal.App(APP_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNTS: dict[Any, Any] = {
    DATASET_MOUNT_PATH: dataset_volume,
    OUTPUTS_MOUNT_PATH: outputs_volume,
}
HF_SECRET = modal.Secret.from_name("hf")


def _utc_now() -> str:
    """Devuelve el instante actual en UTC ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    """Calcula el SHA-256 de un archivo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe un JSON de forma atómica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
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


def _modal_object_id(handle: object) -> str | None:
    """Devuelve el identificador de un objeto Modal cuando está disponible."""
    try:
        value = getattr(handle, "object_id")
    except AttributeError:
        return None
    return str(value) if value else None


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> None:
    """Ejecuta un comando registrando la línea exacta invocada."""
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _project_environment() -> dict[str, str]:
    """Construye el entorno que resuelve raíces y binarios dentro del contenedor."""
    environment = os.environ.copy()
    pythonpath = [
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry != REPO_ANCHOR
    ]
    pythonpath.insert(0, REPO_ANCHOR)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    environment.update(
        {
            "PYTHON": sys.executable,
            "CLOUD_TRAINING_DIR": f"{REPO_ANCHOR}/cloud_training",
            "PROJECT_DATA_ROOT": DATASET_MOUNT_PATH,
            "OUTPUT_ROOT": OUTPUTS_MOUNT_PATH,
            "LEAF_SEGMENTATION_DATASET": str(DATASET_ROOT),
            "LEAF_SEGMENTATION_OUTPUT": str(SEGMENTATION_OUTPUT_ROOT),
            "SEGMENTATION_MODEL": "yolo26n-seg.pt",
            "SEGMENTATION_DEVICE": "0",
        }
    )
    return environment


def _nvidia_smi() -> dict[str, Any]:
    """Consulta el estado de la GPU asignada."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    if len(values) != 5:
        raise RuntimeError(f"Salida inesperada de nvidia-smi: {completed.stdout!r}")
    return {
        "name": values[0],
        "memory_total_mib": int(values[1]),
        "memory_free_mib": int(values[2]),
        "utilization_gpu_percent": int(values[3]),
        "driver_version": values[4],
    }


def _runtime_report(operation: str, *, require_gpu: bool) -> dict[str, Any]:
    """Registra la identidad del runtime remoto dentro del árbol descargable."""
    import torch
    import torchvision

    MODAL_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    build_lock = (
        IMAGE_BUILD_LOCK.read_text(encoding="utf-8") if IMAGE_BUILD_LOCK.is_file() else ""
    )
    actual_versions = {
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "torch_import": str(torch.__version__),
        "torchvision": metadata.version("torchvision"),
        "torchvision_import": str(torchvision.__version__),
        "ultralytics": metadata.version("ultralytics"),
        "faster-coco-eval": metadata.version("faster-coco-eval"),
        "torch_cuda": torch.version.cuda,
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "checking",
        "operation": operation,
        "utc": _utc_now(),
        "app": APP_NAME,
        "dataset_volume": DATASET_VOLUME_NAME,
        "outputs_volume": OUTPUTS_VOLUME_NAME,
        "dataset_mount": DATASET_MOUNT_PATH,
        "outputs_mount": OUTPUTS_MOUNT_PATH,
        "function_call_id": modal.current_function_call_id(),
        "input_id": modal.current_input_id(),
        "requested_gpu": REQUESTED_GPU,
        "allowed_gpus": list(ALLOWED_GPUS),
        "base_image": BASE_IMAGE,
        "base_image_tag": BASE_IMAGE_TAG,
        "base_image_digest": BASE_IMAGE_DIGEST,
        "modal_image_id": _modal_object_id(modal_image),
        "modal_dataset_volume_id": _modal_object_id(dataset_volume),
        "modal_outputs_volume_id": _modal_object_id(outputs_volume),
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "image_build_lock_sha256": (
            hashlib.sha256(build_lock.encode()).hexdigest() if build_lock else None
        ),
        "python": actual_versions["python"],
        "python_executable": sys.executable,
        "torch": actual_versions["torch"],
        "torch_import": actual_versions["torch_import"],
        "torchvision": actual_versions["torchvision"],
        "torchvision_import": actual_versions["torchvision_import"],
        "cuda_compiled": actual_versions["torch_cuda"],
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": actual_versions["ultralytics"],
        "faster_coco_eval": actual_versions["faster-coco-eval"],
        "cuda_available": torch.cuda.is_available(),
        "gpu": None,
    }
    errors: list[str] = []
    try:
        _validate_image_versions(actual_versions, tuple(sys.version_info[:3]))
    except RuntimeError as exc:
        errors.append(str(exc))
    if require_gpu:
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() es false")
        else:
            properties = torch.cuda.get_device_properties(0)
            nvidia = _nvidia_smi()
            report["gpu"] = {
                "requested": REQUESTED_GPU,
                "received": torch.cuda.get_device_name(0),
                "total_vram_bytes": properties.total_memory,
                "free_vram_bytes": torch.cuda.mem_get_info(0)[0],
                "initial_utilization_percent": nvidia["utilization_gpu_percent"],
                "nvidia_smi": nvidia,
            }
            if REQUESTED_GPU not in str(report["gpu"]["received"]).upper():
                errors.append(
                    f"GPU recibida no coincide con {REQUESTED_GPU}: {report['gpu']['received']}"
                )
            if properties.total_memory < MINIMUM_VRAM_BYTES:
                errors.append(
                    f"VRAM insuficiente: {properties.total_memory} < {MINIMUM_VRAM_BYTES}"
                )
    report["errors"] = errors
    report["status"] = "ready" if not errors else "blocked"

    call_id = report["function_call_id"] or report["input_id"] or "unknown"
    _write_json(MODAL_RUNTIME_ROOT / f"{operation}_{call_id}.json", report)
    _write_json(MODAL_RUNTIME_ROOT / f"{operation}_latest.json", report)
    lock_lines = {
        "base_image": BASE_IMAGE,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "python": report["python"],
        "torch": report["torch"],
        "torchvision": report["torchvision"],
        "cuda": report["cuda_compiled"],
        "cudnn": report["cudnn"],
        "ultralytics": report["ultralytics"],
        "faster_coco_eval": report["faster_coco_eval"],
        "requested_gpu": REQUESTED_GPU,
    }
    (MODAL_RUNTIME_ROOT / "runtime_environment.modal.lock").write_text(
        "".join(f"{key}={value}\n" for key, value in lock_lines.items()),
        encoding="utf-8",
    )
    if operation == "preflight":
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            capture_output=True,
            check=True,
        )
        (MODAL_RUNTIME_ROOT / "pip_freeze.txt").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        if build_lock:
            (MODAL_RUNTIME_ROOT / "image_build_pip_freeze.txt").write_text(
                build_lock,
                encoding="utf-8",
            )
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def _make(target: str, *variables: str) -> None:
    """Invoca un target del Makefile del proyecto contra los montajes remotos."""
    command = [
        "make",
        target,
        f"PYTHON={sys.executable}",
        f"CLOUD_TRAINING_DIR={REPO_ANCHOR}/cloud_training",
        f"LEAF_SEGMENTATION_DATASET={DATASET_ROOT}",
        f"LEAF_SEGMENTATION_OUTPUT={SEGMENTATION_OUTPUT_ROOT}",
        "SEGMENTATION_MODEL=yolo26n-seg.pt",
        "SEGMENTATION_DEVICE=0",
        *variables,
    ]
    _run(command, cwd=Path(REPO_ANCHOR), environment=_project_environment())


def _verify_mounted_dataset() -> dict[str, Any]:
    """Recalcula los fingerprints congelados sobre el dataset montado.

    @returns {dict[str, Any]} Reporte de locks verificados.
    """
    from src.training.segmentation_preflight import verify_cloud_training_payload

    if not DATASET_ROOT.is_dir():
        raise RuntimeError(
            f"Falta el dataset en {DATASET_ROOT}; ejecute modal_training.py::seed_dataset"
        )
    locks = verify_cloud_training_payload(DATASET_ROOT)
    if locks["parent_fingerprint"] != EXPECTED_PARENT_FINGERPRINT:
        raise RuntimeError("El fingerprint padre montado no es el congelado")
    if locks["split_fingerprints"]["test"] != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError("El fingerprint de test montado no es el congelado")
    return locks


def _dataset_gate() -> dict[str, Any]:
    """Verifica el dataset y persiste el marcador de la verificación."""
    locks = _verify_mounted_dataset()
    marker = {
        "schema_version": 1,
        "status": "verified",
        "utc": _utc_now(),
        "dataset_root": str(DATASET_ROOT),
        "hf_dataset_repo": HF_DATASET_REPO,
        **locks,
    }
    _write_json(DATASET_MARKER, marker)
    return marker


def _restore_initial_weights() -> Path | None:
    """Repone en el directorio de trabajo los pesos iniciales guardados en el Volume.

    El directorio de trabajo es la imagen, que es efímera: sin esta reposición cada
    contenedor tendría que volver a descargar los pesos y ``weights_manifest.json``
    apuntaría a una ruta inexistente en la operación siguiente.

    @returns {Path|None} Ruta repuesta, o None si aún no hay copia persistente.
    """
    if not PERSISTENT_INITIAL_WEIGHTS.is_file():
        return None
    working = Path(REPO_ANCHOR) / INITIAL_WEIGHTS_NAME
    if working.is_file() and _sha256(working) == _sha256(PERSISTENT_INITIAL_WEIGHTS):
        return working
    shutil.copy2(PERSISTENT_INITIAL_WEIGHTS, working)
    return working


def _persist_initial_weights() -> Path | None:
    """Guarda en el Volume los pesos iniciales que la operación haya descargado.

    @returns {Path|None} Ruta persistida, o None si no hay pesos que guardar.
    """
    working = Path(REPO_ANCHOR) / INITIAL_WEIGHTS_NAME
    if not working.is_file():
        return None
    if (
        PERSISTENT_INITIAL_WEIGHTS.is_file()
        and _sha256(PERSISTENT_INITIAL_WEIGHTS) == _sha256(working)
    ):
        return PERSISTENT_INITIAL_WEIGHTS
    PERSISTENT_INITIAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(working, PERSISTENT_INITIAL_WEIGHTS)
    return PERSISTENT_INITIAL_WEIGHTS


def _reload_volumes() -> None:
    """Sincroniza los Volumes antes de leerlos."""
    os.chdir("/tmp")
    dataset_volume.reload()
    outputs_volume.reload()


def _execute(
    operation: str,
    target: str,
    *,
    require_gpu: bool,
    validate_final_config: bool = False,
    variables: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Ejecuta un target con el dataset verificado y sincroniza los artefactos."""
    _reload_volumes()
    _dataset_gate()
    os.chdir(REPO_ANCHOR)
    _restore_initial_weights()
    if validate_final_config:
        _require_final_training_config()
    try:
        runtime = _runtime_report(operation, require_gpu=require_gpu)
        _make(target, *variables)
    finally:
        _persist_initial_weights()
        outputs_volume.commit()
        print(f"Volume {OUTPUTS_VOLUME_NAME} sincronizado después de {operation}", flush=True)
    return runtime


def _require_summary(
    relative_path: str,
    expected_status: str,
    *required_fields: str,
) -> dict[str, Any]:
    """Exige un resumen con el estado y los campos declarados."""
    path = SEGMENTATION_OUTPUT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Falta el resumen requerido: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise RuntimeError(
            f"Estado inesperado en {path}: {payload.get('status')!r} != {expected_status!r}"
        )
    missing = [field for field in required_fields if payload.get(field) is None]
    if missing:
        raise RuntimeError(f"Campos ausentes en {path}: {missing}")
    return payload


def _checkpoint_record(path: Path) -> dict[str, Any]:
    """Registra ruta, hash y tamaño de un checkpoint del Volume de artefactos."""
    resolved = path.resolve()
    if not resolved.is_relative_to(OUTPUTS_MOUNT.resolve()):
        raise RuntimeError(f"Checkpoint fuera del Volume de artefactos: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_final_training_config() -> Path:
    """Exige la configuración de 150 épocas congelada por el smoke."""
    import yaml

    expected = (
        SEGMENTATION_OUTPUT_ROOT / "segmenter/configs/train_yolo26n_seg.final.yaml"
    ).resolve()
    smoke = _require_summary(
        "segmenter/smoke_summary.json",
        "passed",
        "final_config",
        "final_config_sha256",
    )
    configured = Path(str(smoke["final_config"])).resolve()
    if configured != expected or not expected.is_file():
        raise RuntimeError(f"Configuración final incorrecta: {configured} != {expected}")
    if _sha256(expected) != smoke["final_config_sha256"]:
        raise RuntimeError("La configuración final cambió después del smoke")
    payload = yaml.safe_load(expected.read_text(encoding="utf-8"))
    batch = payload.get("batch") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("task") != "segment"
        or payload.get("epochs") != 150
        or not isinstance(batch, int)
        or isinstance(batch, bool)
        or batch <= 0
        or payload.get("seed") != 42
        or payload.get("deterministic") is not True
    ):
        raise RuntimeError("La configuración final de 150 épocas no es válida")
    return expected


def _require_confirmation(value: str, operation: str) -> None:
    """Exige la confirmación literal para operaciones que entrenan."""
    if value != "true":
        raise RuntimeError(f"{operation} bloqueado: use --confirm true exactamente")


def _experiment_paths(profile: str) -> tuple[Path, Path, Path]:
    """Resuelve configuración y manifiestos de un experimento permitido."""
    filename = TRAINABLE_EXPERIMENTS.get(profile)
    if filename is None:
        allowed = ", ".join(sorted(TRAINABLE_EXPERIMENTS))
        raise RuntimeError(
            f"Perfil de experimento no permitido: {profile!r}; use uno de: {allowed}. "
            "d02b_imgsz768_seed42 y d04_yolo26s_seed42 permanecen bloqueados "
            "hasta completar su smoke específico."
        )
    config = Path(REPO_ANCHOR) / "cloud_training" / "configs" / "experiments" / filename
    manifest = SEGMENTATION_OUTPUT_ROOT / "segmenter" / "experiment_manifests" / f"{profile}.json"
    summary = SEGMENTATION_OUTPUT_ROOT / "segmenter" / "experiment_summaries" / f"{profile}.json"
    return config, manifest, summary


@app.function(
    image=modal_image,
    volumes={DATASET_MOUNT_PATH: dataset_volume},
    secrets=[HF_SECRET],
    cpu=4.0,
    memory=16384,
    timeout=2 * 3600,
)
def seed_dataset(force: bool = False) -> dict[str, Any]:
    """Siembra el dataset congelado en el Volume desde Hugging Face.

    Idempotente: la descarga se omite cuando el árbol ya está materializado. ``force``
    vacía el destino antes de volver a bajarlo.

    @param {bool} force Elimina el árbol existente antes de descargar. Destructivo.
    @returns {dict[str, Any]} Reporte de locks verificados tras la siembra.
    """
    from scripts.dataset.download_leaf_segmentation_dataset import download_dataset

    locks = download_dataset(
        repo_id=HF_DATASET_REPO,
        dataset_root=DATASET_ROOT,
        token=os.environ.get("HF_TOKEN"),
        force=force,
    )
    dataset_volume.commit()
    result = {
        "status": "seeded",
        "dataset_root": str(DATASET_ROOT),
        "hf_dataset_repo": HF_DATASET_REPO,
        **locks,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def verify_dataset() -> dict[str, Any]:
    """Recalcula los fingerprints del dataset montado y deja constancia."""
    _reload_volumes()
    marker = _dataset_gate()
    outputs_volume.commit()
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    return marker


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def preflight() -> dict[str, Any]:
    """Ejecuta el preflight de dataset, runtime, pesos y CUDA."""
    _execute("preflight", "leaf-segmentation-cloud-preflight", require_gpu=True)
    summary = _require_summary(
        "cloud_preflight/summary.json",
        "ready_for_smoke_training",
        "dataset_verified",
        "gpu_verified",
        "model_verified",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=2 * 3600,
)
def smoke(confirm: str = "false") -> dict[str, Any]:
    """Ejecuta una única época con AutoBatch tras ``--confirm true``."""
    _require_confirmation(confirm, "smoke")
    runtime = _execute(
        "smoke",
        "leaf-segmentation-cloud-smoke",
        require_gpu=True,
        variables=("CONFIRM_SEGMENTATION_SMOKE_TRAINING=1",),
    )
    summary = _require_summary(
        "segmenter/smoke_summary.json",
        "passed",
        "save_dir",
        "selected_batch",
        "peak_vram_bytes",
        "duration_seconds",
        "final_config",
        "final_config_sha256",
    )
    weights = Path(str(summary["save_dir"])) / "weights"
    manifest = {
        "schema_version": 2,
        "status": "passed",
        "summary": str(SEGMENTATION_OUTPUT_ROOT / "segmenter/smoke_summary.json"),
        "selected_batch": summary["selected_batch"],
        "peak_vram_bytes": summary["peak_vram_bytes"],
        "duration_seconds": summary["duration_seconds"],
        "final_config": summary["final_config"],
        "final_config_sha256": summary["final_config_sha256"],
        "checkpoints": {
            "best": _checkpoint_record(weights / "best.pt"),
            "last": _checkpoint_record(weights / "last.pt"),
        },
        "runtime": runtime,
    }
    _write_json(SEGMENTATION_OUTPUT_ROOT / "segmenter/modal_smoke_manifest.json", manifest)
    outputs_volume.commit()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 3600,
)
def train(confirm: str = "false") -> dict[str, Any]:
    """Ejecuta la configuración congelada de 150 épocas y persiste los checkpoints."""
    _require_confirmation(confirm, "train")
    _execute(
        "train",
        "leaf-segmentation-cloud-train",
        require_gpu=True,
        validate_final_config=True,
        variables=(
            "CONFIRM_SEGMENTATION_TRAINING=1",
            f"CONFIG={SEGMENTATION_OUTPUT_ROOT}/segmenter/configs/"
            "train_yolo26n_seg.final.yaml",
        ),
    )
    summary = _require_summary(
        "segmenter/training_summary.json",
        "passed",
        "save_dir",
        "selected_batch",
        "peak_vram_bytes",
        "duration_seconds",
        "checkpoints",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 3600,
)
def experiment(profile: str, confirm: str = "false") -> dict[str, Any]:
    """Ejecuta un experimento permitido sin tocar el baseline."""
    _require_confirmation(confirm, "experiment")
    config, _, _ = _experiment_paths(profile)
    _execute(
        f"experiment:{profile}",
        "leaf-segmentation-cloud-train",
        require_gpu=True,
        variables=("CONFIRM_SEGMENTATION_TRAINING=1", f"CONFIG={config}"),
    )
    summary = _require_summary(
        f"segmenter/experiment_summaries/{profile}.json",
        "passed",
        "experiment_id",
        "save_dir",
        "selected_batch",
        "peak_vram_bytes",
        "duration_seconds",
        "checkpoints",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 3600,
)
def resume(confirm: str = "false") -> dict[str, Any]:
    """Reanuda exclusivamente el run identificado por active_run_manifest.json."""
    _require_confirmation(confirm, "resume")
    _execute(
        "resume",
        "leaf-segmentation-cloud-resume",
        require_gpu=True,
        variables=("CONFIRM_SEGMENTATION_TRAINING=1",),
    )
    manifest = _require_summary(
        "segmenter/resume_manifest.json",
        "completed",
        "checkpoint",
        "checkpoint_sha256",
        "run_id",
        "save_dir",
        "checkpoints",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 3600,
)
def resume_experiment(profile: str, confirm: str = "false") -> dict[str, Any]:
    """Reanuda el checkpoint registrado para un experimento permitido."""
    _require_confirmation(confirm, "resume_experiment")
    _, active_manifest, _ = _experiment_paths(profile)
    _execute(
        f"resume_experiment:{profile}",
        "leaf-segmentation-cloud-resume",
        require_gpu=True,
        variables=(
            "CONFIRM_SEGMENTATION_TRAINING=1",
            f"ACTIVE_MANIFEST={active_manifest}",
        ),
    )
    manifest = _require_summary(
        f"segmenter/experiment_resume_manifests/{profile}.json",
        "completed",
        "checkpoint",
        "checkpoint_sha256",
        "run_id",
        "save_dir",
        "checkpoints",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=2.0,
    memory=8192,
    timeout=1800,
)
def promote() -> dict[str, Any]:
    """Promueve el best.pt entrenado al checkpoint que consume la inferencia."""
    _reload_volumes()
    from src.training.checkpoint_promotion import promote_checkpoint

    registry = promote_checkpoint(OUTPUTS_MOUNT)
    outputs_volume.commit()
    print(json.dumps(registry, indent=2, sort_keys=True), flush=True)
    return registry


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=3 * 3600,
)
def validate(force_rerun: str = "false") -> dict[str, Any]:
    """Evalúa el checkpoint promovido exclusivamente sobre el test retenido.

    @param {str} force_rerun ``"true"`` archiva una evaluación previa y repite el test.
        Repetirlo tras ver el resultado invalida su valor metodológico: exige decisión
        formal registrada.
    """
    from src.training.checkpoint_promotion import expected_best_checkpoint_sha256

    locked_counts = _verify_mounted_dataset()
    if force_rerun == "true":
        os.environ["FORCE_INTERNAL_TEST_RERUN"] = "1"
    _execute("validate", "leaf-segmentation-cloud-validate", require_gpu=True)
    summary = _require_summary(
        "segmenter_evaluation/test_summary.json",
        "passed",
        "requested_split",
        "evaluated_split",
        "split",
        "image_count",
        "annotation_count",
        "loader_instance_count",
        "evaluated_instance_count",
        "checkpoint",
        "checkpoint_sha256",
        "metrics",
        "box_metrics",
        "mask_metrics",
        "save_dir",
    )
    if (
        summary.get("requested_split") != "test"
        or summary.get("evaluated_split") != "test"
        or summary.get("split") != "test"
        or summary.get("image_count") != locked_counts["image_counts"]["test"]
        or summary.get("annotation_count") != locked_counts["mask_counts"]["test"]
        or summary.get("evaluated_instance_count") != summary.get("loader_instance_count")
        or summary.get("test_fingerprint") != EXPECTED_TEST_FINGERPRINT
        or summary.get("checkpoint_sha256") != expected_best_checkpoint_sha256(OUTPUTS_MOUNT)
        or summary.get("pilot_used") is not False
        or summary.get("environment_modified") is not False
        or summary.get("environment_before") != summary.get("environment_after")
        or summary.get("faster_coco_eval") != EXPECTED_FASTER_COCO_EVAL
        or Path(str(summary.get("save_dir", ""))).resolve()
        != (SEGMENTATION_OUTPUT_ROOT / "segmenter_evaluation/yolo26n_seg_test").resolve()
    ):
        raise RuntimeError(
            "La evaluación Modal no corresponde exclusivamente al test retenido"
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def results() -> None:
    """Imprime el inventario persistente de resultados de entrenamiento."""
    _execute("results", "leaf-segmentation-cloud-results", require_gpu=False)


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=2.0,
    memory=4096,
    timeout=3600,
)
def checksums() -> None:
    """Escribe los hashes persistentes de los artefactos de entrenamiento."""
    _execute("checksums", "leaf-segmentation-cloud-checksums", require_gpu=False)


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def clean_outputs(confirm: str = "false") -> dict[str, Any]:
    """Vacía el Volume de artefactos tras ``--confirm true``."""
    _require_confirmation(confirm, "clean_outputs")
    _reload_volumes()
    removed = []
    if SEGMENTATION_OUTPUT_ROOT.exists():
        removed = sorted(entry.name for entry in SEGMENTATION_OUTPUT_ROOT.iterdir())
        shutil.rmtree(SEGMENTATION_OUTPUT_ROOT, ignore_errors=True)
    outputs_volume.commit()
    result = {"status": "cleaned", "removed": removed}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result

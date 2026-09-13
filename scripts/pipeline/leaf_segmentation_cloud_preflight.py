#!/usr/bin/env python3
"""GPU/model preflight for the cloud-only leaf segmenter workflow."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

from src.config import PROJECT_ROOT
from src.training.segmentation_preflight import (
    EXPECTED_INITIAL_WEIGHTS_SHA256,
    validate_segmentation_dataset,
    verify_cloud_training_payload,
)


def project_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


OUTPUT_ROOT = project_path("LEAF_SEGMENTATION_OUTPUT", "outputs/leaf_detection")
OUT = OUTPUT_ROOT / "cloud_preflight"
DATASET = project_path(
    "LEAF_SEGMENTATION_DATASET",
    "data/leaf_detection/detector_dataset",
)
MODEL_NAME = os.getenv("SEGMENTATION_MODEL", "yolo26n-seg.pt")
DEVICE = int(os.getenv("SEGMENTATION_DEVICE", "0").split(",", maxsplit=1)[0])
ULTRALYTICS_REQUIREMENT = (
    PROJECT_ROOT / "cloud_training" / "requirements" / "ultralytics.in"
)


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_ultralytics_version() -> str:
    matches = [
        line.strip().removeprefix("ultralytics==")
        for line in ULTRALYTICS_REQUIREMENT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("ultralytics==")
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(
            f"Versión exacta inválida en {ULTRALYTICS_REQUIREMENT}"
        )
    return matches[0]


def tensor_leaves(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for item in value.values()
            for tensor in tensor_leaves(item)
        ]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensor_leaves(item)]
    return []


def package_commit() -> str:
    version_file = PROJECT_ROOT / "cloud_training" / "COMMIT_VERSION.txt"
    if version_file.is_file():
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("commit="):
                return line.partition("=")[2] or "unavailable"
    return "unavailable"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = "ready_for_smoke_training"
    blockers: list[str] = []
    try:
        dataset_locks = verify_cloud_training_payload(DATASET)
        dataset = validate_segmentation_dataset(DATASET)
        if not dataset["passed"]:
            raise RuntimeError(str(dataset["errors"]))
    except Exception as exc:
        dataset_locks = {"passed": False, "error": str(exc)}
        dataset = {"passed": False, "error": str(exc)}
        status = "blocked_by_dataset_change"
        blockers.append(str(exc))
    write("dataset_check.json", {**dataset, "locks": dataset_locks})

    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_compiled": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "device_verified": False,
        "name": None,
        "vram_free_bytes": None,
        "vram_total_bytes": None,
    }
    if gpu["cuda_available"]:
        try:
            gpu["name"] = torch.cuda.get_device_name(DEVICE)
            gpu["vram_free_bytes"], gpu["vram_total_bytes"] = (
                torch.cuda.mem_get_info(DEVICE)
            )
            gpu["device_verified"] = True
        except Exception as exc:
            gpu["device_error"] = repr(exc)
            if status == "ready_for_smoke_training":
                status = "blocked_by_gpu"
            blockers.append(f"GPU device {DEVICE} inválido: {exc!r}")
    elif status == "ready_for_smoke_training":
        status = "blocked_by_gpu"
        blockers.append("CUDA no disponible")
    write("gpu.json", gpu)
    dependency_error: str | None = None
    torchvision_version: str | None = None
    try:
        torchvision = importlib.import_module("torchvision")
        torchvision_version = str(torchvision.__version__)
    except Exception as exc:
        dependency_error = f"torchvision no importable: {exc!r}"
        if status == "ready_for_smoke_training":
            status = "blocked_by_dependency"
        blockers.append(dependency_error)
    try:
        installed_ultralytics = metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        installed_ultralytics = None
    expected_ultralytics = expected_ultralytics_version()
    if (
        installed_ultralytics != expected_ultralytics
        and status == "ready_for_smoke_training"
    ):
        status = "blocked_by_dependency"
        blockers.append(
            "Ultralytics inválido: "
            f"{installed_ultralytics!r} != {expected_ultralytics!r}"
        )
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "torchvision_import_error": dependency_error,
        "ultralytics": installed_ultralytics,
        "expected_ultralytics": expected_ultralytics,
        "commit": package_commit(),
    }
    write("environment.json", environment)
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=(OUT / "dependency_lock.txt").open("w", encoding="utf-8"),
        check=True,
    )

    model_check: dict[str, object] = {
        "candidate": MODEL_NAME,
        "task": "segment",
        "constructed": False,
        "forward_executed": False,
        "segmentation_output": False,
        "segmentation_head_verified": False,
        "forward_finite": False,
        "forward_tensor_shapes": [],
        "ultralytics": None,
    }
    weights_manifest: dict[str, object] = {
        "model": MODEL_NAME,
        "download_allowed_in_cloud": True,
        "resolved": False,
    }
    memory: dict[str, object] = {"checked": False}
    if status == "ready_for_smoke_training":
        try:
            from ultralytics import YOLO
            from ultralytics import __version__ as ultralytics_version

            if ultralytics_version != expected_ultralytics:
                raise RuntimeError(
                    "Versión importada de Ultralytics no coincide con el lock: "
                    f"{ultralytics_version} != {expected_ultralytics}"
                )
            model_check["ultralytics"] = ultralytics_version
            torch.cuda.reset_peak_memory_stats(DEVICE)
            model = YOLO(MODEL_NAME)
            model_check["constructed"] = True
            model_task = str(getattr(model, "task", ""))
            model_check["resolved_task"] = model_task
            if model_task != "segment":
                raise RuntimeError(f"Task inesperado para el modelo: {model_task!r}")
            candidate = Path(str(getattr(model, "ckpt_path", MODEL_NAME))).resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Pesos no resueltos: {candidate}")
            candidate_sha256 = sha256(candidate)
            frozen_sha256 = EXPECTED_INITIAL_WEIGHTS_SHA256.get(candidate.name)
            weights_manifest["expected_sha256"] = frozen_sha256
            if frozen_sha256 is not None and candidate_sha256 != frozen_sha256:
                raise FileNotFoundError(
                    f"Procedencia de {candidate.name} fuera del contrato congelado: "
                    f"{candidate_sha256} != {frozen_sha256}"
                )
            weights_manifest.update(
                {
                    "resolved": True,
                    "path": str(candidate),
                    "sha256": candidate_sha256,
                    "size_bytes": candidate.stat().st_size,
                    "source": "Ultralytics model resolver",
                    "ultralytics_version": ultralytics_version,
                }
            )
            network = model.model
            layers = getattr(network, "model", ())
            head = layers[-1] if len(layers) else None
            head_name = type(head).__name__ if head is not None else ""
            segmentation_head = "segment" in head_name.casefold() or hasattr(
                head, "proto"
            )
            model_check["head_class"] = head_name
            model_check["segmentation_head_verified"] = segmentation_head
            if not segmentation_head:
                raise RuntimeError(
                    f"Head de segmentación no verificado: {head_name!r}"
                )
            cuda_device = torch.device(f"cuda:{DEVICE}")
            network.to(cuda_device)
            network.eval()
            synthetic = torch.zeros(
                (1, 3, 640, 640),
                dtype=torch.float32,
                device=cuda_device,
            )
            with torch.inference_mode():
                raw_output = network(synthetic)
            tensors = tensor_leaves(raw_output)
            model_check["forward_executed"] = True
            model_check["forward_tensor_shapes"] = [
                list(tensor.shape) for tensor in tensors
            ]
            model_check["forward_finite"] = bool(
                tensors and all(torch.isfinite(tensor).all().item() for tensor in tensors)
            )
            model_check["segmentation_output"] = bool(
                segmentation_head and model_check["forward_finite"]
            )
            if not model_check["segmentation_output"]:
                raise RuntimeError(
                    "El forward sintético del segmentador no produjo tensores finitos"
                )
            memory = {
                "checked": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(DEVICE),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(DEVICE),
                "vram_free_bytes": torch.cuda.mem_get_info(DEVICE)[0],
            }
        except (ModuleNotFoundError, metadata.PackageNotFoundError) as exc:
            status = "blocked_by_dependency"
            blockers.append(str(exc))
        except FileNotFoundError as exc:
            status = "blocked_by_weights"
            blockers.append(str(exc))
        except Exception as exc:
            status = "blocked_by_model"
            blockers.append(repr(exc))
    write("model_check.json", model_check)
    write("weights_manifest.json", weights_manifest)
    write("memory_check.json", memory)
    write(
        "summary.json",
        {
            "status": status,
            "blockers": blockers,
            "dataset_verified": bool(dataset.get("passed")),
            "gpu_verified": bool(
                gpu["cuda_available"] and gpu["device_verified"]
            ),
            "model_verified": bool(model_check["segmentation_output"]),
            "training_started": False,
            "epochs_run": 0,
            "optimizer_steps": 0,
        },
    )
    print(status)


if __name__ == "__main__":
    main()

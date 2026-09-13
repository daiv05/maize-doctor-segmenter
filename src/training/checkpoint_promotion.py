"""Promoción registrada del checkpoint entrenado al que consume la inferencia.

El entrenamiento deja ``segmenter/yolo26n_seg_baseline/weights/best.pt`` y la inferencia
lee la ruta declarada en ``config/segmentation.yaml``. Este módulo materializa esa copia y
registra la identidad promovida, que es la que el gate de evaluación exige después.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_BEST_CHECKPOINT_SHA256 = (
    "4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f"
)
PROMOTION_OVERRIDE_VARIABLE = "SEGMENTATION_EXPECTED_BEST_SHA256"
PROMOTION_REGISTRY_NAME = "promoted_checkpoint.json"
PROMOTED_CHECKPOINT_NAME = "doctor_maiz_leaf_segmenter_best.pt"


class CheckpointPromotionError(RuntimeError):
    """Se lanza cuando la promoción o su identidad registrada no son válidas."""


def sha256_file(path: Path) -> str:
    """Calcula el SHA-256 de un archivo por bloques.

    @param {Path} path Archivo a digerir.
    @returns {str} Digest hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def models_root(output_root: Path) -> Path:
    """Devuelve el directorio de modelos servibles bajo la raíz de artefactos."""
    return output_root / "leaf_detection" / "models"


def registry_path(output_root: Path) -> Path:
    """Devuelve la ruta del registro de promoción."""
    return models_root(output_root) / PROMOTION_REGISTRY_NAME


def promoted_checkpoint_path(output_root: Path) -> Path:
    """Devuelve la ruta del checkpoint que consume la inferencia."""
    return models_root(output_root) / PROMOTED_CHECKPOINT_NAME


def read_registry(output_root: Path) -> dict[str, Any] | None:
    """Lee el registro de promoción si existe.

    @param {Path} output_root Raíz de artefactos.
    @returns {dict[str, Any]|None} Contenido del registro o None.
    """
    path = registry_path(output_root)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def expected_best_checkpoint_sha256(output_root: Path) -> str:
    """Resuelve el SHA-256 que la evaluación sobre test debe exigir.

    La precedencia es explícita para que la identidad evaluada siempre esté declarada
    antes de mirar el resultado: variable de entorno, registro de promoción y, en último
    lugar, el checkpoint congelado del run original.

    @param {Path} output_root Raíz de artefactos.
    @returns {str} Digest hexadecimal esperado.
    """
    override = os.getenv(PROMOTION_OVERRIDE_VARIABLE, "").strip().lower()
    if override:
        hexadecimal = set("0123456789abcdef")
        if len(override) != 64 or not set(override) <= hexadecimal:
            raise CheckpointPromotionError(
                f"{PROMOTION_OVERRIDE_VARIABLE} debe ser un SHA-256 hexadecimal de 64 caracteres"
            )
        return override
    registry = read_registry(output_root)
    if registry is not None:
        declared = str(registry.get("sha256", ""))
        if len(declared) != 64:
            raise CheckpointPromotionError(
                f"Registro de promoción inválido en {registry_path(output_root)}"
            )
        return declared
    return FROZEN_BEST_CHECKPOINT_SHA256


def promote_checkpoint(
    output_root: Path,
    *,
    source: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Copia el checkpoint entrenado a la ruta servible y registra su identidad.

    @param {Path} output_root Raíz de artefactos.
    @param {Path|None} source Checkpoint origen; por defecto el best.pt del baseline.
    @param {bool} force Permite reemplazar una promoción previa distinta.
    @returns {dict[str, Any]} Registro escrito.
    """
    origin = source or (
        output_root
        / "leaf_detection"
        / "segmenter"
        / "yolo26n_seg_baseline"
        / "weights"
        / "best.pt"
    )
    if not origin.is_file():
        raise CheckpointPromotionError(f"No existe el checkpoint entrenado: {origin}")

    destination = promoted_checkpoint_path(output_root)
    digest = sha256_file(origin)
    previous = read_registry(output_root)
    if previous is not None and previous.get("sha256") != digest and not force:
        raise CheckpointPromotionError(
            "Ya existe una promoción con otra identidad: "
            f"{previous.get('sha256')} != {digest}; use --force para reemplazarla"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin, destination)
    copied_digest = sha256_file(destination)
    if copied_digest != digest:
        raise CheckpointPromotionError(
            f"La copia promovida no coincide con el origen: {copied_digest} != {digest}"
        )

    registry = {
        "schema_version": 1,
        "status": "promoted",
        "source": str(origin.resolve()),
        "checkpoint": str(destination.resolve()),
        "sha256": digest,
        "size_bytes": destination.stat().st_size,
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "replaces": previous.get("sha256") if previous else None,
    }
    registry_path(output_root).write_text(
        json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry

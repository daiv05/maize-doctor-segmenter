"""Marcadores compartidos por la suite.

Los datos del segmentador están en ``.gitignore``, así que en un checkout limpio no
existen. Las pruebas que los necesitan se marcan con ``requires_dataset`` o
``requires_pilot`` y se saltan con un motivo explícito, para que la suite distinga
"falta el dataset" de "el código se rompió".

Los marcadores se resuelven aquí y no con ``pytest.mark.skipif`` importado desde este
módulo porque Ultralytics publica su propio paquete ``tests`` de primer nivel, que
ensombrece al del proyecto en un ``import tests.conftest``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "leaf_detection" / "detector_dataset"
PILOT_ROOT = PROJECT_ROOT / "data" / "leaf_detection" / "pilot"

REQUIRED_DATASET_ENTRIES = (
    DATASET_ROOT / "dataset.yaml",
    DATASET_ROOT / "manifests" / "dataset_lock.json",
    DATASET_ROOT / "manifests" / "split_lock.json",
    DATASET_ROOT / "images" / "train",
    DATASET_ROOT / "labels" / "train",
)


def dataset_is_materialized() -> bool:
    """Indica si el dataset YOLO-seg está materializado localmente."""
    return all(entry.exists() for entry in REQUIRED_DATASET_ENTRIES)


def pilot_is_materialized() -> bool:
    """Indica si el piloto externo retenido está materializado localmente."""
    return PILOT_ROOT.is_dir() and any(PILOT_ROOT.iterdir())


def pytest_configure(config: pytest.Config) -> None:
    """Registra los marcadores propios de la suite."""
    config.addinivalue_line(
        "markers", "requires_dataset: necesita el dataset YOLO-seg materializado"
    )
    config.addinivalue_line(
        "markers", "requires_pilot: necesita el piloto externo retenido"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Salta las pruebas cuyos datos no están materializados en este checkout."""
    skip_dataset = pytest.mark.skip(reason=f"Falta el dataset materializado en {DATASET_ROOT}")
    skip_pilot = pytest.mark.skip(reason=f"Falta el piloto retenido en {PILOT_ROOT}")
    dataset_present = dataset_is_materialized()
    pilot_present = pilot_is_materialized()
    for item in items:
        if not dataset_present and item.get_closest_marker("requires_dataset"):
            item.add_marker(skip_dataset)
        if not pilot_present and item.get_closest_marker("requires_pilot"):
            item.add_marker(skip_pilot)

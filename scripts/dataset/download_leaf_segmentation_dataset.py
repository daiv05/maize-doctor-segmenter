#!/usr/bin/env python3
"""Descarga el dataset YOLO-seg congelado desde Hugging Face y verifica sus locks.

Reconstruye ``<raiz>/detector_dataset`` a partir de los shards publicados y sólo declara
éxito cuando ``verify_cloud_training_payload`` reproduce los fingerprints congelados.

Uso:
    python scripts/dataset/download_leaf_segmentation_dataset.py
    python scripts/dataset/download_leaf_segmentation_dataset.py --force
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

from src.config import get_project_data_root
from src.training.segmentation_preflight import verify_cloud_training_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "daiv05/corn-leaf-instance-segmentation"
REQUIRED_ENTRIES = ("dataset.yaml", "images", "labels", "manifests")


def default_dataset_root() -> Path:
    """Devuelve la raíz esperada del dataset del segmentador."""
    return get_project_data_root() / "leaf_detection" / "detector_dataset"


def dataset_is_materialized(dataset_root: Path) -> bool:
    """Indica si el árbol está completo y sin shards pendientes de extraer.

    @param {Path} dataset_root Raíz del dataset YOLO-seg.
    @returns {bool} True cuando no falta ninguna entrada obligatoria ni queda un .tar.
    """
    if not dataset_root.is_dir():
        return False
    if any(dataset_root.glob("*.tar")):
        return False
    return all((dataset_root / entry).exists() for entry in REQUIRED_ENTRIES)


def extract_and_remove_tars(dataset_root: Path) -> None:
    """Descomprime cada shard en su lugar y lo elimina al terminar."""
    for shard in sorted(dataset_root.glob("*.tar")):
        logger.info(f"Extrayendo {shard.name}...")
        with tarfile.open(shard) as tar:
            tar.extractall(dataset_root, filter="data")
        shard.unlink()


def download_dataset(
    repo_id: str,
    dataset_root: Path,
    token: str | None,
    force: bool,
) -> dict[str, Any]:
    """Descarga, extrae y verifica el dataset congelado.

    @param {str} repo_id Repo de dataset en el Hub.
    @param {Path} dataset_root Destino del árbol YOLO-seg.
    @param {str|None} token Token de Hugging Face para repos privados.
    @param {bool} force Vacía el destino antes de descargar.
    @returns {dict[str, Any]} Reporte de locks verificados.
    """
    from huggingface_hub import snapshot_download

    if force and dataset_root.exists():
        logger.info(f"--force: eliminando {dataset_root}")
        shutil.rmtree(dataset_root, ignore_errors=True)

    if dataset_is_materialized(dataset_root):
        logger.info(f"{dataset_root} ya está materializado; se omite la descarga.")
    else:
        logger.info(f"Descargando hf://datasets/{repo_id} -> {dataset_root}")
        dataset_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_root),
            token=token,
            ignore_patterns=[".gitattributes", "README.md", "dataset_infos.json"],
        )
        extract_and_remove_tars(dataset_root)
        shutil.rmtree(dataset_root / ".cache", ignore_errors=True)

    locks = verify_cloud_training_payload(dataset_root)
    logger.info(
        f"Locks verificados: parent={str(locks['parent_fingerprint'])[:12]} "
        f"counts={locks['image_counts']} masks={locks['mask_counts']}"
    )
    return locks


def build_parser() -> argparse.ArgumentParser:
    """Construye la CLI de descarga."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SEGMENTATION_DATASET_REPO", DEFAULT_REPO_ID),
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    """Punto de entrada de la descarga."""
    args = build_parser().parse_args()
    download_dataset(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root.resolve(),
        token=args.token,
        force=args.force,
    )


if __name__ == "__main__":
    main()

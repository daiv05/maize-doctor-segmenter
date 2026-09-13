#!/usr/bin/env python3
"""Publica el dataset YOLO-seg congelado en Hugging Face Datasets Hub.

El repo lleva los pares imagen/etiqueta en shards ``detector_dataset-<NNNNN>.tar`` y los
manifiestos de bloqueo como archivos sueltos, de modo que la descarga reconstruya el árbol
que ``verify_cloud_training_payload`` sabe verificar.

Uso:
    python scripts/dataset/upload_leaf_segmentation_dataset.py --stage-dir <ruta> --dry-run
    python scripts/dataset/upload_leaf_segmentation_dataset.py --stage-dir <ruta>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

from scripts.package.build_leaf_segmentation_cloud_package import dataset_paths
from src.config import get_project_data_root
from src.training.segmentation_preflight import verify_cloud_training_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SHARD_PREFIX = "detector_dataset"
SHARD_MAX_BYTES = 800 * 1024 * 1024
DEFAULT_REPO_ID = "daiv05/corn-leaf-instance-segmentation"


def default_dataset_root() -> Path:
    """Devuelve la raíz del dataset del segmentador."""
    return get_project_data_root() / "leaf_detection" / "detector_dataset"


def split_payload(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """Separa el payload en pares imagen/etiqueta y archivos de control.

    @param {Path} dataset_root Raíz del dataset YOLO-seg.
    @returns {tuple[list[Path], list[Path]]} Rutas a empaquetar y rutas a subir sueltas.
    """
    sharded: list[Path] = []
    plain: list[Path] = []
    for path in dataset_paths(dataset_root):
        relative = path.relative_to(dataset_root)
        if relative.parts[0] in {"images", "labels"}:
            sharded.append(path)
        else:
            plain.append(path)
    return sorted(sharded), sorted(plain)


def plan_shards(paths: list[Path], max_bytes: int) -> list[list[Path]]:
    """Reparte rutas en shards por tamaño acumulado y de forma reproducible.

    @param {list[Path]} paths Rutas ya ordenadas.
    @param {int} max_bytes Techo por shard.
    @returns {list[list[Path]]} Shards con sus rutas.
    """
    shards: list[list[Path]] = [[]]
    current = 0
    for path in paths:
        size = path.stat().st_size
        if current + size > max_bytes and shards[-1]:
            shards.append([])
            current = 0
        shards[-1].append(path)
        current += size
    return shards


def write_dataset_infos(
    dataset_root: Path,
    locks: dict[str, Any],
    destination: Path,
) -> None:
    """Escribe la metadata declarativa del dataset con los fingerprints congelados."""
    payload = {
        "default": {
            "description": (
                "Segmentación de instancias de hojas de maíz en formato YOLO-seg, "
                "clase única maize_leaf, splits agrupados con seed 42."
            ),
            "features": {
                "image": {"_type": "Image"},
                "label": {"names": ["maize_leaf"], "_type": "ClassLabel"},
            },
            "splits": {
                name: {"name": name, "num_examples": count}
                for name, count in sorted(locks["image_counts"].items())
            },
            "mask_counts": dict(sorted(locks["mask_counts"].items())),
            "parent_fingerprint": locks["parent_fingerprint"],
            "split_fingerprints": locks["split_fingerprints"],
            "combined_fingerprint": locks["combined_fingerprint"],
            "seed": locks["seed"],
            "dataset_relative_root": dataset_root.name,
        }
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_stage(
    dataset_root: Path,
    sharded: list[Path],
    locks: dict[str, Any],
    stage_dir: Path,
    max_bytes: int,
) -> list[Path]:
    """Materializa los shards y la metadata en el directorio de staging.

    @returns {list[Path]} Artefactos generados, en orden de subida.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    shards = plan_shards(sharded, max_bytes)
    logger.info(f"Empaquetando {len(sharded)} archivos en {len(shards)} shard(s)...")
    written: list[Path] = []
    for index, shard in enumerate(shards):
        shard_path = stage_dir / f"{SHARD_PREFIX}-{index:05d}.tar"
        with tarfile.open(shard_path, "w") as tar:
            for path in shard:
                tar.add(path, arcname=path.relative_to(dataset_root).as_posix())
        logger.info(
            f"  {shard_path.name}: {len(shard)} archivos, "
            f"{shard_path.stat().st_size / 1e6:.1f} MB"
        )
        written.append(shard_path)
    infos_path = stage_dir / "dataset_infos.json"
    write_dataset_infos(dataset_root, locks, infos_path)
    written.append(infos_path)
    return written


def upload_sequentially(
    api: Any,
    repo_id: str,
    entries: list[tuple[Path, str]],
    retries: int = 3,
) -> None:
    """Sube archivo por archivo con reintentos, sin paralelizar shards en memoria."""
    for index, (path, path_in_repo) in enumerate(entries, start=1):
        for attempt in range(1, retries + 1):
            try:
                started = time.time()
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Anade {path_in_repo}",
                )
                logger.info(
                    f"  [{index}/{len(entries)}] {path_in_repo} subido "
                    f"({time.time() - started:.0f}s)"
                )
                break
            except Exception as error:
                logger.warning(
                    f"  [{index}/{len(entries)}] {path_in_repo} intento {attempt}/{retries} "
                    f"fallo: {type(error).__name__}: {str(error)[:150]}"
                )
                if attempt == retries:
                    raise
                time.sleep(5)


def delete_stale_shards(api: Any, repo_id: str, keep: set[str]) -> None:
    """Borra shards de una publicación anterior que ya no se regeneraron."""
    from huggingface_hub import CommitOperationDelete

    remote = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    stale = [name for name in remote if name.endswith(".tar") and name not in keep]
    if not stale:
        return
    logger.info(f"Eliminando {len(stale)} shard(s) obsoleto(s): {stale[:5]}")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=name) for name in stale],
        commit_message="Elimina shards obsoletos de la publicacion anterior",
    )


def upload_dataset(
    repo_id: str,
    dataset_root: Path,
    stage_dir: Path,
    token: str | None,
    private: bool,
    keep_stage: bool,
    max_bytes: int,
    dry_run: bool,
) -> None:
    """Verifica los locks y publica el dataset congelado en el Hub."""
    locks = verify_cloud_training_payload(dataset_root)
    logger.info(
        f"Locks verificados: parent={str(locks['parent_fingerprint'])[:12]} "
        f"counts={locks['image_counts']}"
    )
    sharded, plain = split_payload(dataset_root)
    total_bytes = sum(path.stat().st_size for path in sharded)
    logger.info(
        f"Payload: {len(sharded)} imagenes/etiquetas ({total_bytes / 1e9:.2f} GB) "
        f"y {len(plain)} archivos de control"
    )

    if dry_run:
        for index, shard in enumerate(plan_shards(sharded, max_bytes)):
            shard_bytes = sum(path.stat().st_size for path in shard)
            logger.info(
                f"[dry-run]   {SHARD_PREFIX}-{index:05d}.tar: "
                f"{len(shard)} archivos, {shard_bytes / 1e6:.1f} MB"
            )
        for path in plain:
            logger.info(f"[dry-run]   {path.relative_to(dataset_root).as_posix()}")
        logger.info(f"[dry-run] destino: hf://datasets/{repo_id} (private={private})")
        return

    stage_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(stage_dir).free
    if free_bytes < total_bytes:
        raise SystemExit(
            f"Espacio insuficiente para el staging: {free_bytes / 1e9:.1f} GB libres, "
            f"se necesitan ~{total_bytes / 1e9:.1f} GB"
        )

    from huggingface_hub import HfApi

    staged = build_stage(dataset_root, sharded, locks, stage_dir, max_bytes)
    entries = [(path, path.name) for path in staged]
    entries += [(path, path.relative_to(dataset_root).as_posix()) for path in plain]

    api = HfApi(token=token)
    logger.info(f"Creando/verificando repo: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    pending = [entry for entry in entries if entry[1] not in remote]
    if len(pending) < len(entries):
        logger.info(f"{len(entries) - len(pending)} archivo(s) ya publicados; se omiten.")
    upload_sequentially(api, repo_id, pending)
    delete_stale_shards(api, repo_id, {path.name for path in staged if path.suffix == ".tar"})
    logger.info(f"Publicacion completada: hf://datasets/{repo_id}")

    if keep_stage:
        logger.info(f"Staging conservado en {stage_dir}")
    else:
        shutil.rmtree(stage_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    """Construye la CLI de publicación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SEGMENTATION_DATASET_REPO", DEFAULT_REPO_ID),
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--shard-size-mb", type=int, default=SHARD_MAX_BYTES // (1024 * 1024))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Punto de entrada de la publicación."""
    args = build_parser().parse_args()
    upload_dataset(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root.resolve(),
        stage_dir=args.stage_dir.resolve(),
        token=args.token,
        private=not args.public,
        keep_stage=args.keep_stage,
        max_bytes=args.shard_size_mb * 1024 * 1024,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

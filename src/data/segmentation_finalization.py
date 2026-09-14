"""Finalize the maize-leaf segmentation pool from sources and human reviews."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

from src.data.jpeg_normalization import (
    IMAGE_NORMALIZATION_COLUMNS,
    write_auxiliary_jpeg_audit,
)
from src.data.segmentation_audit import IMAGE_EXTENSIONS, sha256_file
from src.data.segmentation_consolidation import (
    DUPLICATE_COLUMNS,
    MANIFEST_COLUMNS,
    TARGET_CLASS_ID,
    TARGET_CLASS_NAME,
    VALIDATION_COLUMNS,
    _report_rows,
    build_segmentation_consolidation,
    source_files_fingerprint,
    validate_consolidated_dataset,
    write_csv,
)
from src.data.segmentation_review import (
    build_dataset_lock,
    dataset_fingerprint,
    read_review_manifest,
    validate_review_manifests,
    write_applied_review_report,
    write_dataset_lock,
    write_reannotation_queue,
)

INCLUDED_DECISIONS = {"include", "include_after_remap", "recover_from_coco"}
FINAL_MANIFESTS = (
    "consolidation_manifest.csv",
    "included_annotations.csv",
    "excluded_annotations.csv",
    "recovered_annotations.csv",
    "manual_review.csv",
    "mandatory_visual_review.csv",
    "duplicate_groups.csv",
    "review_decisions_applied.csv",
    "reannotation_queue.csv",
    "selection_summary.json",
    "dataset_lock.json",
    "image_normalization_manifest.csv",
    "auxiliary_jpeg_audit.csv",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _matches_review(manifest: dict[str, str], review: dict[str, str]) -> bool:
    if manifest["source_dataset"] != review["source_dataset"]:
        return False
    if Path(manifest["original_image_path"]).name != review["filename"]:
        return False
    review_class = review["original_class_id"]
    if review_class and manifest["original_class_id"] != review_class:
        return False
    review_line = review["original_line_number"]
    return not review_line or manifest["original_line_number"] == review_line


def _unique_reviews(review_summary: dict[str, object]) -> list[dict[str, str]]:
    rows = [
        *review_summary["approved"],
        *review_summary["excluded"],
        *review_summary["reannotation"],
    ]
    return sorted(rows, key=lambda row: row["review_key"])


def decisions_fingerprint(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    """Hash normalized, unique human decisions deterministically."""
    payload = [
        {
            "review_case_id": row["review_case_id"],
            "review_key": row["review_key"],
            "reviewer_decision": row["reviewer_decision"],
            "review_reason": row["review_reason"],
            "review_status": row["review_status"],
        }
        for row in sorted(rows, key=lambda item: item["review_key"])
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "unique_cases": len(payload),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _apply_reviews(
    candidate_root: Path,
    final_dataset_root: Path,
    reviews: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    manifests_root = candidate_root / "manifests"
    manifest = _read_csv(manifests_root / "consolidation_manifest.csv")
    removals: dict[Path, set[int]] = {}
    matches_by_case: dict[str, int] = {}
    approved_without_includable_geometry: list[str] = []

    for review in reviews:
        matches = [row for row in manifest if _matches_review(row, review)]
        if not matches:
            raise RuntimeError(
                f"La revisión {review['review_case_id']} no coincide con la fuente"
            )
        matches_by_case[review["review_case_id"]] = len(matches)
        decision = review["reviewer_decision"]
        if decision == "approved" and not any(
            row["decision"] in INCLUDED_DECISIONS for row in matches
        ):
            approved_without_includable_geometry.append(review["review_case_id"])
        for row in matches:
            row["reviewer_decision"] = decision
            row["review_reason"] = review["review_reason"]
            row["review_status"] = review["review_status"]
            if decision == "approved" or row["decision"] not in INCLUDED_DECISIONS:
                continue
            label_name = Path(row["consolidated_label_path"]).name
            line_number = int(row["consolidated_line_number"])
            removals.setdefault(
                candidate_root / "all" / "labels" / label_name,
                set(),
            ).add(line_number)
            row["decision"] = (
                "exclude_human_review"
                if decision == "exclude"
                else "exclude_needs_reannotation"
            )
            row["decision_reason"] = (
                f"human_review:{decision}:{review['review_reason']}"
            )
            row["quality_status"] = f"human_review_{decision}"
            row["target_class_id"] = ""
            row["target_class_name"] = ""
            row["consolidated_line_number"] = ""
            row["consolidated_annotation_sha256"] = ""

    removed_annotations = 0
    removed_images = 0
    line_maps: dict[str, dict[int, int]] = {}
    for label_path, line_numbers in sorted(
        removals.items(), key=lambda item: item[0].name
    ):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        line_map: dict[int, int] = {}
        for old_number, line in enumerate(lines, start=1):
            if old_number in line_numbers:
                removed_annotations += 1
                continue
            kept.append(line)
            line_map[old_number] = len(kept)
        line_maps[label_path.name] = line_map
        if kept:
            label_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            continue
        image_candidates = [
            path
            for path in (candidate_root / "all" / "images").glob(
                f"{label_path.stem}.*"
            )
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(image_candidates) != 1:
            raise RuntimeError(f"No se pudo resolver la imagen de {label_path}")
        label_path.unlink()
        image_candidates[0].unlink()
        removed_images += 1

    final_images = {
        path.stem: path
        for path in (candidate_root / "all" / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    for row in manifest:
        label_name = Path(row["consolidated_label_path"]).name
        stem = Path(label_name).stem
        if stem not in final_images:
            row["consolidated_image_path"] = ""
            row["consolidated_label_path"] = ""
            if row["decision"] in INCLUDED_DECISIONS:
                raise RuntimeError(f"Geometría incluida sin archivo final: {stem}")
            continue
        row["consolidated_image_path"] = str(
            final_dataset_root / "all" / "images" / final_images[stem].name
        )
        row["consolidated_label_path"] = str(
            final_dataset_root / "all" / "labels" / label_name
        )
        if row["decision"] in INCLUDED_DECISIONS and label_name in line_maps:
            old_number = int(row["consolidated_line_number"])
            if old_number not in line_maps[label_name]:
                raise RuntimeError(f"Línea incluida eliminada por error: {row}")
            row["consolidated_line_number"] = line_maps[label_name][old_number]

    manifest.sort(
        key=lambda row: (
            row["source_dataset"],
            row["original_image_path"].casefold(),
            int(row["original_line_number"] or 0),
            row["decision"],
        )
    )
    included = [row for row in manifest if row["decision"] in INCLUDED_DECISIONS]
    excluded = [row for row in manifest if row["decision"].startswith("exclude")]
    recovered = [row for row in manifest if row["decision"] == "recover_from_coco"]
    write_csv(
        manifests_root / "consolidation_manifest.csv",
        manifest,
        MANIFEST_COLUMNS,
    )
    write_csv(
        manifests_root / "included_annotations.csv",
        included,
        MANIFEST_COLUMNS,
    )
    write_csv(
        manifests_root / "excluded_annotations.csv",
        excluded,
        MANIFEST_COLUMNS,
    )
    write_csv(
        manifests_root / "recovered_annotations.csv",
        recovered,
        MANIFEST_COLUMNS,
    )
    return manifest, {
        "removed_annotations": removed_annotations,
        "removed_images": removed_images,
        "matched_manifest_rows": matches_by_case,
        "approved_without_includable_geometry": (
            approved_without_includable_geometry
        ),
    }


def _validate_copy_hashes(
    all_root: Path,
    manifest: Sequence[dict[str, str]],
) -> None:
    expected = {
        Path(row["consolidated_image_path"]).name: row["image_sha256"]
        for row in manifest
        if row["decision"] in INCLUDED_DECISIONS
    }
    for name, digest in expected.items():
        path = all_root / "images" / name
        if sha256_file(path) != digest:
            raise RuntimeError(f"Hash de copia incorrecto: {path}")


def _filter_image_normalization_manifest(
    candidate_root: Path,
    final_dataset_root: Path,
) -> dict[str, int]:
    """Retain normalization provenance only for images surviving human review."""
    path = candidate_root / "manifests" / "image_normalization_manifest.csv"
    rows = _read_csv(path)
    final_names = {
        image.name
        for image in (candidate_root / "all" / "images").iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    }
    filtered = [
        row for row in rows if Path(row["derived_path"]).name in final_names
    ]
    if len(filtered) != len(final_names):
        raise RuntimeError(
            "El manifiesto de normalización no corresponde al pool final: "
            f"rows={len(filtered)}, images={len(final_names)}"
        )
    for row in filtered:
        row["derived_path"] = (
            final_dataset_root / "all" / "images" / Path(row["derived_path"]).name
        ).as_posix()
    write_csv(path, filtered, IMAGE_NORMALIZATION_COLUMNS)
    return {
        "jpeg_images": len(filtered),
        "jpeg_images_normalized": sum(
            row["status"] == "normalized" for row in filtered
        ),
        "jpeg_lossless_eoi_repairs": sum(
            row["normalization_method"] == "append_ffd9"
            for row in filtered
        ),
        "jpeg_reencodes": sum(
            row["normalization_method"].startswith("reencode_")
            for row in filtered
        ),
    }


def _rewrite_report_paths(root: Path, old: Path, new: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        replaced = text.replace(str(old), str(new))
        if replaced != text:
            path.write_text(replaced, encoding="utf-8")


def _publish(
    candidate_root: Path,
    report_candidate: Path,
    dataset_root: Path,
    report_root: Path,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifests").mkdir(parents=True, exist_ok=True)
    targets = [
        (candidate_root / "all", dataset_root / "all"),
        (candidate_root / "previews", dataset_root / "previews"),
        (candidate_root / "dataset.yaml", dataset_root / "dataset.yaml"),
        (candidate_root / "README.md", dataset_root / "README.md"),
        *[
            (
                candidate_root / "manifests" / name,
                dataset_root / "manifests" / name,
            )
            for name in FINAL_MANIFESTS
        ],
        (report_candidate, report_root),
    ]
    with tempfile.TemporaryDirectory(
        prefix=".segmentation_publish_backup_",
        dir=dataset_root.parent,
    ) as backup_name:
        backup = Path(backup_name)
        moved_old: list[tuple[Path, Path]] = []
        moved_new: list[Path] = []
        try:
            for _, target in targets:
                if target.exists():
                    saved = backup / f"{len(moved_old):03d}_{target.name}"
                    shutil.move(str(target), saved)
                    moved_old.append((saved, target))
            for source, target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), target)
                moved_new.append(target)
        except Exception:
            for target in reversed(moved_new):
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            for saved, target in reversed(moved_old):
                shutil.move(str(saved), target)
            raise


def _final_readme(summary: dict[str, object], lock: dict[str, object]) -> str:
    counts = summary["counts"]
    decisions = summary["human_reviews"]["decision_counts"]
    return f"""# Dataset definitivo del detector de hojas

Consolidado reconstruido exclusivamente desde `external_sources/` y filtrado
con las 35 decisiones humanas únicas. La única clase permitida es
`0 = maize_leaf`.

- imágenes definitivas: {counts['images_included']};
- máscaras definitivas: {counts['annotations_included']};
- aprobados: {decisions.get('approved', 0)};
- excluidos: {decisions.get('exclude', 0)};
- enviados a reanotación: {decisions.get('needs_reannotation', 0)};
- duplicados incluidos: {summary['validation']['duplicate_count']};
- fugas con el piloto: {summary['validation']['pilot_leakage_count']};
- estado: `{lock['status']}`;
- fingerprint: `{lock['global_fingerprint']['sha256']}`.

No se crearon splits y no se entrenó ningún modelo. `dataset.yaml` describe
únicamente el pool `all/`.
"""


def _materialize_final_candidate(
    *,
    external_root: Path,
    pilot_root: Path,
    eda_root: Path,
    candidate_root: Path,
    report_candidate: Path,
    final_dataset_root: Path,
    final_report_root: Path,
    config_path: Path,
    human_manifest_root: Path,
    review_summary: dict[str, object],
    seed: int,
) -> dict[str, object]:
    base_summary = build_segmentation_consolidation(
        external_root,
        pilot_root,
        eda_root,
        candidate_root,
        report_candidate,
        config_path,
        seed=seed,
    )
    reviews = _unique_reviews(review_summary)
    manifest, application = _apply_reviews(
        candidate_root,
        final_dataset_root,
        reviews,
    )
    base_summary["counts"].update(
        _filter_image_normalization_manifest(
            candidate_root,
            final_dataset_root,
        )
    )
    base_summary["auxiliary_jpeg_audit"] = write_auxiliary_jpeg_audit(
        final_dataset_root,
        candidate_root / "manifests" / "auxiliary_jpeg_audit.csv",
    )
    shutil.copy2(
        human_manifest_root / "manual_review.csv",
        candidate_root / "manifests" / "manual_review.csv",
    )
    shutil.copy2(
        human_manifest_root / "mandatory_visual_review.csv",
        candidate_root / "manifests" / "mandatory_visual_review.csv",
    )
    write_applied_review_report(
        candidate_root / "manifests" / "review_decisions_applied.csv",
        reviews,
    )
    write_reannotation_queue(
        candidate_root / "manifests" / "reannotation_queue.csv",
        review_summary["reannotation"],
    )

    pilot_hashes = {
        sha256_file(path)
        for path in (pilot_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    issues, validation = validate_consolidated_dataset(
        candidate_root / "all",
        manifest,
        pilot_hashes,
    )
    _validate_copy_hashes(candidate_root / "all", manifest)
    validation["duplicate_count"] = sum(
        issue["issue_type"] == "included_exact_duplicate" for issue in issues
    )
    validation["pilot_leakage_count"] = sum(
        issue["issue_type"] == "pilot_leakage" for issue in issues
    )
    if not validation["passed"]:
        raise RuntimeError(f"Falló la validación definitiva: {validation}")

    duplicate_rows = _read_csv(
        candidate_root / "manifests" / "duplicate_groups.csv"
    )
    report_rows = _report_rows(manifest, duplicate_rows, duplicate_rows)
    report_specs = {
        "source_flow.csv": (
            report_rows["source_flow"],
            (
                "source_dataset",
                "images_considered",
                "images_included",
                "images_excluded",
                "annotations_considered",
                "annotations_included",
                "annotations_recovered",
                "annotations_excluded",
                "manual_review",
            ),
        ),
        "class_flow.csv": (
            report_rows["class_flow"],
            (
                "source_dataset",
                "original_class_name",
                "semantic_role",
                "decision",
                "annotations",
            ),
        ),
        "exclusion_reasons.csv": (
            report_rows["exclusion_reasons"],
            ("decision", "annotations"),
        ),
        "recovery_summary.csv": (
            report_rows["recovery_summary"],
            ("source_dataset", "original_class_name", "recovered_annotations"),
        ),
        "duplicate_summary.csv": (
            report_rows["duplicate_summary"],
            (
                "candidate_images",
                "exact_groups",
                "excluded_images",
                "pilot_overlap_images",
            ),
        ),
        "pilot_leakage_report.csv": (
            [],
            DUPLICATE_COLUMNS,
        ),
        "validation_issues.csv": (
            issues,
            VALIDATION_COLUMNS,
        ),
    }
    for name, (rows, columns) in report_specs.items():
        write_csv(report_candidate / name, rows, columns)

    included = [row for row in manifest if row["decision"] in INCLUDED_DECISIONS]
    included_images = {
        Path(row["consolidated_image_path"]).name for row in included
    }
    source_distribution = Counter(
        row["source_dataset"]
        for row in {
            name: next(
                item for item in included if Path(item["consolidated_image_path"]).name == name
            )
            for name in included_images
        }.values()
    )
    decision_fp = decisions_fingerprint(reviews)
    pilot_paths = sorted(path for path in pilot_root.rglob("*") if path.is_file())
    pilot_fp = source_files_fingerprint(pilot_paths)
    counts = {
        **base_summary["counts"],
        "images_included": len(included_images),
        "images_excluded": base_summary["counts"]["images_considered"]
        - len(included_images),
        "annotations_included": len(included),
        "annotations_excluded": sum(
            row["decision"].startswith("exclude") for row in manifest
        ),
        "annotations_recovered": sum(
            row["decision"] == "recover_from_coco" for row in manifest
        ),
        "manual_review_rows": 0,
        "mandatory_visual_review_rows": 0,
    }
    summary = {
        **base_summary,
        "schema_version": 3,
        "name": "doctor_maiz_leaf_segmentation_final",
        "paths": {
            **base_summary["paths"],
            "dataset_root": str(final_dataset_root),
            "report_root": str(final_report_root),
        },
        "counts": counts,
        "source_distribution": dict(sorted(source_distribution.items())),
        "human_reviews": {
            "rows_processed": 36,
            "unique_cases": len(reviews),
            "decision_counts": dict(
                sorted(Counter(row["reviewer_decision"] for row in reviews).items())
            ),
            "pending": 0,
            "invalid": 0,
            "contradictions": 0,
            "fingerprint": decision_fp,
            "application": application,
        },
        "validation": validation,
        "pilot_fingerprint": pilot_fp,
        "decisions_applied_from_original_sources": True,
    }
    selection_summary = {
        "schema_version": 2,
        "seed": seed,
        "status": "final_pool_without_splits",
        "target_class": {"id": TARGET_CLASS_ID, "name": TARGET_CLASS_NAME},
        "total_images": counts["images_included"],
        "total_masks": counts["annotations_included"],
        "source_distribution": summary["source_distribution"],
        "human_decisions": summary["human_reviews"],
        "splits_created": False,
        "training_performed": False,
    }
    (candidate_root / "manifests" / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    eda_summary = json.loads((eda_root / "summary.json").read_text(encoding="utf-8"))
    lock = build_dataset_lock(
        candidate_root,
        summary,
        eda_summary,
        review_summary,
        decisions_applied_from_sources=True,
    )
    lock["decision_fingerprint"] = decision_fp
    lock["pilot_fingerprint"] = pilot_fp
    lock["validation"] = validation
    lock["source_distribution"] = summary["source_distribution"]
    lock["blockers"] = []
    lock["status"] = "ready_for_split_generation"
    write_dataset_lock(candidate_root / "manifests" / "dataset_lock.json", lock)
    (candidate_root / "README.md").write_text(
        _final_readme(summary, lock),
        encoding="utf-8",
    )
    summary["global_fingerprint"] = lock["global_fingerprint"]
    (report_candidate / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _rewrite_report_paths(
        candidate_root,
        candidate_root,
        final_dataset_root,
    )
    _rewrite_report_paths(
        report_candidate,
        candidate_root,
        final_dataset_root,
    )
    _rewrite_report_paths(
        report_candidate,
        report_candidate,
        final_report_root,
    )
    return {
        "summary": summary,
        "lock": lock,
        "dataset_fingerprint": dataset_fingerprint(candidate_root),
    }


def finalize_segmentation_dataset(
    external_root: Path,
    pilot_root: Path,
    eda_root: Path,
    dataset_root: Path,
    report_root: Path,
    config_path: Path,
    *,
    seed: int,
) -> dict[str, object]:
    """Rebuild twice, compare fingerprints, and atomically publish the final pool."""
    external_root = external_root.resolve()
    pilot_root = pilot_root.resolve()
    eda_root = eda_root.resolve()
    dataset_root = dataset_root.resolve()
    report_root = report_root.resolve()
    config_path = config_path.resolve()
    manifests = dataset_root / "manifests"
    mandatory = read_review_manifest(manifests / "mandatory_visual_review.csv")
    general = read_review_manifest(manifests / "manual_review.csv")
    reviews = validate_review_manifests(mandatory, general)
    if (
        reviews["mandatory_total"] != 2
        or reviews["general_total"] != 34
        or reviews["unique_total"] != 35
        or reviews["pending"]
        or reviews["invalid"]
    ):
        raise RuntimeError(f"Gate humano incompleto o inesperado: {reviews}")

    source_before = source_files_fingerprint(
        sorted(path for path in external_root.rglob("*") if path.is_file())
    )
    pilot_before = source_files_fingerprint(
        sorted(path for path in pilot_root.rglob("*") if path.is_file())
    )
    with tempfile.TemporaryDirectory(
        prefix=".segmentation_finalization_",
        dir=dataset_root.parent,
    ) as temporary:
        staging = Path(temporary)
        results = []
        for iteration in (1, 2):
            candidate = staging / f"candidate_{iteration}"
            reports = staging / f"reports_{iteration}"
            results.append(
                _materialize_final_candidate(
                    external_root=external_root,
                    pilot_root=pilot_root,
                    eda_root=eda_root,
                    candidate_root=candidate,
                    report_candidate=reports,
                    final_dataset_root=dataset_root,
                    final_report_root=report_root,
                    config_path=config_path,
                    human_manifest_root=manifests,
                    review_summary=reviews,
                    seed=seed,
                )
            )
        first_fp = results[0]["dataset_fingerprint"]
        second_fp = results[1]["dataset_fingerprint"]
        if first_fp != second_fp:
            raise RuntimeError(
                f"Reconstrucción no determinista: {first_fp} != {second_fp}"
            )
        source_after = source_files_fingerprint(
            sorted(path for path in external_root.rglob("*") if path.is_file())
        )
        pilot_after = source_files_fingerprint(
            sorted(path for path in pilot_root.rglob("*") if path.is_file())
        )
        if source_before != source_after or pilot_before != pilot_after:
            raise RuntimeError("Las fuentes originales o el piloto cambiaron")
        first = results[0]
        first["summary"]["deterministic_rebuild"] = True
        first["summary"]["source_tree_preserved"] = source_before
        first["summary"]["pilot_tree_preserved"] = pilot_before
        (staging / "reports_1" / "summary.json").write_text(
            json.dumps(first["summary"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _publish(
            staging / "candidate_1",
            staging / "reports_1",
            dataset_root,
            report_root,
        )
    return results[0]

"""Local-only safety tests for the cloud segmentation package."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.package import build_leaf_segmentation_cloud_package as package
from src.training.segmentation_preflight import verify_cloud_training_payload

ROOT = Path(__file__).resolve().parents[2]
CLOUD = ROOT / "cloud_training"


@pytest.mark.requires_dataset
def test_cloud_payload_fingerprints_are_valid_without_all_tree() -> None:
    report = verify_cloud_training_payload(
        ROOT / "data" / "leaf_detection" / "detector_dataset"
    )
    assert report["passed"]
    assert report["verified_manifest_rows"] == 1155


def test_every_shell_script_is_strict_and_syntax_valid() -> None:
    for path in sorted(CLOUD.glob("*.sh")):
        source = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in source
        result = subprocess.run(
            ["bash", "-n", path.name],
            cwd=str(CLOUD),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_shell_training_guards_precede_all_script_actions() -> None:
    cases = {
        "smoke_train.sh": "CONFIRM_SEGMENTATION_SMOKE_TRAINING",
        "train.sh": "CONFIRM_SEGMENTATION_TRAINING",
        "resume_train.sh": "CONFIRM_SEGMENTATION_TRAINING",
    }
    for script, confirmation in cases.items():
        source = (CLOUD / script).read_text(encoding="utf-8")
        guard = source.index(confirmation)
        first_action = source.index("source ")
        assert guard < first_action
        assert "exit 2" in source[guard:first_action]


def test_internal_test_has_a_single_use_gate() -> None:
    """El test interno se evalúa una vez: repetirlo permitiría ajustar sobre él."""
    source = (CLOUD / "evaluate_test.sh").read_text(encoding="utf-8")
    guard = source.index("test_summary.json")
    invocation = source.index("run_ultralytics.py")
    assert guard < invocation
    assert "FORCE_INTERNAL_TEST_RERUN" in source[:invocation]
    assert "exit 2" in source[guard:invocation]


def test_validation_delegates_to_the_single_retained_test_implementation() -> None:
    """`validate` y `test` son la misma evaluación: una sola implementación y un guard."""
    validate = (CLOUD / "validate.sh").read_text(encoding="utf-8")
    assert "evaluate_test.sh" in validate
    assert "run_ultralytics.py" not in validate

    source = (CLOUD / "evaluate_test.sh").read_text(encoding="utf-8")
    assert "val_summary.json" not in source
    assert "--split test" in source
    assert "--split val" not in source
    guard = source.index("test_summary.json")
    invocation = source.index("run_ultralytics.py")
    assert guard < invocation
    assert "exit 2" in source[guard:invocation]
    assert "FORCE_INTERNAL_TEST_RERUN" in source[guard:invocation]


def test_cloud_scripts_reuse_the_bootstrap_environment() -> None:
    source = (CLOUD / "lib.sh").read_text(encoding="utf-8")
    assert ".venv-cloud/bin/python" in source
    assert "bin/activate" not in source
    assert "VIRTUAL_ENV" not in source


def test_cloud_yaml_configs_are_deterministic_and_safe() -> None:
    smoke = yaml.safe_load((CLOUD / "configs/smoke_yolo26n_seg.yaml").read_text())
    train = yaml.safe_load((CLOUD / "configs/train_yolo26n_seg.yaml").read_text())
    validate = yaml.safe_load(
        (CLOUD / "configs/validate_yolo26n_seg.yaml").read_text()
    )
    assert smoke["epochs"] == 1 and smoke["batch"] == -1
    assert train["epochs"] == 150 and train["batch"] == -1
    assert train["seed"] == smoke["seed"] == validate["seed"] == 42
    assert train["task"] == smoke["task"] == validate["task"] == "segment"
    assert validate["split"] == "test"
    assert validate["name"] == "yolo26n_seg_test"
    assert "pilot" not in str(smoke) + str(train) + str(validate)


def test_bootstrap_uses_one_version_source_and_checks_cuda_before_and_after() -> None:
    source = (CLOUD / "bootstrap_cloud.sh").read_text(encoding="utf-8")
    requirements = {
        line.strip()
        for line in (CLOUD / "requirements/ultralytics.in")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert requirements == {
        "ultralytics==8.4.104",
        "faster-coco-eval==1.7.2",
    }
    assert 'ULTRALYTICS_VERSION="8.4.104"' not in source
    assert "requirements/ultralytics.in" in source
    assert "FASTER_COCO_EVAL_SPEC" in source
    assert "metadata.version(\"faster-coco-eval\")" in source
    assert source.count("import torchvision") >= 2
    assert source.count("torch.cuda.is_available()") >= 2
    assert "pip install --dry-run" in source
    assert "runtime_constraints.txt" in source


def test_cloud_preflight_uses_detection_independent_segment_forward() -> None:
    source = (
        ROOT / "scripts/pipeline/leaf_segmentation_cloud_preflight.py"
    ).read_text(encoding="utf-8")
    assert "model.predict(" not in source
    assert "torch.zeros(" in source
    assert "segmentation_head_verified" in source
    assert "forward_finite" in source


def test_full_train_script_accepts_only_config_environment() -> None:
    source = (CLOUD / "train.sh").read_text(encoding="utf-8")
    assert 'CONFIG_PATH="${CONFIG:-' in source
    assert '"$#" -ne 0' in source
    assert '--config "${CONFIG_PATH}"' in source
    assert "configs/train_yolo26n_seg.yaml" not in source


@pytest.mark.requires_dataset
def test_allow_list_excludes_protected_and_historical_trees() -> None:
    relatives = {
        path.relative_to(ROOT).as_posix() for path in package.collect_payload(ROOT)
    }
    assert any(
        path.startswith("data/leaf_detection/detector_dataset/images/train/")
        for path in relatives
    )
    assert not any("/all/" in f"/{path}/" for path in relatives)
    assert not any("external_sources" in path for path in relatives)
    assert not any(path.startswith("data/leaf_detection/pilot/") for path in relatives)
    assert not any(path.startswith("outputs/") for path in relatives)
    assert not any(".venv" in path or "__pycache__" in path for path in relatives)


def test_tar_metadata_is_deterministic() -> None:
    first = package.tar_info(Path("payload/file.sh"), 123, True)
    second = package.tar_info(Path("payload/file.sh"), 123, True)
    assert (first.name, first.size, first.mtime, first.uid, first.gid, first.mode) == (
        second.name,
        second.size,
        second.mtime,
        second.uid,
        second.gid,
        second.mode,
    )
    assert first.mtime == first.uid == first.gid == 0


@pytest.mark.requires_pilot
def test_pilot_transport_manifest_is_separate() -> None:
    manifest = package.pilot_manifest(ROOT)
    assert manifest["included_in_training_package"] is False
    assert manifest["purpose"] == "external_held_out_evaluation_transport_only"
    assert manifest["file_count"] == 211

"""Safety and orchestration tests for the project Makefile."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = PROJECT_ROOT / "Makefile"

SEGMENTATION_TARGETS = {
    "help",
    "leaf-segmentation-status",
    "leaf-segmentation-verify-locks",
    "leaf-segmentation-verify-splits",
    "leaf-segmentation-preflight",
    "leaf-segmentation-cloud-package",
    "leaf-segmentation-cloud-package-verify",
    "leaf-segmentation-cloud-package-list",
    "leaf-segmentation-cloud-clean-temp",
    "leaf-segmentation-cloud-bootstrap",
    "leaf-segmentation-cloud-preflight",
    "leaf-segmentation-cloud-smoke",
    "leaf-segmentation-cloud-train",
    "leaf-segmentation-cloud-resume",
    "leaf-segmentation-cloud-validate",
    "leaf-segmentation-cloud-test",
    "leaf-segmentation-cloud-results",
    "leaf-segmentation-cloud-checksums",
    "leaf-segmentation-pilot-evaluate",
    "leaf-segmentation-cloud-prepare",
    "leaf-segmentation-cloud-check",
    "leaf-segmentation-downstream-metrics",
    "leaf-segmentation-hf-publish",
    "leaf-segmentation-hf-download",
    "leaf-segmentation-promote-checkpoint",
    "leaf-segmentation-modal-seed",
    "leaf-segmentation-modal-verify-dataset",
    "leaf-segmentation-modal-promote",
    "leaf-segmentation-modal-preflight",
    "leaf-segmentation-modal-smoke",
    "leaf-segmentation-modal-train",
    "leaf-segmentation-modal-resume",
    "leaf-segmentation-modal-validate",
    "leaf-segmentation-modal-results",
    "leaf-segmentation-modal-checksums",
    "leaf-segmentation-modal-download",
}


class MakefileSafetyTests(TestCase):
    def _make(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["make", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )

    def _dry_run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self._make("-n", *arguments)

    @staticmethod
    def _decoded(output: str) -> str:
        """Deshace la doble codificación que introduce printf en Git para Windows.

        El Makefile es UTF-8 válido, pero ese printf lee sus bytes como cp1252 y los
        vuelve a emitir en UTF-8, así que la salida llega con mojibake en los acentos.
        """
        try:
            return output.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return output

    def test_help_works_and_lists_targets(self) -> None:
        result = self._make("help")
        self.assertEqual(result.returncode, 0, result.stderr)
        printed = self._decoded(result.stdout)
        for heading in (
            "LOCAL / SEGURO:",
            "CLOUD / SIN ENTRENAR:",
            "ENTRENAMIENTO / CONFIRMACIÓN OBLIGATORIA:",
        ):
            self.assertIn(heading, printed)

    def test_status_is_read_only(self) -> None:
        watched = [
            PROJECT_ROOT
            / "data/leaf_detection/detector_dataset/manifests/dataset_lock.json",
            PROJECT_ROOT
            / "data/leaf_detection/detector_dataset/manifests/split_lock.json",
        ]
        before = [(path.stat().st_mtime_ns, path.read_bytes()) for path in watched]
        result = self._make(
            "leaf-segmentation-status",
            f"PYTHON={Path(sys.executable).as_posix()}",
        )
        after = [(path.stat().st_mtime_ns, path.read_bytes()) for path in watched]
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, after)
        self.assertIn('"train": 809', result.stdout)
        self.assertIn('"val": 173', result.stdout)
        self.assertIn('"test": 173', result.stdout)

    def test_python_and_segmentation_variables_are_propagated(self) -> None:
        result = self._dry_run(
            "leaf-segmentation-cloud-preflight",
            "PYTHON=/opt/remote/bin/python",
            "SEGMENTATION_DEVICE=7",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PYTHON="/opt/remote/bin/python"', result.stdout)
        self.assertIn('SEGMENTATION_DEVICE="7"', result.stdout)

        status = self._dry_run(
            "leaf-segmentation-status",
            "PYTHON=/opt/remote/bin/python",
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn(
            "/opt/remote/bin/python scripts/package/leaf_segmentation_make.py",
            status.stdout,
        )

    def test_full_training_propagates_the_frozen_config(self) -> None:
        config = (
            "outputs/leaf_detection/segmenter/configs/"
            "train_yolo26n_seg.final.yaml"
        )
        result = self._dry_run(
            "leaf-segmentation-cloud-train",
            "CONFIRM_SEGMENTATION_TRAINING=1",
            f"CONFIG={config}",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'CONFIG="{config}"', result.stdout)
        self.assertNotIn(
            '--config "cloud_training/configs/train_yolo26n_seg.yaml"',
            result.stdout,
        )

    def test_protected_segmentation_targets_fail_before_rendering_scripts(self) -> None:
        cases = {
            "leaf-segmentation-cloud-smoke": (
                "CONFIRM_SEGMENTATION_SMOKE_TRAINING=1",
                "smoke_train.sh",
            ),
            "leaf-segmentation-cloud-train": (
                "CONFIRM_SEGMENTATION_TRAINING=1",
                "train.sh",
            ),
            "leaf-segmentation-cloud-resume": (
                "CONFIRM_SEGMENTATION_TRAINING=1",
                "resume_train.sh",
            ),
            "leaf-segmentation-pilot-evaluate": (
                "CONFIRM_PILOT_EVALUATION=1",
                "leaf_segmentation_pilot_evaluate.py",
            ),
        }
        for target, (confirmation, script) in cases.items():
            result = self._dry_run(target)
            self.assertNotEqual(result.returncode, 0, target)
            self.assertIn(confirmation, result.stderr)
            self.assertNotIn(script, result.stdout)

    def test_confirmations_must_equal_exactly_one(self) -> None:
        cases = (
            (
                "CONFIRM_SEGMENTATION_SMOKE_TRAINING",
                "leaf-segmentation-cloud-smoke",
            ),
            ("CONFIRM_SEGMENTATION_TRAINING", "leaf-segmentation-cloud-train"),
            ("CONFIRM_PILOT_EVALUATION", "leaf-segmentation-pilot-evaluate"),
        )
        for variable, target in cases:
            for invalid in ("true", "1 2"):
                result = self._dry_run(target, f"{variable}={invalid}")
                self.assertNotEqual(result.returncode, 0)

    def test_modal_training_targets_keep_the_local_confirmation_guards(self) -> None:
        cases = {
            "leaf-segmentation-modal-smoke": (
                "CONFIRM_SEGMENTATION_SMOKE_TRAINING=1",
                "modal_training.py::smoke",
            ),
            "leaf-segmentation-modal-train": (
                "CONFIRM_SEGMENTATION_TRAINING=1",
                "modal_training.py::train",
            ),
            "leaf-segmentation-modal-resume": (
                "CONFIRM_SEGMENTATION_TRAINING=1",
                "modal_training.py::resume",
            ),
            "leaf-segmentation-modal-experiment": (
                "CONFIRM_SEGMENTATION_TRAINING=1",
                "modal_training.py::experiment",
            ),
            "leaf-segmentation-modal-experiment-resume": (
                "CONFIRM_SEGMENTATION_TRAINING=1",
                "modal_training.py::resume_experiment",
            ),
        }
        for target, (confirmation, remote_function) in cases.items():
            blocked = self._dry_run(target)
            self.assertNotEqual(blocked.returncode, 0, target)
            self.assertIn(confirmation, blocked.stderr)
            self.assertNotIn(remote_function, blocked.stdout)

        smoke = self._dry_run(
            "leaf-segmentation-modal-smoke",
            "CONFIRM_SEGMENTATION_SMOKE_TRAINING=1",
        )
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertIn(
            "modal run modal_training.py::smoke --confirm true",
            smoke.stdout,
        )
        for target in (
            "leaf-segmentation-modal-train",
            "leaf-segmentation-modal-resume",
            "leaf-segmentation-modal-experiment",
            "leaf-segmentation-modal-experiment-resume",
        ):
            rendered = self._dry_run(
                target,
                "CONFIRM_SEGMENTATION_TRAINING=1",
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertIn("modal run --detach", rendered.stdout)
            self.assertIn("--confirm true", rendered.stdout)

    def test_modal_gpu_allowlist_blocks_invalid_or_ambiguous_values(self) -> None:
        for invalid in ("T4", "H100", "A10 L4"):
            result = self._dry_run(
                "leaf-segmentation-modal-preflight",
                f"MODAL_SEGMENTATION_GPU={invalid}",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("A10, L4 o A100", result.stderr)
            self.assertNotIn("modal_training.py::preflight", result.stdout)

    def test_modal_flow_seeds_from_hugging_face_and_pulls_outputs(self) -> None:
        seed = self._dry_run("leaf-segmentation-modal-seed")
        self.assertEqual(seed.returncode, 0, seed.stderr)
        self.assertIn("modal run modal_training.py::seed_dataset", seed.stdout)
        self.assertNotIn("volume put", seed.stdout)
        self.assertNotIn(".tar.gz", seed.stdout)

        verify = self._dry_run("leaf-segmentation-modal-verify-dataset")
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertIn("modal run modal_training.py::verify_dataset", verify.stdout)

        download = self._dry_run("leaf-segmentation-modal-download")
        self.assertEqual(download.returncode, 0, download.stderr)
        self.assertIn("modal volume get --force", download.stdout)
        self.assertIn("doctor-maiz-leaf-segmentation-outputs", download.stdout)
        self.assertIn("leaf_detection", download.stdout)

    def test_safe_prepare_has_no_accidental_cloud_or_training_dependencies(self) -> None:
        result = self._dry_run("leaf-segmentation-cloud-prepare")
        self.assertEqual(result.returncode, 0, result.stderr)
        for forbidden in (
            "bootstrap_cloud.sh",
            "preflight_cloud.sh",
            "smoke_train.sh",
            "train.sh",
            "resume_train.sh",
            "leaf_segmentation_pilot_evaluate.py",
        ):
            self.assertNotIn(forbidden, result.stdout)
        self.assertIn("build_leaf_segmentation_cloud_package.py", result.stdout)
        self.assertIn("package-verify", result.stdout)

    def test_package_recipe_has_no_gpu_install_download_or_training_action(self) -> None:
        result = self._dry_run("leaf-segmentation-cloud-package")
        self.assertEqual(result.returncode, 0, result.stderr)
        for forbidden in ("nvidia-smi", "pip install", "YOLO(", " train "):
            self.assertNotIn(forbidden, result.stdout)

    def test_downstream_metrics_requires_predictions_and_never_trains(self) -> None:
        result = self._dry_run("leaf-segmentation-downstream-metrics")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PREDICTIONS=", result.stderr)
        rendered = self._dry_run(
            "leaf-segmentation-downstream-metrics",
            "PREDICTIONS=/tmp/predicciones",
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn("leaf_segmentation_downstream_metrics.py", rendered.stdout)
        self.assertIn('--split "val"', rendered.stdout)
        for forbidden in ("yolo ", "train", "pip install", "nvidia-smi"):
            self.assertNotIn(forbidden, rendered.stdout)

    def test_clean_outputs_requires_explicit_confirmation(self) -> None:
        result = self._dry_run("clean-outputs")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONFIRM_CLEAN_OUTPUTS=1", result.stderr)
        self.assertNotIn("rm -rf outputs/", result.stdout)
        confirmed = self._dry_run("clean-outputs", "CONFIRM_CLEAN_OUTPUTS=1")
        self.assertEqual(confirmed.returncode, 0, confirmed.stderr)
        self.assertIn("rm -rf outputs/", confirmed.stdout)

    def test_all_orchestration_targets_are_phony_and_paths_are_portable(self) -> None:
        source = MAKEFILE.read_text(encoding="utf-8")
        phony_block = source.split(".PHONY:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
        for target in SEGMENTATION_TARGETS:
            self.assertIn(target, phony_block)
        self.assertNotIn("/home/", source)

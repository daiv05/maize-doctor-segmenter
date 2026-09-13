"""Static contract tests for the Modal leaf-segmentation control plane."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODAL_TRAINING = PROJECT_ROOT / "modal_training.py"
SOURCE = MODAL_TRAINING.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _remote_functions() -> dict[str, ast.FunctionDef]:
    functions: dict[str, ast.FunctionDef] = {}
    for node in TREE.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
            and decorator.func.attr == "function"
            for decorator in node.decorator_list
        ):
            functions[node.name] = node
    return functions


def _function_source(name: str) -> str:
    node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    source = ast.get_source_segment(SOURCE, node)
    assert source is not None
    return source


def _version_validator():
    constants: dict[str, object] = {}
    validator: ast.FunctionDef | None = None
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.startswith("EXPECTED_")
                ):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
        elif isinstance(node, ast.FunctionDef) and node.name == "_validate_image_versions":
            validator = node
    assert validator is not None
    namespace = dict(constants)
    code = compile(
        ast.Module(body=[validator], type_ignores=[]),
        filename=str(MODAL_TRAINING),
        mode="exec",
    )
    exec(code, namespace)
    return namespace["_validate_image_versions"]


VALIDATE_IMAGE_VERSIONS = _version_validator()


def _project_environment_function(repo_anchor: Path):
    function = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_project_environment"
    )
    namespace = {
        "REPO_ANCHOR": str(repo_anchor),
        "DATASET_MOUNT_PATH": "/data",
        "OUTPUTS_MOUNT_PATH": "/outputs",
        "DATASET_ROOT": Path("/data/leaf_detection/detector_dataset"),
        "SEGMENTATION_OUTPUT_ROOT": Path("/outputs/leaf_detection"),
        "os": os,
        "sys": sys,
    }
    code = compile(
        ast.Module(body=[function], type_ignores=[]),
        filename=str(MODAL_TRAINING),
        mode="exec",
    )
    exec(code, namespace)
    return namespace["_project_environment"]


class ModalTrainingContractTests(TestCase):
    def test_app_volumes_and_mounts_are_exact(self) -> None:
        self.assertIn("app = modal.App(APP_NAME)", SOURCE)
        self.assertIn('APP_NAME = "doctor-maiz-leaf-segmentation"', SOURCE)
        self.assertIn(
            'DATASET_VOLUME_NAME = "doctor-maiz-leaf-segmentation-data"',
            SOURCE,
        )
        self.assertIn(
            'OUTPUTS_VOLUME_NAME = "doctor-maiz-leaf-segmentation-outputs"',
            SOURCE,
        )
        self.assertIn('DATASET_MOUNT_PATH = "/data"', SOURCE)
        self.assertIn('OUTPUTS_MOUNT_PATH = "/outputs"', SOURCE)
        self.assertIn("DATASET_MOUNT = Path(DATASET_MOUNT_PATH)", SOURCE)
        self.assertIn("OUTPUTS_MOUNT = Path(OUTPUTS_MOUNT_PATH)", SOURCE)
        self.assertIn(
            'DATASET_ROOT = DATASET_MOUNT / "leaf_detection" / "detector_dataset"',
            SOURCE,
        )
        self.assertIn(
            'SEGMENTATION_OUTPUT_ROOT = OUTPUTS_MOUNT / "leaf_detection"',
            SOURCE,
        )
        for removed in ("PACKAGE_SHA256", "PACKAGE_VERSION", "INCOMING_ROOT", "tarfile"):
            self.assertNotIn(removed, SOURCE)

    def test_dataset_identity_is_verified_on_the_mount(self) -> None:
        verify_source = _function_source("_verify_mounted_dataset")
        self.assertIn("verify_cloud_training_payload(DATASET_ROOT)", verify_source)
        self.assertIn("EXPECTED_PARENT_FINGERPRINT", verify_source)
        self.assertIn("EXPECTED_TEST_FINGERPRINT", verify_source)
        gate_source = _function_source("_dataset_gate")
        self.assertIn("_verify_mounted_dataset()", gate_source)
        self.assertIn("_write_json(DATASET_MARKER", gate_source)

    def test_seeding_downloads_from_hugging_face_only(self) -> None:
        seed_source = _function_source("seed_dataset")
        self.assertIn("download_dataset(", seed_source)
        self.assertIn("HF_DATASET_REPO", seed_source)
        self.assertIn("dataset_volume.commit()", seed_source)
        self.assertNotIn("outputs_volume.commit()", seed_source)

    def test_image_is_reproducible_and_never_embeds_local_data(self) -> None:
        self.assertIn(
            'BASE_IMAGE_TAG = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"',
            SOURCE,
        )
        self.assertIn(
            '"sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"',
            SOURCE,
        )
        for version in (
            'EXPECTED_TORCH = "2.6.0"',
            'EXPECTED_TORCHVISION = "0.21.0"',
            'EXPECTED_ULTRALYTICS = "8.4.104"',
            'EXPECTED_FASTER_COCO_EVAL = "1.7.2"',
            'EXPECTED_CUDA = "12.4"',
            'EXPECTED_CUDA_LOCAL = "cu124"',
        ):
            self.assertIn(version, SOURCE)
        for forbidden in (
            ".venv-cloud",
            'add_local_dir("data"',
            'add_local_dir("outputs"',
            'add_local_dir("notebooks"',
            "add_local_dir(\"public\"",
        ):
            self.assertNotIn(forbidden, SOURCE)
        for shipped in (
            '.add_local_file("Makefile"',
            '.add_local_dir("config"',
            '.add_local_dir("cloud_training"',
            '.add_local_python_source("src", "scripts")',
        ):
            self.assertIn(shipped, SOURCE)
        self.assertIn("IMAGE_RECIPE_SHA256", SOURCE)
        self.assertIn('"modal_image_id": _modal_object_id(modal_image)', SOURCE)
        self.assertIn("python -m pip freeze", SOURCE)

    def test_cuda_local_versions_match_their_base_releases(self) -> None:
        VALIDATE_IMAGE_VERSIONS(
            {
                "python": "3.11.13",
                "torch": "2.6.0+cu124",
                "torch_import": "2.6.0+cu124",
                "torchvision": "0.21.0+cu124",
                "torchvision_import": "0.21.0+cu124",
                "ultralytics": "8.4.104",
                "faster-coco-eval": "1.7.2",
                "torch_cuda": "12.4",
            },
            (3, 11, 13),
        )

    def test_different_base_release_is_rejected(self) -> None:
        actual = self._valid_image_versions()
        actual["torch"] = "2.6.1+cu124"
        with self.assertRaisesRegex(RuntimeError, r"torch='2\.6\.1\+cu124'"):
            VALIDATE_IMAGE_VERSIONS(actual, (3, 11, 13))

    def test_startswith_lookalike_release_is_rejected(self) -> None:
        actual = self._valid_image_versions()
        actual["torch"] = "2.6.01+cu124"
        with self.assertRaisesRegex(RuntimeError, r"torch='2\.6\.01\+cu124'"):
            VALIDATE_IMAGE_VERSIONS(actual, (3, 11, 13))

    def test_different_cuda_is_rejected(self) -> None:
        actual = self._valid_image_versions()
        actual["torch_cuda"] = "12.6"
        with self.assertRaisesRegex(RuntimeError, r"torch\.version\.cuda='12\.6'"):
            VALIDATE_IMAGE_VERSIONS(actual, (3, 11, 13))

    def test_missing_cuda_local_suffix_is_rejected(self) -> None:
        actual = self._valid_image_versions()
        actual["torch_import"] = "2.6.0"
        with self.assertRaisesRegex(RuntimeError, r"torch_import='2\.6\.0'"):
            VALIDATE_IMAGE_VERSIONS(actual, (3, 11, 13))

    def test_different_ultralytics_is_rejected(self) -> None:
        actual = self._valid_image_versions()
        actual["ultralytics"] = "8.4.105"
        with self.assertRaisesRegex(RuntimeError, r"ultralytics='8\.4\.105'"):
            VALIDATE_IMAGE_VERSIONS(actual, (3, 11, 13))

    def test_different_faster_coco_eval_is_rejected(self) -> None:
        actual = self._valid_image_versions()
        actual["faster-coco-eval"] = "1.7.1"
        with self.assertRaisesRegex(RuntimeError, r"faster-coco-eval='1\.7\.1'"):
            VALIDATE_IMAGE_VERSIONS(actual, (3, 11, 13))

    def test_build_validation_does_not_require_a_gpu(self) -> None:
        build_source = _function_source("_validate_modal_image_versions")
        self.assertNotIn("torch.cuda.is_available", build_source)

    def test_image_uses_module_level_run_function_for_version_check(self) -> None:
        validator = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_modal_image_versions"
        )
        self.assertIsInstance(validator, ast.FunctionDef)
        self.assertIn(
            ".run_function(_validate_modal_image_versions)",
            SOURCE,
        )
        self.assertNotIn("_IMAGE_VERSION_CHECK", SOURCE)

    def test_python_validation_is_never_passed_as_dockerfile_commands(self) -> None:
        self.assertNotIn(".dockerfile_commands(", SOURCE)
        self.assertNotIn("setup_dockerfile_commands=", SOURCE)
        image_source = SOURCE.split("modal_image =", maxsplit=1)[1].split(
            "app = modal.App", maxsplit=1
        )[0]
        self.assertNotIn("python -c", image_source)
        run_commands_source = image_source.split(".run_commands(", 1)[1].split(
            ")", 1
        )[0]
        self.assertNotIn("_validate_modal_image_versions", run_commands_source)

    def test_runtime_preflight_still_requires_cuda(self) -> None:
        runtime_source = _function_source("_runtime_report")
        self.assertIn("if require_gpu:", runtime_source)
        self.assertIn("torch.cuda.is_available()", runtime_source)

    def test_project_root_is_first_without_duplicate_in_pythonpath(self) -> None:
        project_root = Path("/root")
        project_environment = _project_environment_function(project_root)
        previous = os.pathsep.join(
            (
                "/existing/one",
                str(project_root),
                "/existing/two",
                str(project_root),
            )
        )
        with patch.dict(
            os.environ,
            {
                "PYTHONPATH": previous,
                "DOCTOR_MAIZ_EXISTING_ENV": "preserved",
            },
        ):
            environment = project_environment()

        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep),
            [str(project_root), "/existing/one", "/existing/two"],
        )
        self.assertEqual(environment["DOCTOR_MAIZ_EXISTING_ENV"], "preserved")

    def test_project_environment_has_no_hardcoded_local_path(self) -> None:
        environment_source = _function_source("_project_environment")
        self.assertNotIn("/home/desarrolloab", environment_source)
        self.assertIn("pythonpath.insert(0, REPO_ANCHOR)", environment_source)
        self.assertIn('"PROJECT_DATA_ROOT": DATASET_MOUNT_PATH', environment_source)
        self.assertIn('"OUTPUT_ROOT": OUTPUTS_MOUNT_PATH', environment_source)

    def test_project_environment_allows_importing_src_config(self) -> None:
        project_environment = _project_environment_function(PROJECT_ROOT)
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/pythonpath"}):
            environment = project_environment()
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.config import PROJECT_ROOT; print(PROJECT_ROOT)",
            ],
            cwd=tempfile.gettempdir(),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), str(PROJECT_ROOT))

    def test_all_remote_operations_receive_project_environment(self) -> None:
        self.assertIn("env=environment", _function_source("_run"))
        self.assertIn(
            "environment=_project_environment()",
            _function_source("_make"),
        )
        self.assertIn("_make(", _function_source("_execute"))
        for operation in (
            "preflight",
            "smoke",
            "train",
            "resume",
            "validate",
            "results",
            "checksums",
        ):
            self.assertIn("_execute(", _function_source(operation), operation)

    def test_initial_weights_survive_the_ephemeral_working_directory(self) -> None:
        self.assertIn(
            "PERSISTENT_INITIAL_WEIGHTS = (\n"
            '    SEGMENTATION_OUTPUT_ROOT / "initial_weights" / INITIAL_WEIGHTS_NAME\n'
            ")",
            SOURCE,
        )
        restore_source = _function_source("_restore_initial_weights")
        self.assertIn("PERSISTENT_INITIAL_WEIGHTS.is_file()", restore_source)
        self.assertIn("_sha256(working) == _sha256(PERSISTENT_INITIAL_WEIGHTS)", restore_source)
        persist_source = _function_source("_persist_initial_weights")
        self.assertIn("PERSISTENT_INITIAL_WEIGHTS.parent.mkdir", persist_source)

        execute_source = _function_source("_execute")
        ordered = (
            "_restore_initial_weights()",
            "_make(",
            "_persist_initial_weights()",
            "outputs_volume.commit()",
        )
        offsets = [execute_source.index(step) for step in ordered]
        self.assertEqual(offsets, sorted(offsets))

    def test_dataset_verification_persists_its_marker(self) -> None:
        verify_source = _function_source("verify_dataset")
        self.assertIn("_reload_volumes()", verify_source)
        self.assertIn("_dataset_gate()", verify_source)
        self.assertLess(
            verify_source.index("_dataset_gate()"),
            verify_source.index("outputs_volume.commit()"),
        )

    def test_reload_is_centralized_outside_the_volume_before_access(self) -> None:
        reload_source = _function_source("_reload_volumes")
        self.assertEqual(SOURCE.count("dataset_volume.reload()"), 1)
        self.assertEqual(SOURCE.count("outputs_volume.reload()"), 1)
        self.assertLess(
            reload_source.index('os.chdir("/tmp")'),
            reload_source.index("dataset_volume.reload()"),
        )
        execute_source = _function_source("_execute")
        ordered_operations = (
            "_reload_volumes()",
            "_dataset_gate()",
            "os.chdir(REPO_ANCHOR)",
            "_runtime_report(",
            "_make(",
            "outputs_volume.commit()",
        )
        offsets = [execute_source.index(operation) for operation in ordered_operations]
        self.assertEqual(offsets, sorted(offsets))
        for operation in (
            "preflight",
            "smoke",
            "train",
            "resume",
            "validate",
            "results",
            "checksums",
        ):
            self.assertIn("_execute(", _function_source(operation), operation)

    def test_train_has_no_reload_during_training(self) -> None:
        train_source = _function_source("train")
        self.assertNotIn("reload", train_source)
        self.assertIn("validate_final_config=True", train_source)
        self.assertIn("CONFIRM_SEGMENTATION_TRAINING=1", train_source)

    def test_only_outputs_can_be_cleaned_and_never_the_dataset(self) -> None:
        clean_source = _function_source("clean_outputs")
        self.assertIn("_require_confirmation(confirm", clean_source)
        self.assertIn("shutil.rmtree(SEGMENTATION_OUTPUT_ROOT", clean_source)
        self.assertNotIn("DATASET_ROOT", clean_source)
        self.assertNotIn("DATASET_MOUNT", clean_source)

    def test_promotion_registers_the_evaluated_checkpoint_identity(self) -> None:
        promote_source = _function_source("promote")
        self.assertIn("promote_checkpoint(OUTPUTS_MOUNT)", promote_source)
        self.assertIn("outputs_volume.commit()", promote_source)

    @staticmethod
    def _valid_image_versions() -> dict[str, str | None]:
        return {
            "python": "3.11.13",
            "torch": "2.6.0+cu124",
            "torch_import": "2.6.0+cu124",
            "torchvision": "0.21.0+cu124",
            "torchvision_import": "0.21.0+cu124",
            "ultralytics": "8.4.104",
            "faster-coco-eval": "1.7.2",
            "torch_cuda": "12.4",
        }

    def test_only_allowed_gpu_types_can_be_requested(self) -> None:
        self.assertIn('ALLOWED_GPUS = ("A10", "L4", "A100")', SOURCE)
        self.assertIn(
            'os.getenv("DOCTOR_MAIZ_MODAL_GPU", "A10")',
            SOURCE,
        )
        self.assertIn("MINIMUM_VRAM_BYTES = 12 * 1024**3", SOURCE)
        self.assertNotIn("gpu=[", SOURCE)

    def test_all_required_remote_functions_mount_the_volume(self) -> None:
        functions = _remote_functions()
        self.assertEqual(
            set(functions),
            {
                "seed_dataset",
                "verify_dataset",
                "preflight",
                "smoke",
                "train",
                "experiment",
                "resume",
                "resume_experiment",
                "promote",
                "validate",
                "results",
                "checksums",
                "clean_outputs",
            },
        )
        for name, function in functions.items():
            decorator = function.decorator_list[0]
            assert isinstance(decorator, ast.Call)
            keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
            self.assertIn("volumes", keywords, name)
            if name == "seed_dataset":
                self.assertIsInstance(keywords["volumes"], ast.Dict)
                continue
            self.assertIsInstance(keywords["volumes"], ast.Name)
            self.assertEqual(keywords["volumes"].id, "VOLUME_MOUNTS")

    def test_training_requires_literal_string_confirmation(self) -> None:
        functions = _remote_functions()
        for name in ("smoke", "train", "experiment", "resume", "resume_experiment"):
            function = functions[name]
            defaults = function.args.defaults
            self.assertEqual(len(defaults), 1)
            self.assertIsInstance(defaults[0], ast.Constant)
            self.assertEqual(defaults[0].value, "false")
        self.assertIn('if value != "true":', SOURCE)
        self.assertIn("use --confirm true exactamente", SOURCE)

    def test_validate_enforces_retained_test_contract(self) -> None:
        validate_source = _function_source("validate")
        for contract in (
            '"requested_split") != "test"',
            '"evaluated_split") != "test"',
            '"split") != "test"',
            'locked_counts["image_counts"]["test"]',
            'locked_counts["mask_counts"]["test"]',
            '"evaluated_instance_count") != summary.get("loader_instance_count")',
            "EXPECTED_TEST_FINGERPRINT",
            "expected_best_checkpoint_sha256(OUTPUTS_MOUNT)",
            '"pilot_used") is not False',
            "yolo26n_seg_test",
        ):
            self.assertIn(contract, validate_source)
        self.assertNotIn("val_summary.json", validate_source)

    def test_dataset_and_runtime_guards_are_persistent(self) -> None:
        for contract in (
            'DATASET_ROOT = DATASET_MOUNT / "leaf_detection" / "detector_dataset"',
            'MODAL_RUNTIME_ROOT = SEGMENTATION_OUTPUT_ROOT / "modal_runtime"',
            'DATASET_MARKER = MODAL_RUNTIME_ROOT / "dataset_verification.json"',
            "dataset_volume.reload()",
            "outputs_volume.commit()",
            "runtime_environment.modal.lock",
            "ready_for_smoke_training",
            "memory_total_mib",
            "initial_utilization_percent",
            'payload.get("epochs") != 150',
        ):
            self.assertIn(contract, SOURCE)

    def test_no_remote_test_or_pilot_entrypoint_exists(self) -> None:
        functions = _remote_functions()
        self.assertNotIn("test", functions)
        self.assertNotIn("pilot", functions)

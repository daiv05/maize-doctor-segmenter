"""Unit tests for the JSON-safe summary helpers of the cloud runner."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ultralytics",
        PROJECT_ROOT / "cloud_training" / "run_ultralytics.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_ultralytics", module)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def test_to_serializable_handles_tensors_scalars_and_none() -> None:
    payload = {
        "tensor": runner.to_serializable(_FakeTensor([1.5, 2.0])),
        "none": runner.to_serializable(None),
        "nested": runner.to_serializable([_FakeTensor([0.25]), 3, "x"]),
        "mapping": runner.to_serializable({"seg": _FakeTensor([0.125])}),
        "fallback": runner.to_serializable(object()),
    }
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["tensor"] == [1.5, 2.0]
    assert decoded["none"] is None
    assert decoded["nested"] == [[0.25], 3, "x"]
    assert decoded["mapping"] == {"seg": [0.125]}
    assert isinstance(decoded["fallback"], str)


def test_resolve_trainer_prefers_model_trainer_over_metrics() -> None:
    trainer = SimpleNamespace(args=SimpleNamespace(batch=16), loss_items=_FakeTensor([0.1]))
    model = SimpleNamespace(trainer=trainer)
    metrics = SimpleNamespace()
    assert runner.resolve_trainer(model, metrics) is trainer
    selected = getattr(getattr(runner.resolve_trainer(model, metrics), "args", None), "batch", -1)
    assert selected == 16


def test_resume_manifests_never_collide() -> None:
    """Cada reanudación debe conservar su propio manifiesto, sin sobrescribir."""
    from datetime import datetime, timezone

    directory = Path("/tmp/segmenter")
    first = runner.timestamped_manifest_path(
        directory, datetime(2026, 7, 28, 10, 0, 0, 1, tzinfo=timezone.utc)
    )
    second = runner.timestamped_manifest_path(
        directory, datetime(2026, 7, 28, 10, 0, 0, 2, tzinfo=timezone.utc)
    )
    assert first != second
    assert first.parent == directory
    assert first.name.startswith("resume_manifest_")
    assert first.suffix == ".json"


def test_resolve_trainer_falls_back_to_result_then_none() -> None:
    trainer = SimpleNamespace(args=SimpleNamespace(batch=8))
    model = SimpleNamespace(trainer=None)
    metrics = SimpleNamespace(trainer=trainer)
    assert runner.resolve_trainer(model, metrics) is trainer
    empty_model = SimpleNamespace(trainer=None)
    empty_metrics = SimpleNamespace()
    resolved = runner.resolve_trainer(empty_model, empty_metrics)
    assert resolved is None
    fallback_batch = getattr(getattr(resolved, "args", None), "batch", -1)
    assert fallback_batch == -1


def test_selected_batch_prefers_resolved_positive_batch() -> None:
    trainer = SimpleNamespace(
        batch_size=12,
        args=SimpleNamespace(batch=-1),
    )
    assert runner.selected_positive_batch(trainer) == 12


def test_selected_batch_rejects_autobatch_sentinel_bool_and_zero() -> None:
    for value in (-1, 0, True):
        trainer = SimpleNamespace(
            batch_size=None,
            args=SimpleNamespace(batch=value),
        )
        try:
            runner.selected_positive_batch(trainer)
        except RuntimeError as exc:
            assert "Batch efectivo inválido" in str(exc)
        else:
            raise AssertionError(f"batch inválido aceptado: {value!r}")


def test_finite_numeric_gate_rejects_nan_inf_and_missing_values() -> None:
    assert runner.require_finite_numeric(
        "loss", {"box": [0.5], "seg": _FakeTensor([0.25])}
    ) == [0.5, 0.25]
    for value in (None, [], [math.nan], {"loss": math.inf}):
        try:
            runner.require_finite_numeric("loss", value)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"valor no finito aceptado: {value!r}")


def test_direct_training_modes_require_exact_confirmation(monkeypatch) -> None:
    for mode, variable in (
        ("smoke", "CONFIRM_SEGMENTATION_SMOKE_TRAINING"),
        ("train", "CONFIRM_SEGMENTATION_TRAINING"),
        ("resume", "CONFIRM_SEGMENTATION_TRAINING"),
    ):
        monkeypatch.delenv(variable, raising=False)
        try:
            runner.require_confirmation(mode)
        except RuntimeError as exc:
            assert variable in str(exc)
        else:
            raise AssertionError(f"{mode} fue aceptado sin confirmación")
        monkeypatch.setenv(variable, "true")
        try:
            runner.require_confirmation(mode)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{mode} aceptó una confirmación no exacta")
        monkeypatch.setenv(variable, "1")
        runner.require_confirmation(mode)


def test_experiment_profiles_allow_only_documented_seeds_and_initializers() -> None:
    assert runner.ALLOWED_EXPERIMENT_SEEDS == {7, 42, 1337}
    model, metadata = runner.resolve_initial_model("scratch_yolo26n")
    assert model == "yolo26n-seg.yaml"
    assert metadata["initialization"] == "scratch"
    with pytest.raises(RuntimeError, match="initialization_profile no permitido"):
        runner.resolve_initial_model("download_something_implicit")


def test_train_mode_has_scoped_experiment_manifests_and_sampling() -> None:
    source = inspect.getsource(runner.train_mode)
    assert 'config.pop("experiment_id", None)' in source
    assert "ALLOWED_EXPERIMENT_SEEDS" in source
    assert "materialize_source_balanced_dataset(" in source
    assert '"experiment_manifests"' in source


def test_cuda_initialization_selects_and_initializes_before_reset(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        runner.torch.cuda,
        "is_available",
        lambda: calls.append("is_available") or True,
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "device_count",
        lambda: calls.append("device_count") or 1,
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "set_device",
        lambda index: calls.append(("set_device", index)),
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "init",
        lambda: calls.append("init"),
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "is_initialized",
        lambda: calls.append("is_initialized") or True,
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "current_device",
        lambda: calls.append("current_device") or 0,
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset_peak_memory_stats", device)),
    )

    device = runner.initialize_cuda_and_reset_peak_memory_stats(0)

    assert device == torch.device("cuda:0")
    assert calls == [
        "is_available",
        "device_count",
        ("set_device", 0),
        "init",
        "is_initialized",
        "current_device",
        ("reset_peak_memory_stats", torch.device("cuda:0")),
    ]


@pytest.mark.parametrize("device_index,device_count", [(-1, 1), (1, 1), (0, 0)])
def test_cuda_initialization_rejects_out_of_range_device(
    monkeypatch,
    device_index: int,
    device_count: int,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(
        runner.torch.cuda,
        "set_device",
        lambda index: pytest.fail(f"set_device no debía recibir {index}"),
    )

    with pytest.raises(RuntimeError, match="Índice CUDA fuera de rango"):
        runner.initialize_cuda_and_reset_peak_memory_stats(device_index)


def test_cuda_initialization_requires_available_and_active_device(monkeypatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA dejó de estar disponible"):
        runner.initialize_cuda_and_reset_peak_memory_stats(0)

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner.torch.cuda, "set_device", lambda _index: None)
    monkeypatch.setattr(runner.torch.cuda, "init", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="PyTorch no inicializó CUDA"):
        runner.initialize_cuda_and_reset_peak_memory_stats(0)

    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        runner.torch.cuda,
        "reset_peak_memory_stats",
        lambda _device: pytest.fail("reset no debía ejecutarse"),
    )

    with pytest.raises(RuntimeError, match="Dispositivo CUDA activo inesperado"):
        runner.initialize_cuda_and_reset_peak_memory_stats(0)


def test_smoke_full_train_and_resume_use_safe_cuda_initialization() -> None:
    train_source = inspect.getsource(runner.train_mode)
    resume_source = inspect.getsource(runner.resume_mode)
    initialization = "initialize_cuda_and_reset_peak_memory_stats(DEVICE_INDEX)"
    assert initialization in train_source
    assert train_source.index(initialization) < train_source.index("model.train(**config)")
    assert initialization in resume_source
    assert resume_source.index(initialization) < resume_source.index(
        "model.train(resume=True)"
    )


def test_evaluated_split_mismatch_blocks_summary() -> None:
    runner.require_evaluated_split("test", "test")
    with pytest.raises(
        RuntimeError,
        match="requested_split='test', evaluated_split='val'",
    ):
        runner.require_evaluated_split("test", "val")


def test_validation_observation_uses_effective_ultralytics_loader() -> None:
    observation = {}
    validator = SimpleNamespace(
        args=SimpleNamespace(split="test"),
        data={"test": "/dataset/images/test"},
        dataloader=SimpleNamespace(
            dataset=SimpleNamespace(
                labels=[
                    {"cls": [[0], [0]]},
                    {"cls": [[0]]},
                ],
                __len__=lambda: 2,
            )
        ),
    )
    validator.dataloader.dataset = type(
        "FakeDataset",
        (),
        {
            "labels": validator.dataloader.dataset.labels,
            "__len__": lambda self: 2,
        },
    )()

    runner.capture_validation_observation(validator, observation)

    assert observation == {
        "evaluated_split": "test",
        "resolved_split_path": str(Path("/dataset/images/test").resolve()),
        "image_count": 2,
        "instance_count": 3,
    }


def test_retained_test_contract_counts_fingerprint_pilot_and_best(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "detector_dataset"
    image_dir = dataset / "images" / "test"
    label_dir = dataset / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: maize_leaf\n",
        encoding="utf-8",
    )
    label_line = "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
    other_line = "0 0.5 0.5 0.7 0.5 0.7 0.7\n"
    image_total = 173
    for index in range(image_total):
        stem = f"leaf_{index:03d}"
        (image_dir / f"{stem}.jpg").write_bytes(b"jpeg")
        if index < 9:
            payload = label_line + other_line
        elif index == 9:
            payload = label_line * 2
        else:
            payload = label_line
        (label_dir / f"{stem}.txt").write_text(payload, encoding="utf-8")

    output = tmp_path / "outputs" / "leaf_detection"
    checkpoint = (
        output / "segmenter/yolo26n_seg_baseline/weights/best.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable-best-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr(runner, "DATASET", dataset)
    monkeypatch.setattr(runner, "OUTPUTS", output)
    monkeypatch.setenv("SEGMENTATION_EXPECTED_BEST_SHA256", checkpoint_sha256)
    contract = runner.validate_test_evaluation_inputs(
        checkpoint,
        {
            "split_fingerprints": {"test": runner.EXPECTED_TEST_FINGERPRINT},
            "image_counts": {"test": image_total},
            "mask_counts": {"test": 183},
        },
    )

    assert contract["requested_split"] == "test"
    assert Path(contract["resolved_split_path"]).parts[-2:] == ("images", "test")
    assert contract["image_count"] == 173
    assert contract["instance_count"] == 183
    assert contract["annotation_count"] == 183
    assert contract["loader_instance_count"] == 182
    assert contract["loader_deduplicated_annotations"] == 1
    assert contract["loader_deduplicated_images"] == ["leaf_009"]
    assert contract["test_fingerprint"] == runner.EXPECTED_TEST_FINGERPRINT
    assert contract["pilot_used"] is False
    assert contract["checkpoint"]["path"] == str(checkpoint.resolve())
    assert contract["checkpoint"]["sha256"] == checkpoint_sha256


def test_evaluate_mode_forces_test_and_gates_effective_split_before_summary() -> None:
    source = inspect.getsource(runner.evaluate_mode)
    assert 'config["split"] = REQUESTED_EVALUATION_SPLIT' in source
    assert 'config["name"] = "yolo26n_seg_test"' in source
    assert "save_txt=True" in source
    assert "save_conf=False" in source
    assert "evaluate_downstream(" in source
    assert "require_evaluated_split(" in source
    assert source.index("require_evaluated_split(") < source.index(
        'Path(str(contract["summary_path"]))'
    )
    assert '"pilot_used": False' in source

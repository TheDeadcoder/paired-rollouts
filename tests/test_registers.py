import json
from pathlib import Path

import pytest

REGISTERS = Path(__file__).resolve().parents[1] / "registers"
COMMON_KEYS = {"created_utc", "provider", "gpu", "python", "versions"}
TOY_KEYS = {
    "model", "max_steps", "errors", "adaptations", "reset_kwargs_keys", "reset_calls", "step_times_s",
    "peak_mem_gb", "tools_call_frequency", "tools_failure_frequency", "reward_per_step", "mask_check", "checks",
    "wall_time_s",
}
CHECK_KEYS = {"reset_receives_row_fields", "tool_calls_parsed", "tool_mask_applied", "trained_without_error"}
STEP_TIME_CHECKS = {"step_time_under_30s", "step_time_under_60s"}
LOCAL_KEYS = {"platform", "torch_backend", "torch_backend_version", "fla_importable", "toy"}


def check_stack_register(data: dict, name: str) -> None:
    """Two shapes exist: the day-1 Modal register (flat) and the local register (`toy` block plus backend facts,
    optional `real_eval`). Both carry the same toy checks; the local one uses the 60 s first-step bound."""
    missing = COMMON_KEYS - set(data)
    assert not missing, f"{name} missing keys: {sorted(missing)}"
    if "toy" in data:
        assert LOCAL_KEYS <= set(data), f"{name} missing local keys: {sorted(LOCAL_KEYS - set(data))}"
        toy = data["toy"]
        assert data["torch_backend"] in ("cuda", "hip")
        assert isinstance(data["fla_importable"], bool)
        if "real_eval" in data:
            assert {"episodes", "eval_timings", "episodes_per_second", "errors"} <= set(data["real_eval"])
    else:
        toy = data
    missing = TOY_KEYS - set(toy)
    assert not missing, f"{name} missing toy keys: {sorted(missing)}"
    checks = set(toy["checks"])
    assert CHECK_KEYS <= checks and len(checks & STEP_TIME_CHECKS) == 1 and checks <= CHECK_KEYS | STEP_TIME_CHECKS, \
        f"{name} has unexpected check keys {sorted(checks)}"
    assert all(isinstance(v, bool) for v in toy["checks"].values())
    for p in ["torch", "transformers", "trl", "vllm", "peft"]:
        assert data["versions"][p], f"{name} lacks a version for {p}"


def _stack_registers():
    return sorted(REGISTERS.glob("stack_check_*.json"))


@pytest.mark.parametrize("path", _stack_registers(), ids=lambda p: p.name)
def test_stack_check_register_schema(path):
    check_stack_register(json.loads(path.read_text(encoding="utf-8")), path.name)


def test_local_register_shape_is_accepted_and_incomplete_shapes_are_not():
    toy = {k: [] for k in TOY_KEYS}
    toy["checks"] = dict.fromkeys(CHECK_KEYS | {"step_time_under_60s"}, True)
    data = {
        "created_utc": "x", "provider": "digitalocean", "gpu": None, "python": "3.12", "platform": "linux",
        "torch_backend": "hip", "torch_backend_version": "7.2", "fla_importable": True,
        "versions": {p: "1" for p in ["torch", "transformers", "trl", "vllm", "peft"]}, "toy": toy,
        "real_eval": {"episodes": 544, "eval_timings": [], "episodes_per_second": 0.4, "errors": []},
    }
    check_stack_register(data, "local")
    with pytest.raises(AssertionError, match="missing toy keys"):
        check_stack_register({**data, "toy": {k: v for k, v in toy.items() if k != "mask_check"}}, "local")
    with pytest.raises(AssertionError, match="unexpected check keys"):
        check_stack_register({**data, "toy": {**toy, "checks": {**toy["checks"], "step_time_under_30s": True}}}, "local")


def test_registers_directory_exists():
    assert REGISTERS.is_dir()

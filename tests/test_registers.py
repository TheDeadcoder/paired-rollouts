import json
from pathlib import Path

import pytest

REGISTERS = Path(__file__).resolve().parents[1] / "registers"
STACK_KEYS = {
    "created_utc", "provider", "gpu", "python", "versions", "model", "max_steps", "errors",
    "adaptations", "reset_kwargs_keys", "reset_calls", "step_times_s", "peak_mem_gb",
    "tools_call_frequency", "tools_failure_frequency", "reward_per_step", "mask_check", "checks",
    "wall_time_s",
}
CHECK_KEYS = {
    "reset_receives_row_fields", "tool_calls_parsed", "tool_mask_applied",
    "trained_without_error", "step_time_under_30s",
}


def _stack_registers():
    return sorted(REGISTERS.glob("stack_check_*.json"))


@pytest.mark.parametrize("path", _stack_registers(), ids=lambda p: p.name)
def test_stack_check_register_schema(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = STACK_KEYS - set(data)
    assert not missing, f"{path.name} missing keys: {sorted(missing)}"
    assert CHECK_KEYS == set(data["checks"]), f"{path.name} has unexpected check keys"
    assert all(isinstance(v, bool) for v in data["checks"].values())
    for p in ["torch", "transformers", "trl", "vllm", "peft"]:
        assert data["versions"][p], f"{path.name} lacks a version for {p}"


def test_registers_directory_exists():
    assert REGISTERS.is_dir()

import sys
import time
import types

import pytest

from pairedrl.train.runner import EvalSchedule, RunSpec, load_task_splits


class FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)


class FakeModel:
    def __init__(self):
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class FakeTrainer:
    def __init__(self):
        self.model = FakeModel()
        self.evaluations = []

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval"):
        self.model.eval()
        self.evaluations.append((metric_key_prefix, len(eval_dataset)))
        time.sleep(0.01)
        return {}


class Control:
    should_log = True
    should_save = True


@pytest.fixture
def fake_stack(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object))


def make_schedule(spec, checkpoints):
    splits = load_task_splits("data/tasks")
    trainer = FakeTrainer()
    periodic = {"clean": [{"x": 1}] * 3, "noisy": [{"x": 2}] * 3}
    return EvalSchedule(spec, trainer, splits, periodic, on_checkpoint=lambda: checkpoints.append(1), checkpoint_every=10), trainer


def test_callback_restores_flags_and_mode_and_records_timings(fake_stack):
    spec = RunSpec(run_id="t", model="m", condition="C2", arm="paired", p=0.25, steps=3, eval_every=2, diagnostic_steps=[2])
    checkpoints = []
    schedule, trainer = make_schedule(spec, checkpoints)
    cb = schedule.callback
    state = types.SimpleNamespace(global_step=0)
    for step in (1, 2, 3):
        state.global_step = step - 1
        cb.on_step_begin(None, state, Control())
        trainer.generation_seconds = 1.5
        state.global_step = step
        control = Control()
        cb.on_step_end(None, state, control)
        assert control.should_log and control.should_save and trainer.model.training
        cb.on_save(None, state, control)
    what = [e[0] for e in trainer.evaluations]
    assert what == ["eval_clean", "eval_noisy", "diag", "eval_clean", "eval_noisy"]
    assert [t["step"] for t in schedule.step_timings] == [1, 2, 3]
    assert all(t["generation_s"] == 1.5 and t["seconds"] >= 0 and "save_s" in t for t in schedule.step_timings)
    assert len(checkpoints) == len(trainer.evaluations)


def test_train_only_skips_evaluations_but_keeps_timings(fake_stack):
    spec = RunSpec(run_id="t", model="m", condition="C2", arm="paired", p=0.25, steps=2, eval_every=1, train_only=True)
    schedule, trainer = make_schedule(spec, [])
    cb = schedule.callback
    state = types.SimpleNamespace(global_step=0)
    cb.on_step_begin(None, state, Control())
    state.global_step = 1
    cb.on_step_end(None, state, Control())
    assert trainer.evaluations == [] and len(schedule.step_timings) == 1
    assert schedule.step_timings[0]["generation_s"] is None

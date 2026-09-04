"""TRL environment for the back-office task: reset(**row) builds the episode, tools delegate, get_reward grades."""

import functools
import json
import os
import pathlib
import threading
from typing import ClassVar

from pairedrl.env.backoffice.grader import grade
from pairedrl.env.backoffice.tasks import Task, read_jsonl
from pairedrl.env.backoffice.tools import TOOL_NAMES, ToolAPI, _error
from pairedrl.env.backoffice.world import World
from pairedrl.noise.config import NoiseConfig
from pairedrl.noise.noisy_api import NoisyToolAPI
from pairedrl.noise.oracle import observe
from pairedrl.noise.schedule import NoiseSchedule, resolve_seed

BUDGET_BASE = 10
BUDGET_PER_SUBGOAL = 3
BUDGET_CAP = 30

_TASK_CACHE: dict[str, dict[str, Task]] = {}
_CACHE_LOCK = threading.Lock()


def budget_for(task: Task) -> int:
    return min(BUDGET_CAP, BUDGET_BASE + BUDGET_PER_SUBGOAL * task.n_subgoals)


def load_tasks(paths) -> dict[str, Task]:
    key = "|".join(str(p) for p in paths)
    with _CACHE_LOCK:
        if key not in _TASK_CACHE:
            tasks: dict[str, Task] = {}
            for p in paths:
                for task in read_jsonl(p):
                    tasks[task.task_id] = task
            _TASK_CACHE[key] = tasks
        return _TASK_CACHE[key]


def parse_noise(value) -> NoiseConfig:
    if isinstance(value, NoiseConfig):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return NoiseConfig.from_dict(value)


class BackOfficeEnv:
    """One instance per concurrent rollout. TRL calls reset(**row), then the public tools, then get_reward()."""

    _slot_counter: ClassVar[int] = 0
    _slot_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, task_paths, log_path: str | os.PathLike | None = None):
        with BackOfficeEnv._slot_lock:
            self.slot = BackOfficeEnv._slot_counter
            BackOfficeEnv._slot_counter += 1
        self.task_paths = [str(p) for p in task_paths]
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.episode_index = -1
        self.task: Task | None = None
        self.api: NoisyToolAPI | None = None
        self.schedule: NoiseSchedule | None = None
        self.config: NoiseConfig | None = None
        self.row: dict = {}
        self.calls = 0
        self.budget = 0
        self.budget_exceeded = False
        self.initial_snapshot: dict | None = None
        self.last_episode: dict | None = None

    def reset(self, **kwargs):
        tasks = load_tasks(self.task_paths)
        self.episode_index += 1
        self.row = {k: v for k, v in kwargs.items() if k != "prompt"}
        self.task = tasks[kwargs["task_id"]]
        self.config = parse_noise(kwargs["noise"])
        seed = resolve_seed(self.config, int(kwargs["schedule_seed"]), self.slot, self.episode_index)
        self.schedule = None if self.config.is_clean else NoiseSchedule(seed, self.config)
        world = World.generate(self.task.world_seed)
        self.api = NoisyToolAPI(world, self.schedule)
        self.initial_snapshot = world.snapshot()
        self.calls = 0
        self.budget = int(kwargs.get("budget") or budget_for(self.task))
        self.budget_exceeded = False
        self.last_episode = None

    def _call(self, name: str, *args, **kwargs) -> str:
        if self.api is None:
            return _error("NO_EPISODE", "reset must be called before using tools")
        if args:
            return _error("INVALID_ARGUMENT", "tools take keyword arguments only")
        if name != "finish" and not self.api.finished:
            self.calls += 1
            if self.calls > self.budget:
                self.budget_exceeded = True
                self.api.finished = True
                return _error(
                    "BUDGET_EXCEEDED",
                    f"the budget of {self.budget} tool calls is exhausted; no further calls are accepted",
                )
        return getattr(self.api, name)(**kwargs)

    def get_reward(self) -> float:
        if self.api is None or self.task is None:
            return 0.0
        result = grade(self.task, self.api.world.snapshot(), self.initial_snapshot)
        observed, flipped = observe(result.success, self.schedule)
        record = {
            "task_id": self.task.task_id,
            "condition": self.row.get("condition"),
            "mode": self.config.mode if self.config else "clean",
            "schedule_seed": int(self.row.get("schedule_seed", 0)),
            "resolved_seed": self.schedule.seed if self.schedule else None,
            "slot": self.slot,
            "episode_index": self.episode_index,
            "true_success": bool(result.success),
            "observed_reward": float(observed),
            "flipped": bool(flipped),
            "calls": self.calls,
            "budget": self.budget,
            "budget_exceeded": self.budget_exceeded,
            "finished": bool(self.api.finished and not self.budget_exceeded),
            "exposed": self.api.exposed,
            "faults": self.api.exposure_counts(),
            "changed_from_initial": result.changed_from_initial,
            "diffs": result.diffs[:5],
            "tool_sequence": [
                {"tool": c["tool"], "ok": c["ok"], "code": c["code"]} for c in self.api.call_log
            ],
        }
        self.last_episode = record
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        return float(observed)


def _delegate(name: str):
    source = getattr(ToolAPI, name)

    @functools.wraps(source)
    def method(self, *args, **kwargs):
        return self._call(name, *args, **kwargs)

    return method


for _name in sorted(TOOL_NAMES):
    setattr(BackOfficeEnv, _name, _delegate(_name))


def make_env_factory(task_paths, log_path=None):
    def factory() -> BackOfficeEnv:
        return BackOfficeEnv(task_paths, log_path=log_path)

    return factory

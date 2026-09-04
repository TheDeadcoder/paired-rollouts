"""ToolAPI wrapped with a fault schedule. Same 13 public tools; faults injected in _run."""

import json
from collections import Counter

from pairedrl.env.backoffice.tools import CONTROL_TOOLS, READ_TOOLS, ToolAPI, _dumps, _error
from pairedrl.env.backoffice.world import World
from pairedrl.noise.schedule import LIST_TOOLS, NoiseSchedule

TRUNCATED_PAGE = 2


class NoisyToolAPI(ToolAPI):
    def __init__(self, world: World, schedule: NoiseSchedule | None = None):
        super().__init__(world)
        self.schedule = schedule
        self.call_index: Counter = Counter()
        self.fault_log: list[dict] = []
        self.rate_limited_until = 0
        self._stale_snapshot: dict | None = None

    @property
    def exposed(self) -> bool:
        return bool(self.fault_log)

    def exposure_counts(self) -> dict[str, int]:
        return dict(Counter(f["kind"] for f in self.fault_log))

    def _view(self) -> dict:
        if self._stale_snapshot is not None:
            return self._stale_snapshot
        return self.world.snapshot()

    def _log_fault(self, name: str, k: int, kind: str, kwargs: dict, code: str) -> None:
        self.fault_log.append({"call_no": len(self.call_log) + 1, "tool": name, "index": k, "kind": kind})
        self.call_log.append({"tool": name, "args": kwargs, "ok": False, "code": code, "fault": kind})

    def _run(self, name: str, fn, **kwargs) -> str:
        if self.finished or self.schedule is None or name in CONTROL_TOOLS:
            return super()._run(name, fn, **kwargs)
        k = self.call_index[name]
        self.call_index[name] += 1
        clock = self.world.clock_seconds
        if clock < self.rate_limited_until:
            remaining = self.rate_limited_until - clock
            self._log_fault(name, k, "rate_limit_ongoing", kwargs, "RATE_LIMITED")
            return _error(
                "RATE_LIMITED",
                f"too many requests; wait {remaining} seconds before calling again",
                retry_after_seconds=remaining,
            )
        fate = self.schedule.fate(name, k)
        if fate.kind is None:
            return super()._run(name, fn, **kwargs)
        if fate.kind == "transient":
            self._log_fault(name, k, "transient", kwargs, "SERVICE_UNAVAILABLE")
            return _error("SERVICE_UNAVAILABLE", "temporary failure; the request was not applied, retry it")
        if fate.kind == "rate_limit":
            self.rate_limited_until = clock + fate.retry_after
            self._log_fault(name, k, "rate_limit", kwargs, "RATE_LIMITED")
            return _error(
                "RATE_LIMITED",
                f"too many requests; wait {fate.retry_after} seconds before calling again",
                retry_after_seconds=fate.retry_after,
            )
        if fate.kind == "stale":
            if self.world.version < 1:
                return super()._run(name, fn, **kwargs)
            self._stale_snapshot = self.world.as_of(self.world.version - 1)
            try:
                out = super()._run(name, fn, **kwargs)
            finally:
                self._stale_snapshot = None
            self.fault_log.append(
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "stale"}
            )
            return out
        if fate.kind == "truncate":
            out = super()._run(name, fn, **kwargs)
            payload = json.loads(out)
            if "error" in payload or name not in LIST_TOOLS:
                return out
            shown = payload["results"][:TRUNCATED_PAGE]
            payload["results"] = shown
            payload["truncated"] = True
            nxt = payload["offset"] + len(shown)
            payload["next_offset"] = nxt if nxt < payload["total"] else None
            self.fault_log.append(
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "truncate"}
            )
            return _dumps(payload)
        if fate.kind == "timeout_after_commit":
            out = super()._run(name, fn, **kwargs)
            if "error" in json.loads(out):
                return out
            self.fault_log.append(
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "timeout_after_commit"}
            )
            return _error(
                "TIMEOUT",
                "the request timed out; it may or may not have been applied, check before retrying",
            )
        if fate.kind == "field_dropout":
            out = super()._run(name, fn, **kwargs)
            payload = json.loads(out)
            if "error" in payload or name not in READ_TOOLS:
                return out
            keys = sorted(payload)
            if not keys:
                return out
            del payload[keys[fate.dropped_field_index % len(keys)]]
            self.fault_log.append(
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "field_dropout"}
            )
            return _dumps(payload)
        raise ValueError(f"unknown fault kind {fate.kind!r}")

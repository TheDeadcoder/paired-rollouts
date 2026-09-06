"""ToolAPI wrapped with a fault schedule. Same 13 public tools; faults injected in _run."""

import json
from collections import Counter

from pairedrl.env.backoffice.tools import CONTROL_TOOLS, READ_TOOLS, ToolAPI, _dumps, _error
from pairedrl.env.backoffice.world import World
from pairedrl.noise.schedule import LIST_TOOLS, NoiseSchedule, fault_key

TRUNCATED_PAGE = 2
OUTAGE_MESSAGE = "temporary failure; the request was not applied, retry it"


class NoisyToolAPI(ToolAPI):
    """Faults are keyed by (tool, fault_key(args), repeat): the n-th time a tool touches the same resource."""

    def __init__(self, world: World, schedule: NoiseSchedule | None = None):
        super().__init__(world)
        self.schedule = schedule
        self.call_index: Counter = Counter()
        self.fault_log: list[dict] = []
        self.rate_limited_until = 0
        self.outages: set[tuple[str, str]] = set()
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

    def _log_fault(self, name: str, k: int, kind: str, kwargs: dict, code: str, event: str) -> None:
        self.fault_log.append({"call_no": len(self.call_log) + 1, "tool": name, "index": k, "kind": kind, "event": event})
        self.call_log.append({"tool": name, "args": kwargs, "ok": False, "code": code, "fault": kind, "event": event})

    def _run(self, name: str, fn, **kwargs) -> str:
        if self.finished or self.schedule is None or name in CONTROL_TOOLS:
            return super()._run(name, fn, **kwargs)
        request = fault_key(name, kwargs)
        k = self.call_index[(name, request)]
        self.call_index[(name, request)] += 1
        before = len(self.call_log)
        out = self._run_event(name, fn, request, k, **kwargs)
        if len(self.call_log) > before:
            self.call_log[-1].setdefault("event", request)
            self.call_log[-1].setdefault("repeat", k)
        return out

    def _run_event(self, name: str, fn, request: str, k: int, **kwargs) -> str:
        clock = self.world.clock_seconds
        if clock < self.rate_limited_until:
            remaining = self.rate_limited_until - clock
            self._log_fault(name, k, "rate_limit_ongoing", kwargs, "RATE_LIMITED", request)
            return _error(
                "RATE_LIMITED",
                f"too many requests; wait {remaining} seconds before calling again",
                retry_after_seconds=remaining,
            )
        if (name, request) in self.outages:
            self._log_fault(name, k, "outage_ongoing", kwargs, "SERVICE_UNAVAILABLE", request)
            return _error("SERVICE_UNAVAILABLE", OUTAGE_MESSAGE)
        fate = self.schedule.fate(name, request, k)
        if fate.kind is None:
            return super()._run(name, fn, **kwargs)
        if fate.kind == "transient":
            self._log_fault(name, k, "transient", kwargs, "SERVICE_UNAVAILABLE", request)
            return _error("SERVICE_UNAVAILABLE", OUTAGE_MESSAGE)
        if fate.kind == "outage":
            self.outages.add((name, request))
            self._log_fault(name, k, "outage", kwargs, "SERVICE_UNAVAILABLE", request)
            return _error("SERVICE_UNAVAILABLE", OUTAGE_MESSAGE)
        if fate.kind == "rate_limit":
            self.rate_limited_until = clock + fate.retry_after
            self._log_fault(name, k, "rate_limit", kwargs, "RATE_LIMITED", request)
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
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "stale", "event": request}
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
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "truncate", "event": request}
            )
            return _dumps(payload)
        if fate.kind == "timeout_after_commit":
            out = super()._run(name, fn, **kwargs)
            if "error" in json.loads(out):
                return out
            self.fault_log.append(
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "timeout_after_commit", "event": request}
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
                {"call_no": len(self.call_log), "tool": name, "index": k, "kind": "field_dropout", "event": request}
            )
            return _dumps(payload)
        raise ValueError(f"unknown fault kind {fate.kind!r}")

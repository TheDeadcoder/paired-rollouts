"""Stall decisions for queued jobs. Progress is the growth of files a job is known to write steadily (for a training
job its episodes.jsonl and its stdout log), never run_manifest.json or trainer_log_history.json, which are rewritten
only after evaluation phases and every few training steps. The clocks are injected so the decisions are testable."""

import glob
import os
from dataclasses import dataclass

STARTUP_GRACE_S = 45 * 60
STALL_S = 45 * 60


@dataclass(frozen=True)
class Thresholds:
    startup_grace_s: float = STARTUP_GRACE_S
    stall_s: float = STALL_S


DEFAULT_THRESHOLDS = Thresholds()


def progress_sizes(patterns) -> tuple[int, ...]:
    """Total bytes of the files matching each pattern (a path or a glob), in pattern order; a missing file is 0."""
    out = []
    for pattern in patterns:
        out.append(sum(os.path.getsize(p) for p in glob.glob(str(pattern)) if os.path.isfile(p)))
    return tuple(out)


def stalled(now: float, started: float, last_progress: float | None, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """A job is never stalled inside its start-up grace. After it, the no-progress clock runs from the later of the
    last progress and the end of the grace, and the job is stalled once that clock reaches stall_s: a job that never
    progresses is first stalled at started + startup_grace_s + stall_s."""
    if now - started < thresholds.startup_grace_s:
        return False
    anchor = max(last_progress if last_progress is not None else started, started + thresholds.startup_grace_s)
    return now - anchor >= thresholds.stall_s

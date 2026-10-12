"""The droplet-side job queue (docs/DIGITALOCEAN.md, "The droplet queue"). Entries run one after another; an entry is
one job or a concurrent pair sharing the GPU on slots 0 and 1, each slot with its own ports and its own temp and
compile-cache directories. Every job is watched for stalls (pairedrl.ops.watchdog) and relaunched at most once in
total, after a stall or after an exit without COMPLETE (a training job then resumes from its newest complete
checkpoint under the same run id); after that it is FAILED and the queue moves on. One job's stall or failure never
touches the other job of its pair. A training pair also gets the step-0 identity check: when the two arms' inputs
differ, both are stopped; when the inputs agree but the step-0 scores differ (GPU nondeterminism, which a relaunch
cannot remove), the difference is recorded and both continue.

A queue directory (one per droplet, <runs_dir>/queue by default) holds:
    entries.jsonl   the entries in order, one JSON object per line, appended while the queue runs
    status.json     the runner's record: entry and job states, attempts, events, step-0 checks, when it went empty
    logs/           one stdout and stderr log per job attempt
    queue.lock      held while an entry is appended, and while the runner checks for more work before it exits
    runner.lock     held by the one live runner
    runner.log      the runner's own output

Entry format (one line of entries.jsonl):
    {"id": "e0001", "kind": "run" | "pair" | "command", "label": "...", "commit": "<sha>",
     "checkout": "<checkout of that commit>", "added_utc": "...", "jobs": [<job>, ...],
     "startup_grace_s": 2700, "stall_s": 2700}                         (the last two are optional)
    training job: {"name": "<run_id>", "spec": "configs/<run_id>.json"}
    command job:  {"name": "...", "argv": ["python", "scripts/...", ...], "progress": ["<path or glob>", ...],
                   "run_dir": "<directory that receives its queue_events.jsonl>"}       (run_dir optional)
    run = one training job; command = one command job; pair = two jobs of either kind, on slots 0 and 1.
Every job runs from the entry's checkout with PYTHONPATH=<checkout>/src, so entries appended at a later commit run
their own code while earlier entries keep theirs.
"""

import contextlib
import datetime as dt
import fcntl
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

from pairedrl.ops.watchdog import STALL_S, STARTUP_GRACE_S, Thresholds, progress_sizes, stalled

KINDS = ("run", "pair", "command")
TERMINAL = ("COMPLETE", "FAILED")
MAX_ATTEMPTS = 2
POLL_S = 30
SCORE_CHECK_EVERY_S = 300
PAIR_START_TIMEOUT_S = 30 * 60
KILL_GRACE_S = 60
RELEASE_WAIT_S = 30
CHECKPOINT_EVERY = 2
MASTER_PORT_BASE, VLLM_PORT_BASE, PORT_STRIDE = 29500, 31000, 100
SLOT_DIRS = ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT")
STEP0_SPEC_EXCLUDED = ("arm", "run_id", "notes")
STEP0_INPUTS = ("git_commit", "deployed_commit", "spec", "model_commit_hash", "initial_adapter_sha256",
                "eval_sets_sha256", "versions")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _iso(epoch: float | None) -> str | None:
    return None if epoch is None else dt.datetime.fromtimestamp(epoch, dt.UTC).isoformat(timespec="seconds")


def slot_environment(slot: int, cache_root: str | os.PathLike | None = None) -> dict[str, str]:
    """The environment that keeps two jobs on one GPU apart: torch.distributed's rendezvous port (TRL 1.12.0 keeps a
    set MASTER_PORT), vLLM's base port, and separate temp, Triton, Inductor and vLLM compile-cache directories."""
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot!r}")
    root = pathlib.Path(cache_root or os.path.expanduser("~/.cache/pairedrl")) / f"slot{slot}"
    return {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(MASTER_PORT_BASE + PORT_STRIDE * slot),
        "VLLM_PORT": str(VLLM_PORT_BASE + PORT_STRIDE * slot),
        "TMPDIR": str(root / "tmp"),
        "TRITON_CACHE_DIR": str(root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
        "PAIREDRL_SLOT": str(slot),
    }


# ---- locks and files ----------------------------------------------------------------------------------------------

@contextlib.contextmanager
def locked(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def try_lock(path) -> int | None:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def unlock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def runner_alive(queue_dir) -> bool:
    fd = try_lock(pathlib.Path(queue_dir) / "runner.lock")
    if fd is None:
        return True
    unlock(fd)
    return False


def read_json(path) -> dict | None:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, obj) -> None:
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_entries(queue_dir) -> list[dict]:
    path = pathlib.Path(queue_dir) / "entries.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_status(queue_dir) -> dict:
    return read_json(pathlib.Path(queue_dir) / "status.json") or {"entries": {}}


def validate_entry(entry: dict) -> None:
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError(f"entry kind must be one of {KINDS}, got {kind!r}")
    jobs = entry.get("jobs") or []
    expected = {"run": 1, "command": 1, "pair": 2}[kind]
    if len(jobs) != expected:
        raise ValueError(f"a {kind} entry has {expected} job(s), got {len(jobs)}")
    for job in jobs:
        if not job.get("name"):
            raise ValueError("every job needs a name")
        has_spec, has_argv = bool(job.get("spec")), bool(job.get("argv"))
        if has_spec == has_argv:
            raise ValueError(f"{job['name']}: a job has exactly one of spec (training) or argv (command)")
        if (kind == "run" and not has_spec) or (kind == "command" and not has_argv):
            raise ValueError(f"{job['name']}: a {kind} entry needs a {'spec' if kind == 'run' else 'command'} job")
        if has_argv and not job.get("progress"):
            raise ValueError(f"{job['name']}: a command job names the files whose growth is its progress")
    if len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("the jobs of an entry need different names")
    for key in ("commit", "checkout"):
        if not entry.get(key):
            raise ValueError(f"an entry needs {key}")
    for key in ("startup_grace_s", "stall_s"):
        if key in entry and float(entry[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def append_entry(queue_dir, entry: dict) -> tuple[str, bool]:
    """Validate and append one entry under the queue lock, give it the next id, and report whether a runner is alive
    to pick it up (if not, the caller starts one; the runner's own exit check holds the same lock, so an entry
    appended while the last runner was finishing is never stranded)."""
    queue_dir = pathlib.Path(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    validate_entry(entry)
    with locked(queue_dir / "queue.lock"):
        entry = dict(entry, id=f"e{len(read_entries(queue_dir)) + 1:04d}")
        entry.setdefault("added_utc", utc_now())
        with open(queue_dir / "entries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        alive = runner_alive(queue_dir)
    return entry["id"], alive


# ---- the step-0 identity check ------------------------------------------------------------------------------------

def step0_inputs(manifest: dict) -> dict:
    """What must agree between the two arms of a pair before step 1: commit, the spec without its arm and labels,
    the model snapshot, the initial adapter, the evaluation sets and the package versions."""
    out = {key: manifest.get(key) for key in STEP0_INPUTS}
    out["spec"] = {k: v for k, v in (manifest.get("spec") or {}).items() if k not in STEP0_SPEC_EXCLUDED}
    return out


def input_differences(manifest_a: dict, manifest_b: dict) -> list[str]:
    a, b = step0_inputs(manifest_a), step0_inputs(manifest_b)
    diffs = [key for key in STEP0_INPUTS if key != "spec" and a[key] != b[key]]
    diffs += [f"spec.{k}" for k in sorted(set(a["spec"]) | set(b["spec"])) if a["spec"].get(k) != b["spec"].get(k)]
    return diffs


def step0_scores(run_dir, spec: dict) -> dict | None:
    """(set, task, schedule) -> true success of every step-0 periodic evaluation along the run's attempt lineage, or
    None while any periodic set of the profile is incomplete."""
    from pairedrl.analysis.curves import lineage_records, load_attempts
    from pairedrl.train.runner import RunSpec, expected_eval_sizes

    run_dir = pathlib.Path(run_dir)
    if not (run_dir / "run_manifest.json").exists():
        return None
    records, _ = lineage_records(load_attempts(run_dir), "episodes.jsonl")
    out = {}
    for name, size in expected_eval_sizes(RunSpec.from_dict(spec))["periodic"].items():
        found = [r for r in records if r.get("phase") == f"eval_{name}:step0"]
        if len(found) < size:
            return None
        for r in found:
            out[f"{name}|{r['task_id']}|{int(r['schedule_seed'])}"] = bool(r["true_success"])
    return out


def score_differences(scores_a: dict, scores_b: dict) -> dict:
    keys = sorted(set(scores_a) | set(scores_b))
    differing = [k for k in keys if scores_a.get(k) != scores_b.get(k)]
    return {"identities": len(keys), "differing": len(differing), "match": not differing, "examples": differing[:5]}


def has_step0_evaluation(spec: dict) -> bool:
    from pairedrl.train.runner import RunSpec, periodic_steps

    return 0 in periodic_steps(RunSpec.from_dict(spec))


# ---- processes --------------------------------------------------------------------------------------------------

def default_spawn(argv, cwd, env, log_path):
    """Start a job in its own session (so its process group can be killed alone), output appended to its log."""
    with open(log_path, "ab") as log:
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def default_kill(handle, grace_s: float = KILL_GRACE_S, sleep=time.sleep) -> None:
    """SIGTERM to the job's process group, SIGKILL after the grace period, then make sure no member is left."""
    try:
        os.killpg(handle.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    waited = 0.0
    while handle.poll() is None and waited < grace_s:
        sleep(1)
        waited += 1
    with contextlib.suppress(ProcessLookupError):
        os.killpg(handle.pid, signal.SIGKILL)
    handle.wait()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Job:
    """One job of an entry: its slot, attempts and progress."""

    def __init__(self, entry: dict, index: int, job: dict, runs_dir: pathlib.Path):
        self.entry_id = entry["id"]
        self.name = job["name"]
        self.slot = index
        self.spec_path = job.get("spec")
        self.argv = job.get("argv")
        self.training = self.spec_path is not None
        if job.get("run_dir"):
            self.run_dir = pathlib.Path(job["run_dir"])
        else:
            self.run_dir = runs_dir / self.name if self.training else None
        self.progress = list(job.get("progress") or [])
        self.state = "PENDING"
        self.attempts = 0
        self.handle = None
        self.started = None
        self.start_wall = None
        self.first_started = None
        self.start_offset_s = 0.0
        self.last_progress = None
        self.sizes = None
        self.log_path = None

    def progress_patterns(self) -> list[str]:
        patterns = [str(self.run_dir / "episodes.jsonl")] if self.training else list(self.progress)
        if self.log_path is not None:
            patterns.append(str(self.log_path))
        return patterns

    def current_manifest(self) -> dict | None:
        """The run manifest written by this job's current attempt (an older attempt's is ignored)."""
        if not self.training or self.start_wall is None:
            return None
        manifest = read_json(self.run_dir / "run_manifest.json")
        if not manifest or str(manifest.get("started_utc", "")) < self.start_wall:
            return None
        return manifest


class Step0Check:
    """The step-0 identity check of a training pair: inputs as soon as both manifests carry them, then the step-0
    periodic scores once both arms have evaluated them."""

    def __init__(self, jobs: list[Job], specs: dict, record: dict):
        self.jobs, self.specs = jobs, specs
        self.inputs_done = "inputs" in record
        self.scores_done = "scores" in record or not all(has_step0_evaluation(specs[j.name]) for j in jobs)
        self.next_score_check = 0.0

    def update(self, runner, est: dict, now: float) -> None:
        record = est.setdefault("step0", {})
        if not self.inputs_done:
            manifests = [j.current_manifest() for j in self.jobs]
            if not all(m and "initial_adapter_sha256" in m for m in manifests):
                return
            diffs = input_differences(*manifests)
            self.inputs_done = True
            record["inputs"] = {"checked_utc": utc_now(), "match": not diffs, "differing": diffs}
            if diffs:
                detail = f"inputs differ ({', '.join(diffs)}): both arms stopped; fix the difference and relaunch once"
                for job in self.jobs:
                    runner.stop(est, job, "step0_inputs_differ", detail)
                self.scores_done = True
                return
            for job in self.jobs:
                runner.event(est, job, "step0_inputs_match", "commit, spec, model, initial adapter, evaluation sets and packages agree")
        if self.scores_done or now < self.next_score_check:
            return
        if any(j.state == "FAILED" for j in self.jobs):
            record["scores"] = {"checked_utc": utc_now(), "compared": False, "reason": "an arm failed before step 0 was compared"}
            self.scores_done = True
            return
        self.next_score_check = now + SCORE_CHECK_EVERY_S
        scores = [step0_scores(j.run_dir, self.specs[j.name]) for j in self.jobs]
        if any(s is None for s in scores):
            return
        diff = score_differences(*scores)
        record["scores"] = {"checked_utc": utc_now(), "compared": True, **diff}
        self.scores_done = True
        if diff["match"]:
            event, detail = "step0_scores_match", f"all {diff['identities']} step-0 evaluations agree"
        else:
            event = "step0_scores_differ"
            detail = (f"{diff['differing']} of {diff['identities']} step-0 evaluations differ with identical inputs "
                      f"(GPU nondeterminism); recorded, both arms continue")
        for job in self.jobs:
            runner.event(est, job, event, detail)


class QueueRunner:
    """Runs a queue directory's entries in order until none is left, then records when the queue went empty and
    exits. Process control, clocks and sleeping are injected so the semantics are testable without a GPU."""

    def __init__(self, queue_dir, runs_dir, provider: str, usd_per_hour: float, python: str | None = None,
                 poll_s: float = POLL_S, checkpoint_every: int = CHECKPOINT_EVERY, cache_root=None, spawn=default_spawn,
                 kill=None, clock=time.time, sleep=time.sleep, release_wait_s: float = RELEASE_WAIT_S,
                 pair_start_timeout_s: float = PAIR_START_TIMEOUT_S, env: dict | None = None):
        self.queue_dir, self.runs_dir = pathlib.Path(queue_dir), pathlib.Path(runs_dir)
        self.provider, self.usd_per_hour = provider, usd_per_hour
        self.python = python or sys.executable
        self.poll_s, self.checkpoint_every, self.cache_root = poll_s, checkpoint_every, cache_root
        self.spawn, self.clock, self.sleep = spawn, clock, sleep
        self.kill = kill or (lambda handle: default_kill(handle, sleep=sleep))
        self.release_wait_s, self.pair_start_timeout_s = release_wait_s, pair_start_timeout_s
        self.env = dict(os.environ if env is None else env)
        self.status: dict = {"entries": {}}

    # -- bookkeeping --

    def save(self) -> None:
        self.status["updated_utc"] = utc_now()
        write_json(self.queue_dir / "status.json", self.status)

    def event(self, est: dict, job: Job | None, kind: str, detail: str) -> None:
        record = {"utc": utc_now(), "event": kind, "job": job.name if job else None, "detail": detail}
        est.setdefault("events", []).append(record)
        if job is not None and job.run_dir is not None:
            job.run_dir.mkdir(parents=True, exist_ok=True)
            with open(job.run_dir / "queue_events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(record, entry=job.entry_id), sort_keys=True) + "\n")

    def job_record(self, est: dict, job: Job) -> dict:
        record = est.setdefault("jobs", {}).setdefault(job.name, {"attempts": []})
        record.update({"state": job.state, "slot": job.slot, "start_offset_s": job.start_offset_s,
                       "last_progress_utc": _iso(job.last_progress),
                       "progress_bytes": sum(job.sizes) if job.sizes else 0})
        return record

    def end_attempt(self, est: dict, job: Job, exit_code, outcome: str) -> None:
        attempts = self.job_record(est, job)["attempts"]
        if attempts and not attempts[-1].get("outcome"):
            attempts[-1].update({"end_utc": utc_now(), "exit_code": exit_code, "outcome": outcome})

    # -- the loop --

    def run(self) -> int:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        fd = try_lock(self.queue_dir / "runner.lock")
        if fd is None:
            print(f"another runner holds {self.queue_dir}")
            return 0
        try:
            self.status = read_status(self.queue_dir)
            self.status.setdefault("entries", {})
            self.status.pop("empty_utc", None)
            self.status["runner"] = {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now(),
                                     "heartbeat_utc": utc_now(), "state": "RUNNING"}
            orphans = self.live_orphans()
            if orphans:
                self.status["runner"].update(state="REFUSED", error=f"jobs of an earlier runner are still alive: {orphans}")
                self.save()
                return 1
            self.save()
            while True:
                entry = self.next_entry()
                if entry is None:
                    with locked(self.queue_dir / "queue.lock"):
                        entry = self.next_entry()
                        if entry is None:
                            self.status["empty_utc"] = utc_now()
                            self.status["runner"].update(state="EXITED", heartbeat_utc=utc_now())
                            self.save()
                            unlock(fd)
                            fd = None
                            return 0
                self.run_entry(entry)
        finally:
            if fd is not None:
                unlock(fd)

    def next_entry(self) -> dict | None:
        for entry in read_entries(self.queue_dir):
            if self.status["entries"].get(entry["id"], {}).get("state") not in TERMINAL:
                return entry
        return None

    def live_orphans(self) -> list[str]:
        alive = []
        for entry_id, est in self.status["entries"].items():
            if est.get("state") != "RUNNING":
                continue
            for name, record in est.get("jobs", {}).items():
                attempts = record.get("attempts") or []
                if record.get("state") == "RUNNING" and attempts and attempts[-1].get("pid") and pid_alive(attempts[-1]["pid"]):
                    alive.append(f"{entry_id}/{name} pid {attempts[-1]['pid']}")
        return alive

    def run_entry(self, entry: dict) -> None:
        est = self.status["entries"].setdefault(entry["id"], {})
        est.update({"kind": entry["kind"], "label": entry.get("label"), "commit": entry.get("commit"), "state": "RUNNING"})
        est.setdefault("start_utc", utc_now())
        thresholds = Thresholds(float(entry.get("startup_grace_s", STARTUP_GRACE_S)), float(entry.get("stall_s", STALL_S)))
        jobs = [Job(entry, i, job, self.runs_dir) for i, job in enumerate(entry["jobs"])]
        specs, problem = self.load_specs(entry, jobs)
        if problem is not None:
            for job in jobs:
                job.state = "FAILED"
                self.job_record(est, job)
            self.event(est, None, "entry_refused", problem)
            est.update(state="FAILED", end_utc=utc_now())
            self.save()
            return
        self.restore(est, jobs)
        check = (Step0Check(jobs, specs, est.get("step0", {}))
                 if entry["kind"] == "pair" and all(j.training for j in jobs) else None)
        for job in jobs[:1] if entry["kind"] == "pair" else jobs:
            if job.state == "PENDING":
                self.start(job, entry, est, jobs)
        for job in jobs:
            self.job_record(est, job)
        self.save()
        while not all(j.state in TERMINAL for j in jobs):
            self.sleep(self.poll_s)
            now = self.clock()
            for job in jobs:
                self.supervise(job, entry, est, jobs, thresholds, now)
            if entry["kind"] == "pair" and jobs[1].state == "PENDING" and self.pair_ready(jobs[0], now):
                offset = now - jobs[0].first_started if jobs[0].first_started is not None else 0.0
                self.start(jobs[1], entry, est, jobs, offset=offset)
            if check is not None:
                check.update(self, est, now)
            self.status["runner"]["heartbeat_utc"] = utc_now()
            for job in jobs:
                self.job_record(est, job)
            self.save()
        est.update(state="COMPLETE" if all(j.state == "COMPLETE" for j in jobs) else "FAILED", end_utc=utc_now())
        self.save()

    def load_specs(self, entry: dict, jobs: list[Job]) -> tuple[dict, str | None]:
        checkout = pathlib.Path(entry["checkout"])
        if not checkout.is_dir():
            return {}, f"checkout {checkout} does not exist"
        specs = {}
        for job in jobs:
            if not job.training:
                continue
            spec = read_json(checkout / job.spec_path)
            if spec is None:
                return {}, f"{job.name}: cannot read {job.spec_path} in {checkout}"
            if spec.get("run_id") != job.name:
                return {}, f"{job.name}: {job.spec_path} has run_id {spec.get('run_id')!r}"
            specs[job.name] = spec
        return specs, None

    def restore(self, est: dict, jobs: list[Job]) -> None:
        """Pick up an entry an earlier runner left RUNNING (its processes are gone; live ones refuse the start)."""
        for job in jobs:
            record = est.get("jobs", {}).get(job.name)
            if not record:
                continue
            job.attempts = len(record.get("attempts") or [])
            job.start_offset_s = record.get("start_offset_s", 0.0)
            if record.get("state") in TERMINAL:
                job.state = record["state"]
            elif record.get("state") == "RUNNING":
                self.end_attempt(est, job, None, "INTERRUPTED")
                if job.attempts >= MAX_ATTEMPTS:
                    job.state = "FAILED"
                    self.event(est, job, "failed", "its runner stopped while it ran, and no relaunch is left")
                else:
                    self.event(est, job, "relaunch", "its runner stopped while it ran")

    def argv(self, job: Job, entry: dict, other: str | None) -> list[str]:
        if not job.training:
            return list(job.argv)
        argv = [self.python, "scripts/run_local.py", "--spec", job.spec_path, "--runs-dir", str(self.runs_dir),
                "--provider", self.provider, "--usd-per-hour", str(self.usd_per_hour), "--commit", entry["commit"],
                "--checkpoint-every", str(self.checkpoint_every), "--slot", str(job.slot),
                "--start-offset-s", f"{job.start_offset_s:.1f}", "--queue-entry", entry["id"]]
        return argv + (["--shares-gpu-with", other] if other else [])

    def job_env(self, job: Job, entry: dict) -> dict:
        env = dict(self.env)
        slot_env = slot_environment(job.slot, self.cache_root)
        for key in SLOT_DIRS:
            pathlib.Path(slot_env[key]).mkdir(parents=True, exist_ok=True)
        env.update(slot_env)
        src = str(pathlib.Path(entry["checkout"]) / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    def start(self, job: Job, entry: dict, est: dict, jobs: list[Job], offset: float | None = None) -> None:
        job.attempts += 1
        if job.first_started is None and offset is not None:
            job.start_offset_s = round(offset, 1)
        logs = self.queue_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        job.log_path = logs / f"{entry['id']}-{job.name}-a{job.attempts}.log"
        other = next((j.name for j in jobs if j is not job), None) if entry["kind"] == "pair" else None
        argv, env = self.argv(job, entry, other), self.job_env(job, entry)
        job.start_wall = utc_now()
        job.handle = self.spawn(argv, str(entry["checkout"]), env, str(job.log_path))
        job.state, job.started, job.last_progress = "RUNNING", self.clock(), None
        if job.first_started is None:
            job.first_started = job.started
        job.sizes = progress_sizes(job.progress_patterns())
        record = self.job_record(est, job)
        record["attempts"].append({"n": job.attempts, "pid": getattr(job.handle, "pid", None), "start_utc": job.start_wall,
                                   "log": str(job.log_path)})
        where = f"slot {job.slot}" + (f", {job.start_offset_s:.0f} s after {other}" if other and job.slot == 1 else "")
        self.event(est, job, "start", f"attempt {job.attempts} on {where}")
        self.save()

    def pair_ready(self, first: Job, now: float) -> bool:
        """Start the second job of a pair once the first has finished vLLM initialization (its manifest carries the
        engine facts), has ended, or has had PAIR_START_TIMEOUT_S to get there."""
        if first.state in TERMINAL or not first.training:
            return True
        if first.first_started is not None and now - first.first_started >= self.pair_start_timeout_s:
            return True
        manifest = first.current_manifest()
        return bool(manifest and manifest.get("vllm"))

    def supervise(self, job: Job, entry: dict, est: dict, jobs: list[Job], thresholds: Thresholds, now: float) -> None:
        if job.state != "RUNNING":
            return
        code = job.handle.poll()
        if code is not None:
            if code == 0:
                self.end_attempt(est, job, code, "COMPLETE")
                job.state = "COMPLETE"
                self.event(est, job, "complete", f"attempt {job.attempts} ended COMPLETE")
            else:
                self.end_attempt(est, job, code, "EXIT_NONZERO")
                self.event(est, job, "exited", f"attempt {job.attempts} exited with code {code} without COMPLETE")
                self.retry_or_fail(job, entry, est, jobs, f"exit code {code}")
            return
        sizes = progress_sizes(job.progress_patterns())
        if sizes != job.sizes:
            job.sizes, job.last_progress = sizes, now
        if stalled(now, job.started, job.last_progress, thresholds):
            idle_min = (now - (job.last_progress if job.last_progress is not None else job.started)) / 60
            self.kill(job.handle)
            self.end_attempt(est, job, job.handle.poll(), "STALLED")
            self.event(est, job, "stalled", f"attempt {job.attempts}: no progress for {idle_min:.0f} min; its process group was killed")
            self.retry_or_fail(job, entry, est, jobs, "stalled")

    def retry_or_fail(self, job: Job, entry: dict, est: dict, jobs: list[Job], reason: str) -> None:
        if job.attempts < MAX_ATTEMPTS:
            self.event(est, job, "relaunch", f"after {reason}: attempt {job.attempts + 1} of {MAX_ATTEMPTS}")
            self.sleep(self.release_wait_s)
            self.start(job, entry, est, jobs)
        else:
            job.state = "FAILED"
            self.event(est, job, "failed", f"after {reason}; no relaunch left")

    def stop(self, est: dict, job: Job, kind: str, detail: str) -> None:
        """Stop a job for good (no relaunch), for the step-0 input check."""
        if job.state == "RUNNING":
            self.kill(job.handle)
            self.end_attempt(est, job, job.handle.poll(), "STOPPED")
        job.state = "FAILED"
        self.event(est, job, kind, detail)


def format_status(status: dict, entries: list[dict] | None = None) -> str:
    """A compact report of a queue's status file (and its entries, so entries not yet reached show as PENDING)."""
    runner = status.get("runner") or {}
    line = (f"queue runner {runner.get('state', 'never started')} on {runner.get('host')} (pid {runner.get('pid')}, "
            f"heartbeat {runner.get('heartbeat_utc')})")
    if status.get("empty_utc"):
        line += f" | queue empty since {status['empty_utc']}"
    if runner.get("error"):
        line += f" | {runner['error']}"
    lines = [line]
    known = status.get("entries", {})
    ids = [e["id"] for e in entries] if entries is not None else list(known)
    labels = {e["id"]: e for e in entries or []}
    for entry_id in ids:
        est = known.get(entry_id)
        if est is None:
            entry = labels.get(entry_id, {})
            names = ", ".join(j["name"] for j in entry.get("jobs", []))
            lines.append(f"  {entry_id} {entry.get('kind')} {entry.get('label') or ''}: PENDING | {names}")
            continue
        jobs = "; ".join(
            f"{name} {rec.get('state')} (slot {rec.get('slot')}, attempts {len(rec.get('attempts') or [])}, "
            f"last progress {rec.get('last_progress_utc')})" for name, rec in est.get("jobs", {}).items())
        text = f"  {entry_id} {est.get('kind')} {est.get('label') or ''}: {est.get('state')} ({est.get('start_utc')} to {est.get('end_utc')}) | {jobs}"
        step0 = est.get("step0") or {}
        if step0:
            inputs = step0.get("inputs") or {}
            scores = step0.get("scores") or {}
            text += f" | step 0: inputs {'match' if inputs.get('match') else inputs.get('differing', 'pending')}"
            if scores.get("compared"):
                text += f", scores {scores['identities'] - scores['differing']}/{scores['identities']} equal"
        lines.append(text)
        for event in (est.get("events") or [])[-3:]:
            lines.append(f"    {event['utc']} {event.get('job') or ''} {event['event']}: {event['detail']}")
    return "\n".join(lines)

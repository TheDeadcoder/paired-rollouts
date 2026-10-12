"""The droplet queue (pairedrl.ops.queue) and its stall watchdog (pairedrl.ops.watchdog), driven by a fake clock and
fake job processes: slot ports and directories, entry order, appends while the queue runs, pair start-up and
independence, the single relaunch, stall decisions, and the step-0 identity check."""

import itertools
import json
import pathlib

import pytest
from synthetic import periodic_records, small_spec

from pairedrl.ops import queue as q
from pairedrl.ops.watchdog import Thresholds, progress_sizes, stalled
from pairedrl.train.runner import eval_sets_sha256, load_task_splits

INPUTS = {"git_commit": "abc1234", "deployed_commit": "abc1234", "model_commit_hash": "m1", "initial_adapter_sha256": "a1",
          "versions": {"trl": "1.12.0", "vllm": "0.27.1"}}


def test_slot_environment_keeps_two_jobs_apart(tmp_path):
    zero, one = q.slot_environment(0, tmp_path), q.slot_environment(1, tmp_path)
    assert zero["MASTER_PORT"] != one["MASTER_PORT"] and zero["VLLM_PORT"] != one["VLLM_PORT"]
    assert {zero["MASTER_PORT"], one["MASTER_PORT"]}.isdisjoint({zero["VLLM_PORT"], one["VLLM_PORT"]})
    for key in q.SLOT_DIRS:
        assert zero[key] != one[key] and zero[key].startswith(str(tmp_path))
    with pytest.raises(ValueError):
        q.slot_environment(2, tmp_path)


def test_watchdog_decisions_with_a_fake_clock(tmp_path):
    t = Thresholds(startup_grace_s=2700, stall_s=2700)
    assert not stalled(2699, 0, None, t) and not stalled(5399, 0, None, t) and stalled(5400, 0, None, t)
    assert not stalled(5399, 0, 600, t) and stalled(5400, 0, 600, t)
    assert not stalled(5699, 0, 3000, t) and stalled(5700, 0, 3000, t)
    log = tmp_path / "job.log"
    assert progress_sizes([log, tmp_path / "*.jsonl"]) == (0, 0)
    log.write_text("abc")
    (tmp_path / "e.jsonl").write_text("{}\n")
    assert progress_sizes([log, tmp_path / "*.jsonl"]) == (3, 3)


# ---- a fake world: clock, processes and their behaviours --------------------------------------------------------

class FakeProc:
    pids = itertools.count(1000)

    def __init__(self, world, name, attempt, argv, env, log_path, behavior):
        self.pid, self.world, self.name, self.attempt = next(self.pids), world, name, attempt
        self.argv, self.env, self.log_path, self.behavior = argv, env, pathlib.Path(log_path), behavior
        self.start, self.code, self.killed = world.clock, None, False
        self.run_dir = world.runs_dir / name

    def poll(self):
        return self.code

    def wait(self):
        return self.code


class World:
    def __init__(self, tmp_path):
        self.clock = 1_000_000.0
        self.runs_dir = tmp_path / "runs"
        self.queue_dir = self.runs_dir / "queue"
        self.checkout = tmp_path / "checkout"
        (self.checkout / "configs").mkdir(parents=True)
        (self.checkout / "src").mkdir()
        self.behaviors, self.procs, self.kills, self.hooks = {}, [], [], []

    def spec(self, name, **kw):
        spec = small_spec(run_id=name, **kw)
        (self.checkout / "configs" / f"{name}.json").write_text(json.dumps(spec.to_dict()))
        return spec

    def entry(self, kind, *names, **extra):
        return {"kind": kind, "label": "test", "commit": "abc1234", "checkout": str(self.checkout),
                "jobs": [{"name": n, "spec": f"configs/{n}.json"} for n in names], **extra}

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds
        for hook in list(self.hooks):
            hook(self)
        for proc in list(self.procs):
            if proc.code is None:
                proc.behavior(proc, self.clock)

    def spawn(self, argv, cwd, env, log_path):
        name = pathlib.Path(argv[argv.index("--spec") + 1]).stem if "--spec" in argv else argv[-1]
        attempt = sum(1 for p in self.procs if p.name == name) + 1
        proc = FakeProc(self, name, attempt, argv, env, log_path, self.behaviors[name][attempt - 1])
        self.procs.append(proc)
        proc.behavior(proc, self.clock)
        return proc

    def kill(self, proc):
        self.kills.append(proc.name)
        proc.killed, proc.code = True, -9

    def runner(self, **kw):
        return q.QueueRunner(self.queue_dir, self.runs_dir, "test", 1.0, python="python", poll_s=60, spawn=self.spawn,
                             kill=self.kill, clock=self.now, sleep=self.sleep, release_wait_s=0, pair_start_timeout_s=1200,
                             cache_root=self.runs_dir.parent / "cache", env={"PATH": "/usr/bin"}, **kw)

    def started(self, name):
        return [p for p in self.procs if p.name == name]


def write_manifest(proc, spec, vllm=True, **inputs):
    proc.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": proc.name, "attempt": proc.attempt, "status": "RUNNING", "started_utc": q.utc_now(),
                "spec": spec.to_dict(), "vllm": {"kv_cache_tokens": 1000} if vllm else {}, **INPUTS,
                "eval_sets_sha256": "e1", **inputs}
    (proc.run_dir / "run_manifest.json").write_text(json.dumps(manifest))


def healthy(spec, duration=600, vllm_after=0, step0=None, **inputs):
    """Writes its manifest (engine facts after `vllm_after` s), optional step-0 records, then one line a tick; exits 0."""
    def behave(proc, now):
        if now == proc.start:
            write_manifest(proc, spec, vllm=vllm_after == 0, **inputs)
            if step0 is not None:
                records = periodic_records(spec, [0], step0)
                (proc.run_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        if vllm_after and now - proc.start >= vllm_after:
            write_manifest(proc, spec, vllm=True, **inputs)
        with open(proc.run_dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps({"phase": "train:step1", "t": now}) + "\n")
        if now - proc.start >= duration:
            proc.code = 0
    return behave


def silent(spec=None):
    def behave(proc, now):
        pass
    return behave


def exits(spec, code, after=120):
    def behave(proc, now):
        if now == proc.start:
            write_manifest(proc, spec)
        with open(proc.log_path, "a") as f:
            f.write(".")
        if now - proc.start >= after:
            proc.code = code
    return behave


def test_entries_run_in_order_and_a_pair_waits_for_the_first_engine(tmp_path):
    w = World(tmp_path)
    a, b, c = w.spec("a"), w.spec("b"), w.spec("c")
    w.behaviors = {"a": [healthy(a, 300)], "b": [healthy(b, 1800, vllm_after=240)], "c": [healthy(c, 1800)]}
    assert q.append_entry(w.queue_dir, w.entry("run", "a")) == ("e0001", False)
    assert q.append_entry(w.queue_dir, w.entry("pair", "b", "c"))[0] == "e0002"
    assert w.runner().run() == 0
    first_b, first_c = w.started("b")[0], w.started("c")[0]
    assert w.started("a")[0].start < first_b.start and first_c.start - first_b.start >= 240
    assert first_b.argv[first_b.argv.index("--slot") + 1] == "0" and first_c.argv[first_c.argv.index("--slot") + 1] == "1"
    assert first_c.argv[first_c.argv.index("--shares-gpu-with") + 1] == "b"
    assert float(first_c.argv[first_c.argv.index("--start-offset-s") + 1]) >= 240
    assert first_b.env["MASTER_PORT"] != first_c.env["MASTER_PORT"] and first_b.env["PYTHONPATH"].startswith(str(w.checkout / "src"))
    status = q.read_status(w.queue_dir)
    assert [status["entries"][e]["state"] for e in ("e0001", "e0002")] == ["COMPLETE", "COMPLETE"]
    assert status["empty_utc"] and status["runner"]["state"] == "EXITED"
    assert status["entries"]["e0002"]["jobs"]["c"]["start_offset_s"] >= 240
    assert status["entries"]["e0002"]["step0"]["inputs"]["match"] and w.kills == []
    assert "e0002 pair test: COMPLETE" in q.format_status(status, q.read_entries(w.queue_dir))


def test_appends_while_running_are_picked_up_and_a_drained_queue_needs_a_new_runner(tmp_path):
    w = World(tmp_path)
    a, d, d2 = w.spec("a"), w.spec("d"), w.spec("d2")
    w.behaviors = {"a": [healthy(a, 600)], "d": [healthy(d, 300)], "d2": [healthy(d2, 300)]}
    replies = []

    def append_once(world):
        if not replies and world.started("a"):
            replies.append(q.append_entry(world.queue_dir, world.entry("run", "d")))

    w.hooks.append(append_once)
    q.append_entry(w.queue_dir, w.entry("run", "a"))
    w.runner().run()
    assert replies == [("e0002", True)] and len(w.started("d")) == 1
    assert q.read_status(w.queue_dir)["entries"]["e0002"]["state"] == "COMPLETE"
    assert q.append_entry(w.queue_dir, w.entry("run", "d2")) == ("e0003", False)
    assert w.runner().run() == 0 and q.read_status(w.queue_dir)["entries"]["e0003"]["state"] == "COMPLETE"


def test_a_stalled_job_is_relaunched_once_then_failed_and_never_touches_its_pair(tmp_path):
    w = World(tmp_path)
    w.spec("b")
    c, e = w.spec("c"), w.spec("e")
    w.behaviors = {"b": [silent(), silent()], "c": [healthy(c, 12000)], "e": [healthy(e, 300)]}
    q.append_entry(w.queue_dir, w.entry("pair", "b", "c", startup_grace_s=600, stall_s=600))
    q.append_entry(w.queue_dir, w.entry("run", "e"))
    w.runner().run()
    status = q.read_status(w.queue_dir)
    pair = status["entries"]["e0001"]
    assert w.kills == ["b", "b"] and len(w.started("b")) == 2 and len(w.started("c")) == 1
    assert w.started("c")[0].start - w.started("b")[0].start >= 1200
    assert [a["outcome"] for a in pair["jobs"]["b"]["attempts"]] == ["STALLED", "STALLED"]
    assert pair["jobs"]["b"]["state"] == "FAILED" and pair["jobs"]["c"]["state"] == "COMPLETE" and pair["state"] == "FAILED"
    kinds = [ev["event"] for ev in pair["events"] if ev["job"] == "b"]
    assert kinds.count("stalled") == 2 and kinds.count("relaunch") == 1 and kinds[-1] == "failed"
    events = [json.loads(line)["event"] for line in (w.runs_dir / "b" / "queue_events.jsonl").read_text().splitlines()]
    assert events.count("stalled") == 2 and "failed" in events
    assert status["entries"]["e0002"]["state"] == "COMPLETE"


def test_an_exit_without_complete_and_a_stall_share_the_single_relaunch(tmp_path):
    w = World(tmp_path)
    f = w.spec("f")
    w.behaviors = {"f": [exits(f, 1), silent()]}
    q.append_entry(w.queue_dir, w.entry("run", "f", startup_grace_s=300, stall_s=300))
    w.runner().run()
    job = q.read_status(w.queue_dir)["entries"]["e0001"]["jobs"]["f"]
    assert [a["outcome"] for a in job["attempts"]] == ["EXIT_NONZERO", "STALLED"] and job["state"] == "FAILED"
    assert job["attempts"][0]["exit_code"] == 1


def test_step0_inputs_that_differ_stop_both_arms(tmp_path):
    w = World(tmp_path)
    g, h = w.spec("g"), w.spec("h")
    w.behaviors = {"g": [healthy(g, 5000)], "h": [healthy(h, 5000, eval_sets_sha256="e2")]}
    q.append_entry(w.queue_dir, w.entry("pair", "g", "h"))
    w.runner().run()
    est = q.read_status(w.queue_dir)["entries"]["e0001"]
    assert est["state"] == "FAILED" and est["step0"]["inputs"]["differing"] == ["eval_sets_sha256"]
    assert sorted(w.kills) == ["g", "h"] and len(w.started("g")) == len(w.started("h")) == 1
    assert all(est["jobs"][n]["attempts"][-1]["outcome"] == "STOPPED" for n in ("g", "h"))
    for name in ("g", "h"):
        assert "step0_inputs_differ" in (w.runs_dir / name / "queue_events.jsonl").read_text()


def test_step0_scores_that_differ_with_equal_inputs_are_recorded_and_both_continue(tmp_path):
    w = World(tmp_path)
    i, j = w.spec("i"), w.spec("j")
    same = (lambda name, step, ti, k: (ti + k) % 2 == 0)
    one_off = (lambda name, step, ti, k: not same(name, step, ti, k) if (name, ti, k) == ("noisy", 0, 0) else same(name, step, ti, k))
    w.behaviors = {"i": [healthy(i, 1500, step0=same)], "j": [healthy(j, 1500, step0=one_off)]}
    q.append_entry(w.queue_dir, w.entry("pair", "i", "j"))
    w.runner().run()
    est = q.read_status(w.queue_dir)["entries"]["e0001"]
    assert est["state"] == "COMPLETE" and w.kills == []
    scores = est["step0"]["scores"]
    assert scores["compared"] and scores["differing"] == 1 and scores["identities"] == 4 + 4 + 2
    assert any(ev["event"] == "step0_scores_differ" for ev in est["events"])


def test_step0_input_comparison_ignores_arm_and_labels_only():
    base = dict(INPUTS, eval_sets_sha256="e1", spec=small_spec(run_id="p").to_dict())
    other = dict(base, spec=small_spec(run_id="i", arm="independent", notes="x").to_dict())
    assert q.input_differences(base, other) == []
    assert q.input_differences(base, dict(other, spec=small_spec(run_id="i", arm="independent", seed=3).to_dict())) == ["spec.seed"]
    splits = load_task_splits(pathlib.Path(__file__).resolve().parents[1] / "data" / "tasks")
    assert eval_sets_sha256(small_spec(), splits) == eval_sets_sha256(small_spec(arm="independent"), splits)


def test_entry_validation_and_a_second_runner(tmp_path):
    ok = {"kind": "command", "commit": "c", "checkout": str(tmp_path), "jobs": [{"name": "p", "argv": ["x"], "progress": ["*.json"]}]}
    q.validate_entry(ok)
    bad = [
        dict(ok, kind="chain"),
        dict(ok, kind="pair"),
        dict(ok, jobs=[{"name": "p", "argv": ["x"]}]),
        dict(ok, kind="run"),
        dict(ok, jobs=[{"name": "p", "argv": ["x"], "spec": "s.json", "progress": ["y"]}]),
        dict(ok, kind="pair", jobs=[{"name": "p", "spec": "a.json"}, {"name": "p", "spec": "b.json"}]),
        dict(ok, checkout=""),
        dict(ok, stall_s=0),
    ]
    for entry in bad:
        with pytest.raises(ValueError):
            q.validate_entry(entry)
    held = q.try_lock(tmp_path / "runner.lock")
    assert q.runner_alive(tmp_path)
    assert q.QueueRunner(tmp_path, tmp_path, "test", 1.0).run() == 0
    q.unlock(held)
    assert not q.runner_alive(tmp_path)

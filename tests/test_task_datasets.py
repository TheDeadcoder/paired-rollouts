import hashlib
import json
import pathlib
import sys

import pytest

from pairedrl.env.backoffice import TEMPLATES, read_jsonl, simulate_plan
from pairedrl.env.backoffice.tasks import SPLIT_SEED_RANGES

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tasks"
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not (DATA / "MANIFEST.json").exists(), reason="datasets not built")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def datasets():
    return {name: read_jsonl(DATA / f"{name}.jsonl") for name in ("train", "heldout", "diagnostic")}


def test_counts_and_ids(datasets):
    assert len(datasets["train"]) == 2000
    assert len(datasets["heldout"]) == 300
    assert len(datasets["diagnostic"]) == 16
    for name, tasks in datasets.items():
        assert [t.task_id for t in tasks] == [f"{name}-{i:05d}" for i in range(len(tasks))]
        assert all(t.split == name for t in tasks)


def test_world_seeds_unique_and_ranged(datasets):
    seen = set()
    for name, tasks in datasets.items():
        seeds = [t.world_seed for t in tasks]
        assert len(set(seeds)) == len(seeds), name
        assert not (set(seeds) & seen), name
        seen |= set(seeds)
        lo, hi = SPLIT_SEED_RANGES["train" if name == "train" else "heldout"]
        assert all(lo <= s < hi for s in seeds), name


def test_checksums_match_manifest(manifest):
    for name, info in manifest["files"].items():
        path = ROOT / info["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], name
        assert info["count"] == sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def test_generator_code_unchanged_since_build(manifest):
    from build_tasks import generator_code_sha256

    assert generator_code_sha256() == manifest["generator_code_sha256"], (
        "environment code changed after the datasets were built; rebuild with scripts/build_tasks.py"
    )


def test_template_coverage(datasets):
    for name in ("train", "heldout"):
        templates = {t for task in datasets[name] for t in task.templates}
        assert templates == set(TEMPLATES), name


def test_oracle_plans_execute_cleanly(datasets):
    sample = datasets["heldout"] + datasets["diagnostic"] + datasets["train"][::10]
    for task in sample:
        _, outcomes = simulate_plan(task.world_seed, task.oracle_plan)
        assert all(o["ok"] for o in outcomes), task.task_id

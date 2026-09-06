"""Build the frozen task datasets for Environment 1 and write a manifest with checksums."""

import argparse
import hashlib
import json
import pathlib
from collections import Counter

import pairedrl
from pairedrl.env.backoffice import Task, generate_tasks, write_jsonl

GENERATOR_SEED = 20260904
SPECS = {"train": ("train", 2000, 60), "heldout": ("heldout", 300, 0), "diagnostic": ("heldout", 16, 0), "test": ("heldout", 300, 0)}
DIAGNOSTIC_SUBGOALS = (2, 4)
CODE_FILES = ("world.py", "tools.py", "tasks.py", "grader.py")


def generator_code_sha256() -> str:
    base = pathlib.Path(pairedrl.__file__).parent / "env" / "backoffice"
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update((base / name).read_bytes())
    return h.hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dedupe(tasks: list[Task], used: set[int]) -> list[Task]:
    kept = []
    for task in tasks:
        if task.world_seed in used:
            continue
        used.add(task.world_seed)
        kept.append(task)
    return kept


def relabel(tasks: list[Task], split: str) -> list[Task]:
    out = []
    for i, task in enumerate(tasks):
        data = task.to_dict()
        data["task_id"] = f"{split}-{i:05d}"
        data["split"] = split
        out.append(Task.from_dict(data))
    return out


def build(out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[int] = set()
    heldout_pool = dedupe(generate_tasks(700, "heldout", GENERATOR_SEED), used)
    train_pool = dedupe(generate_tasks(2060, "train", GENERATOR_SEED), used)
    diagnostic_pool = [t for t in heldout_pool[300:360] if DIAGNOSTIC_SUBGOALS[0] <= t.n_subgoals <= DIAGNOSTIC_SUBGOALS[1]]
    diagnostic = diagnostic_pool[:16]
    diagnostic_seeds = {t.world_seed for t in diagnostic}
    test_pool = [t for t in heldout_pool[360:] if t.world_seed not in diagnostic_seeds]
    datasets = {
        "train": relabel(train_pool[:2000], "train"),
        "heldout": relabel(heldout_pool[:300], "heldout"),
        "diagnostic": relabel(diagnostic, "diagnostic"),
        "test": relabel(test_pool[:300], "test"),
    }
    manifest = {
        "generator_seed": GENERATOR_SEED,
        "pairedrl_version": pairedrl.__version__,
        "generator_code_sha256": generator_code_sha256(),
        "files": {},
    }
    for name, tasks in datasets.items():
        assert len(tasks) == SPECS[name][1], (name, len(tasks))
        path = out_dir / f"{name}.jsonl"
        write_jsonl(tasks, path)
        manifest["files"][name] = {
            "path": str(path.as_posix()),
            "count": len(tasks),
            "sha256": file_sha256(path),
            "templates": dict(sorted(Counter(t for task in tasks for t in task.templates).items())),
            "requests_per_task": dict(sorted(Counter(len(t.templates) for t in tasks).items())),
            "write_calls_per_task": dict(sorted(Counter(t.n_subgoals for t in tasks).items())),
        }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tasks")
    args = parser.parse_args()
    manifest = build(pathlib.Path(args.out))
    for name, info in manifest["files"].items():
        print(name, info["count"], info["sha256"])
    print("generator_code_sha256", manifest["generator_code_sha256"])


if __name__ == "__main__":
    main()

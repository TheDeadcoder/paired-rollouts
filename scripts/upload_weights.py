"""Upload a verified run's trainer checkpoints and final adapter to the project's Hugging Face model repository,
in one commit, and record the durable copy.

    python scripts/upload_weights.py outputs/runs/<run_id> --repo <user>/<name> [--data-dir data/tasks] [--out-dir registers/weights]

Refuses a run that verify_run would not ACCEPT with --weights. Uploads every file under trainer/checkpoint-*/ and
adapter_final/ to <run_id>/<same relative path> in the repository and writes registers/weights/<run_id>.json: the
repository, the upload revision and URL, the upload time, the run's code commit, the checkpoint inventory the
verdict saw, and (path, bytes, sha256) of every uploaded file. Authentication is the token of `huggingface-cli login`.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import sys

from pairedrl.analysis.curves import run_register

HERE = pathlib.Path(__file__).resolve().parent


def load_verdict():
    spec = importlib.util.spec_from_file_location("verify_run", HERE / "verify_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verdict


def archive_files(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every file of the trainer checkpoints and of adapter_final, in a fixed order."""
    files = sorted(p for p in (run_dir / "trainer").glob("checkpoint-*/*") if p.is_file())
    files += sorted(p for p in (run_dir / "adapter_final").glob("*") if p.is_file())
    return files


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(run_dir: pathlib.Path, files: list[pathlib.Path]) -> list[dict]:
    return [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]


def hf_commit(repo: str, uploads: list[tuple[str, pathlib.Path]], message: str) -> tuple[str, str]:
    """One commit with every file; returns (revision, commit url)."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local)) for path, local in uploads]
    info = api.create_commit(repo_id=repo, repo_type="model", operations=operations, commit_message=message)
    return info.oid, info.commit_url


def upload_run(run_dir, repo: str, commit_fn=hf_commit, data_dir=None, out_dir="registers/weights") -> dict:
    """Verify, upload and record; raises ValueError with the verdict's reasons when the run is not accepted."""
    run_dir = pathlib.Path(run_dir)
    entry = run_register(run_dir, data_dir=data_dir)
    ok, reasons = load_verdict()(entry, weights=True)
    if not ok:
        raise ValueError(f"{entry['run_id']} is not accepted with weights: {'; '.join(reasons)}")
    files = archive_files(run_dir)
    listed = inventory(run_dir, files)
    uploads = [(f"{entry['run_id']}/{item['path']}", path) for item, path in zip(listed, files, strict=True)]
    revision, url = commit_fn(repo, uploads, f"{entry['run_id']}: trainer checkpoints and final adapter (code {entry['git_commit']})")
    record = {
        "run_id": entry["run_id"],
        "repo": repo,
        "revision": revision,
        "commit_url": url,
        "uploaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": entry["git_commit"],
        "provider": entry["provider"],
        "attempts": [a["attempt"] for a in entry["attempts"]],
        "checkpoints": entry["checkpoints"],
        "files": listed,
        "total_bytes": sum(item["bytes"] for item in listed),
    }
    out = pathlib.Path(out_dir) / f"{entry['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--repo", required=True, help="Hugging Face model repository, <user>/<name>")
    parser.add_argument("--data-dir", default="data/tasks")
    parser.add_argument("--out-dir", default="registers/weights")
    args = parser.parse_args()
    try:
        record = upload_run(args.run_dir, args.repo, data_dir=args.data_dir, out_dir=args.out_dir)
    except ValueError as e:
        print(f"refusing to upload: {e}")
        return 1
    print(f"uploaded {len(record['files'])} files ({record['total_bytes'] / 1e9:.2f} GB) to {record['repo']} at {record['revision']}")
    print(f"wrote {pathlib.Path(args.out_dir) / (record['run_id'] + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Archive verified runs' weights to Hugging Face from wherever the run directories live (on Modal, the read-only runs
volume: scripts/archive_modal.py). Each run is uploaded once through scripts/upload_weights.upload_run (verify_run with
weights, one Hugging Face commit, the weights record), then checked: the adapter of its last complete checkpoint is
downloaded from the Hub at the recorded revision and its sha256 compared with the record. One status file per run makes
the loop resumable: a verified run is skipped, and a run whose record exists is never uploaded again, only re-verified."""

import datetime as dt
import hashlib
import json
import pathlib
import time

UPLOAD_TRIES = 3


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest(paths: dict) -> dict:
    """name -> sha256 of the file at each path: which code did the archiving, comparable between the launching tree
    and the deployed image."""
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def last_adapter(record: dict) -> tuple[str, str]:
    """(path in the repository, recorded sha256) of the adapter weights of the run's last complete checkpoint."""
    last = max(record["checkpoints"]["complete"])
    rel = f"trainer/checkpoint-{last}/adapter_model.safetensors"
    sha = next(f["sha256"] for f in record["files"] if f["path"] == rel)
    return f"{record['run_id']}/{rel}", sha


def verify_record(record: dict, download_fn) -> dict:
    """Download the last checkpoint's adapter at the recorded revision (download_fn(repo, path, revision) returns a
    local path) and compare its sha256 with the record."""
    path, expected = last_adapter(record)
    got = sha256_file(download_fn(record["repo"], path, record["revision"]))
    return {"path": path, "revision": record["revision"], "sha256_expected": expected, "sha256_downloaded": got,
            "match": got == expected, "checked_utc": utc_now()}


def _write(path: pathlib.Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _upload(upload_fn, run_dir, repo, data_dir, records_dir, options, sleep) -> dict:
    """upload_run with bounded retries: a refusal (ValueError) is final, other errors are retried."""
    for attempt in range(1, UPLOAD_TRIES + 1):
        try:
            return upload_fn(run_dir, repo, data_dir=data_dir, out_dir=records_dir,
                             accept_missing=options.get("steps", ()), accept_reason=options.get("reason"))
        except ValueError:
            raise
        except Exception:
            if attempt == UPLOAD_TRIES:
                raise
            sleep(60 * attempt)
    raise AssertionError("unreachable")


def archive_runs(run_ids, runs_root, out_dir, repo, upload_fn, download_fn, accept_missing=None, data_dir=None,
                 persist=None, extra=None, sleep=time.sleep) -> dict:
    """For each run: upload it unless out_dir/records/<run_id>.json exists, verify it, and write
    out_dir/status/<run_id>.json (UPLOADED, VERIFIED, MISMATCH or FAILED, with the revision, commit URL, file count,
    bytes, verification and any error). `upload_fn` has upload_run's signature; `accept_missing` maps a run id to
    {"steps": [...], "reason": "..."}; `persist` (a volume commit) runs after every write; `extra` is copied into every
    status. Returns run id -> status."""
    runs_root, out_dir = pathlib.Path(runs_root), pathlib.Path(out_dir)
    persist = persist or (lambda: None)
    results = {}
    for run_id in run_ids:
        status_path = out_dir / "status" / f"{run_id}.json"
        record_path = out_dir / "records" / f"{run_id}.json"
        status = _read(status_path) or {"run_id": run_id}
        status.update(extra or {})
        record = _read(record_path)
        if status.get("status") == "VERIFIED" and record is not None:
            results[run_id] = status
            continue
        try:
            if record is None:
                options = (accept_missing or {}).get(run_id) or {}
                record = _upload(upload_fn, runs_root / run_id, repo, data_dir, out_dir / "records", options, sleep)
                status.update({"status": "UPLOADED", "uploaded_utc": record["uploaded_utc"]})
                status.pop("error", None)
            status.update({"revision": record["revision"], "commit_url": record["commit_url"],
                           "files": len(record["files"]), "total_bytes": record["total_bytes"]})
            _write(status_path, status)
            persist()
            verification = verify_record(record, download_fn)
            status.update({"status": "VERIFIED" if verification["match"] else "MISMATCH", "verification": verification})
            status.pop("error", None)
        except Exception as e:  # noqa: BLE001
            status.update({"status": "FAILED", "error": f"{type(e).__name__}: {e}", "failed_utc": utc_now()})
        _write(status_path, status)
        persist()
        results[run_id] = status
    return results

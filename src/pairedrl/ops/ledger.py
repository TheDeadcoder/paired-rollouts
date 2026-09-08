"""Run-ledger rows and launch registers."""

import datetime as dt
import json
import pathlib
import re

from pairedrl.train.runner import RunSpec

LEDGER_HEADER = "| run_id | launched_utc | provider | model | condition | arm | seed | steps | status | notes |"
TERMINAL = ("COMPLETE", "FAILED", "INFRA_FAILED", "CANCELLED")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def preregistration_note(spec: RunSpec, preregistration: str | None) -> str:
    """Provenance suffix of a ledger row. A pre-registration label is accepted only when the spec's own notes carry
    it, so a launch flag cannot relabel a spec that was not part of the frozen document."""
    if preregistration is None:
        return "not pre-registered"
    if f"pre-registered {preregistration}" not in (spec.notes or ""):
        raise ValueError(f"{spec.run_id}: launched as pre-registered {preregistration} but its notes do not say so")
    return f"pre-registered {preregistration}"


def ledger_row(
    spec: RunSpec,
    launched_utc: str,
    call_id: str,
    status: str,
    provider: str = "modal",
    preregistration: str | None = None,
) -> str:
    steps = 0 if spec.eval_only else spec.steps
    notes = f"call {call_id}"
    if spec.notes:
        notes += f"; {spec.notes}"
    notes += "; " + preregistration_note(spec, preregistration)
    notes = notes.replace("|", "/").replace("\n", " ")
    return (
        f"| {spec.run_id} | {launched_utc} | {provider} | {spec.model} | {spec.condition} | {spec.arm} | "
        f"{spec.seed} | {steps} | {status} | {notes} |"
    )


def append_rows(ledger_path, rows: list[str]) -> None:
    path = pathlib.Path(ledger_path)
    text = path.read_text(encoding="utf-8")
    if LEDGER_HEADER not in text:
        raise ValueError(f"{path} does not contain the ledger table header")
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def set_status(ledger_path, call_id: str, status: str) -> bool:
    """Rewrite the status cell of the row whose notes mention `call <call_id>`. Returns whether a row changed."""
    path = pathlib.Path(ledger_path)
    lines = path.read_text(encoding="utf-8").split("\n")
    changed = False
    for i, line in enumerate(lines):
        if not line.startswith("|") or f"call {call_id}" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 10:
            continue
        if cells[8] != status:
            cells[8] = status
            lines[i] = "| " + " | ".join(cells) + " |"
            changed = True
    if changed:
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def launch_register_path(register_dir, label: str, launched_utc: str) -> pathlib.Path:
    stamp = re.sub(r"[^0-9T]", "", launched_utc.split("+")[0])
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", label)
    return pathlib.Path(register_dir) / f"{stamp}_{safe}.json"


def write_launch_register(
    path, commit: str, deployed_app: str, launched_utc: str, entries: list[dict], preregistration: str | None = None
) -> dict:
    register = {
        "launched_utc": launched_utc,
        "commit": commit,
        "app": deployed_app,
        "preregistration": preregistration,
        "entries": entries,
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return register


def read_launch_register(path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def latest_launch_register(register_dir) -> pathlib.Path:
    files = sorted(pathlib.Path(register_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no launch registers under {register_dir}")
    return files[-1]

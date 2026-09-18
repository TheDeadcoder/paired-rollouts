"""The gradient probe GPU job (H1(b), amendment A1), made correct, memory-bounded and auditable. Torch and TRL are
imported lazily so this module and its pure helpers import on any machine. The arithmetic is in
pairedrl.analysis.probe. One checkpoint is scored per process (run_probe drives subprocesses); each step writes a
summary and a gzipped evidence file from which the summary is recomputable without a GPU. See docs/PROBE.md."""

import contextlib
import dataclasses
import gzip
import hashlib
import json
import pathlib
import socket
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np

from pairedrl.analysis import probe as pm
from pairedrl.ops.job import ADAPTER_CONFIG, VERSION_PACKAGES, package_version, utc_now

SMOKE_DESIGN = {"diagnostic_tasks": 2, "schedules": 2, "samples": 2, "clean_tasks": 4, "clean_rollouts": 2, "resamples": 4}
CONDITION, ARM = "C2", "paired"
GRAM_CHUNK_BYTES = 2 * 1024 ** 3
TRAIN_GROUPS, TRAIN_GENERATIONS = 24, 8


@dataclass
class ProbeSpec:
    probe_id: str
    model: str
    trajectory: str
    steps: list[int]
    diagnostic_tasks: int = 16
    schedules: int = 8
    samples: int = 8
    p: float = 0.25
    clean_tasks: int = 64
    clean_rollouts: int = 8
    q: float = 0.10
    resamples: int = 64
    seed: int = 0
    fault_weights: dict[str, float] | None = None
    notes: str = ""
    smoke: bool = False
    validate_loss: bool = False
    max_completion_length: int = 6144
    max_tool_calling_iterations: int = 24
    vllm_max_model_length: int = 12288
    vllm_gpu_memory_utilization: float = 0.35
    per_device_eval_batch_size: int = 128
    logprob_chunk: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeSpec":
        return cls(**data)

    def design(self) -> dict:
        base = {"diagnostic_tasks": self.diagnostic_tasks, "schedules": self.schedules, "samples": self.samples,
                "clean_tasks": self.clean_tasks, "clean_rollouts": self.clean_rollouts, "resamples": self.resamples}
        return dict(SMOKE_DESIGN) if self.smoke else base

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def probe_ledger_row(spec: ProbeSpec, launched_utc: str, call_id: str, model: str, condition: str, arm: str,
                     status: str = "LAUNCHED", provider: str = "modal", preregistration: str | None = None) -> str:
    """The ten-column ledger row for a probe launch: run id = probe_id, the trajectory's condition and arm, seed,
    steps = number of checkpoints, and the pre-registration note when the probe belongs to the frozen protocol."""
    from pairedrl.ops.ledger import preregistration_note

    steps = ", ".join(str(s) for s in spec.steps)
    notes = f"call {call_id}; gradient probe H1(b) amendment A1 on {spec.trajectory} steps [{steps}]"
    if spec.notes:
        notes += f"; {spec.notes}"
    notes += "; " + preregistration_note(spec, preregistration)
    notes = notes.replace("|", "/").replace("\n", " ")
    return (
        f"| {spec.probe_id} | {launched_utc} | {provider} | {model} | {condition} | {arm} | "
        f"{spec.seed} | {len(spec.steps)} | {status} | {notes} |"
    )


# ---- A: disjoint captures ---------------------------------------------------

@contextlib.contextmanager
def capture_batches(trainer):
    """Install one wrapper around the instance's _generate_and_score_completions that appends (output, episodes) for
    each eval batch into one explicit buffer, and restore the original callable on exit. Each evaluate runs in its
    own context, so the diagnostic and clean captures never share a buffer (the earlier nested-wrapper bug)."""
    buffer: list = []
    original = trainer._generate_and_score_completions

    def wrapper(inputs):
        output = original(inputs)
        n = len(inputs)
        buffer.append((output, [env.last_episode for env in trainer.environments[:n]]))
        return output

    trainer._generate_and_score_completions = wrapper
    try:
        yield buffer
    finally:
        trainer._generate_and_score_completions = original


def flatten_episodes(captured) -> list:
    return [ep for _, episodes in captured for ep in episodes]


def assert_capture_identities(diag_captured, clean_captured, design) -> None:
    """Hard error unless the diagnostic capture is `diagnostic_tasks` tasks of schedules x samples rollouts each
    (partitioning into `schedules` distinct schedule seeds of `samples` rollouts, phase probe_diag) and the clean
    capture is `clean_tasks` tasks of `clean_rollouts` rollouts each (phase probe_clean, no faults exposed), with no
    task id in both."""
    diag, clean = flatten_episodes(diag_captured), flatten_episodes(clean_captured)
    bad = {(e.get("phase") or "").split(":")[0] for e in diag} - {"probe_diag"}
    if bad:
        raise AssertionError(f"diagnostic capture phases {bad}, expected probe_diag")
    diag_by_task = defaultdict(list)
    for e in diag:
        diag_by_task[e["task_id"]].append(e)
    if len(diag_by_task) != design["diagnostic_tasks"]:
        raise AssertionError(f"diagnostic tasks {len(diag_by_task)} != {design['diagnostic_tasks']}")
    for tid, eps in diag_by_task.items():
        if len(eps) != design["schedules"] * design["samples"]:
            raise AssertionError(f"diagnostic task {tid}: {len(eps)} rollouts != {design['schedules'] * design['samples']}")
        seeds = Counter(e["schedule_seed"] for e in eps)
        if len(seeds) != design["schedules"] or any(c != design["samples"] for c in seeds.values()):
            raise AssertionError(f"diagnostic task {tid}: schedule seeds {dict(seeds)}, expected {design['schedules']}x{design['samples']}")
    bad = {(e.get("phase") or "").split(":")[0] for e in clean} - {"probe_clean"}
    if bad:
        raise AssertionError(f"clean capture phases {bad}, expected probe_clean")
    clean_by_task = defaultdict(list)
    for e in clean:
        clean_by_task[e["task_id"]].append(e)
    if len(clean_by_task) != design["clean_tasks"]:
        raise AssertionError(f"clean tasks {len(clean_by_task)} != {design['clean_tasks']}")
    for tid, eps in clean_by_task.items():
        if len(eps) != design["clean_rollouts"]:
            raise AssertionError(f"clean task {tid}: {len(eps)} rollouts != {design['clean_rollouts']}")
        if any(e.get("exposed") for e in eps):
            raise AssertionError(f"clean task {tid}: a fault was exposed on a clean rollout")
    overlap = set(diag_by_task) & set(clean_by_task)
    if overlap:
        raise AssertionError(f"tasks in both diagnostic and clean captures: {sorted(overlap)}")


def _by_task(captured, order=None):
    tasks = OrderedDict()
    for output, episodes in captured:
        for i, episode in enumerate(episodes):
            tasks.setdefault(episode["task_id"], []).append((output, i, episode))
    if order is not None:
        tasks = OrderedDict((tid, tasks[tid]) for tid in order if tid in tasks)
    return tasks


# ---- B: score coordinates (trainable, text-only, stable) --------------------

def enable_lora_grads(model) -> None:
    """PEFT 0.20.0 loads a checkpoint adapter with inference_mode (no grads); turn grad on for the LoRA parameters
    and off for everything else, so the score is over the trainable adapter only."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)


def _any_gradient_checkpointing(model) -> bool:
    return any(getattr(module, "gradient_checkpointing", False) for module in model.modules())


def _enable_gradient_checkpointing(model) -> None:
    """Activation checkpointing for the single-sequence backward, so it does not materialize the full activations of
    an ~8000-token sequence next to vLLM's reservation. PeftModel forwards the call to the base model; if it does
    not, call it on base_model.model. LoRA dropout is 0 and the base has no dropout, so train mode changes only the
    checkpointing path (docs/PROBE.md)."""
    kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    model.gradient_checkpointing_enable(**kwargs)
    if not _any_gradient_checkpointing(model):
        model.base_model.model.gradient_checkpointing_enable(**kwargs)
    if not _any_gradient_checkpointing(model):
        raise AssertionError("gradient checkpointing did not enable on any module")


def lora_grad_names(model) -> list:
    return sorted(name for name, param in model.named_parameters() if "lora_" in name and param.requires_grad)


def select_coordinates(named_has_grad) -> list:
    """The ordered coordinate set: LoRA parameter names (sorted) whose gradient is not None after one backward. The
    vision-tower LoRA layers get no gradient on text-only rollouts; a coordinate that is always zero contributes
    nothing to any inner product, so this is the full-parameter measurement without the structural zeros."""
    return sorted(name for name, has_grad in named_has_grad if has_grad)


def check_coordinates_stable(named_has_grad, coords) -> None:
    """Hard error unless exactly the recorded coordinate set has gradients on this rollout and no other LoRA
    parameter does."""
    have = {name for name, has_grad in named_has_grad if has_grad}
    coordset = set(coords)
    if have != coordset:
        raise AssertionError(f"coordinate set changed: +{sorted(have - coordset)} -{sorted(coordset - have)}")


def _named_has_grad(model):
    return [(name, param.grad is not None) for name, param in model.named_parameters() if "lora_" in name]


def _bf16_bytes(tensor):
    import torch

    return tensor.detach().to(torch.bfloat16).cpu().contiguous().numpy().tobytes()


def lora_fingerprint(model, coords) -> str:
    """sha256 over the concatenated bf16 bytes of the coordinate-set parameters, in coordinate order. Asserting it
    is unchanged after scoring proves the backward passes did not mutate the weights."""
    params = dict(model.named_parameters())
    h = hashlib.sha256()
    for name in coords:
        h.update(_bf16_bytes(params[name].data))
    return h.hexdigest()


# ---- C: memory-bounded Gram and weighted sums -------------------------------

def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gram_chunked(scores, chunk_rows: int):
    """The fp64 Gram K_ij = S_i . S_j formed by moving row chunks (at most `chunk_rows` rows) to the GPU in fp64 and
    accumulating each K[i0:i1, j0:j1] block. `scores` is (n, P) on CPU. Returns an (n, n) numpy fp64 matrix."""
    import torch

    s = scores if torch.is_tensor(scores) else torch.as_tensor(np.asarray(scores))
    n = s.shape[0]
    out = torch.zeros((n, n), dtype=torch.float64)
    dev = _device()
    for i0 in range(0, n, chunk_rows):
        i1 = min(i0 + chunk_rows, n)
        ci = s[i0:i1].to(dev, torch.float64)
        for j0 in range(0, n, chunk_rows):
            j1 = min(j0 + chunk_rows, n)
            cj = s[j0:j1].to(dev, torch.float64)
            out[i0:i1, j0:j1] = (ci @ cj.T).cpu()
    return out.numpy()


def weighted_sum_chunked(weights, scores, chunk_rows: int):
    """sum_i w_i S_i as a P-vector, chunked over rows in fp64. Returns a numpy fp64 vector of length P."""
    import torch

    s = scores if torch.is_tensor(scores) else torch.as_tensor(np.asarray(scores))
    w = torch.as_tensor(np.asarray(weights), dtype=torch.float64)
    acc = torch.zeros(s.shape[1], dtype=torch.float64)
    dev = _device()
    for i0 in range(0, s.shape[0], chunk_rows):
        i1 = min(i0 + chunk_rows, s.shape[0])
        acc += (w[i0:i1].to(dev) @ s[i0:i1].to(dev, torch.float64)).cpu()
    return acc.numpy()


def _chunk_rows(p: int) -> int:
    return max(1, GRAM_CHUNK_BYTES // (max(p, 1) * 8))


# ---- E: importance ratio and the batch token normalizer ---------------------

def _mask_and_ratio(output):
    """The loss mask (completion_mask * tool_mask) and the per-sequence importance ratio rho (B,) from a captured
    generation output, on the CPU. rho is 1 when the correction is off."""
    import torch

    completion_mask = output["completion_mask"]
    tool_mask = output.get("tool_mask")
    mask = completion_mask if tool_mask is None else completion_mask * tool_mask
    ratio = output.get("importance_sampling_ratio")
    if ratio is None:
        rho = torch.ones(completion_mask.shape[0])
    else:
        rho = ratio.reshape(-1).float().cpu()
    return mask, rho


# ---- score one rollout ------------------------------------------------------

def score_rollout(trainer, output, i: int, coords):
    """One backward of (per_token_logps * mask).sum() over the coordinate set for rollout i, reproducing the loss's
    forward (input_ids = cat(prompt, completion), mask = completion_mask * tool_mask). Returns the fp32 gradient
    vector (CPU), the masked-token count T_i and the per-sequence importance ratio rho_i."""
    import torch

    params = dict(trainer.model.named_parameters())
    prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
    completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
    tool_mask = output.get("tool_mask")
    mask_i = (completion_mask if tool_mask is None else completion_mask * tool_mask)[i:i + 1]
    input_ids = torch.cat([prompt_ids[i:i + 1], completion_ids[i:i + 1]], dim=1)
    attention_mask = torch.cat([prompt_mask[i:i + 1], completion_mask[i:i + 1]], dim=1)
    logits_to_keep = completion_ids.size(1)
    trainer.model.zero_grad(set_to_none=True)
    logps, _, _ = trainer._get_per_token_logps_and_entropies(trainer.model, input_ids, attention_mask, logits_to_keep)
    (logps * mask_i).sum().backward()
    check_coordinates_stable(_named_has_grad(trainer.model), coords)
    flat = torch.cat([params[name].grad.reshape(-1).float() for name in coords]).detach().cpu()
    trainer.model.zero_grad(set_to_none=True)
    ratio = output.get("importance_sampling_ratio")
    rho_i = 1.0 if ratio is None else float(ratio.reshape(-1)[i].item())
    return flat, float(mask_i.sum().item()), rho_i


# ---- F: validate the reconstruction against the pinned loss -----------------

def validate_loss_gradient(trainer, validation, coords) -> dict:
    """Compare -grad of `trainer._compute_loss` on one clean group (advantages = grpo_advantages(true_success),
    ratio 1 via the batch's own old log-probs, num_items_in_batch = the group's masked tokens) with the
    reconstruction (1/T_group) sum_i A_i rho_i S_i from the already scored vectors. Returns cosine, norm ratio,
    relative L2 difference and T_group."""
    import torch

    from pairedrl.analysis.diagnostics import grpo_advantages

    output, indices = validation["output"], list(validation["indices"])
    true = np.asarray(validation["true"], dtype=float)
    rho = np.asarray(validation["rho"], dtype=float)
    scores = validation["scores"].numpy() if hasattr(validation["scores"], "numpy") else np.asarray(validation["scores"])
    keys = ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "tool_mask", "importance_sampling_ratio")
    sub = {k: output[k][indices] for k in keys if output.get(k) is not None}
    mask = sub["completion_mask"] if sub.get("tool_mask") is None else sub["completion_mask"] * sub["tool_mask"]
    t_group = float(mask.sum().item())
    sub["advantages"] = torch.tensor(grpo_advantages(true.tolist(), "group"), dtype=torch.float32, device=mask.device)
    sub["num_items_in_batch"] = mask.sum()
    steps_per_generation = trainer.args.steps_per_generation
    set_accum = not hasattr(trainer, "current_gradient_accumulation_steps")
    if set_accum:
        trainer.current_gradient_accumulation_steps = steps_per_generation
    trainer.model.zero_grad(set_to_none=True)
    try:
        trainer._compute_loss(trainer.model, sub).backward()
    finally:
        if set_accum:
            del trainer.current_gradient_accumulation_steps
    params = dict(trainer.model.named_parameters())
    grad = torch.cat([params[name].grad.reshape(-1).float() for name in coords]).detach().cpu().numpy()
    trainer.model.zero_grad(set_to_none=True)
    recon = (scores.T @ (np.asarray(grpo_advantages(true.tolist(), "group")) * rho)) / t_group
    minus_grad = -grad
    denom = np.linalg.norm(minus_grad) + 1e-30
    return {"cosine": float(minus_grad @ recon / (denom * (np.linalg.norm(recon) + 1e-30))),
            "norm_ratio": float(np.linalg.norm(recon) / denom),
            "relative_l2": float(np.linalg.norm(minus_grad - recon) / denom), "t_group": t_group,
            "steps_per_generation": steps_per_generation}


# ---- G: atomic writes and provenance-checked resume -------------------------

def atomic_write_json(path, obj) -> None:
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_gzip(path, obj) -> None:
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True)
    tmp.replace(path)


def _provenance(spec: ProbeSpec, adapter_sha, base_fingerprint, git_commit, design) -> dict:
    return {"spec_sha256": spec.sha256(), "trajectory": spec.trajectory, "adapter_sha256": adapter_sha,
            "base_fingerprint": base_fingerprint, "git_commit": git_commit,
            "design_counts": {k: design[k] for k in sorted(design)}}


def resume_matches(step_payload: dict, provenance: dict) -> tuple[bool, str]:
    """Whether an existing step file may be reused: its provenance must match the current job exactly. Returns
    (ok, reason); reason names the first mismatch when not ok."""
    have = step_payload.get("provenance", {})
    for key in ("spec_sha256", "trajectory", "adapter_sha256", "base_fingerprint", "git_commit", "design_counts"):
        if have.get(key) != provenance.get(key):
            return False, f"{key} mismatch: {have.get(key)!r} != {provenance.get(key)!r}"
    return True, "match"


def _sha256_file(path) -> str | None:
    path = pathlib.Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- within-task decomposition (numpy, testable) ----------------------------

def within_task_trace_transition(sum_sq_norm, summed_weights, gram, n_groups) -> float:
    """Within-task Tr Var for transition noise: mean ||g||^2 - ||mean g||^2 over the task's groups, with ||mean g||^2
    from the task's summed weights and its Gram."""
    return pm.trace_variance(sum_sq_norm, pm.quadratic(summed_weights, gram), n_groups)


def order_transition_members(members) -> list:
    """Lay a task's captured members out schedule-major by a stable sort on (schedule_seed, capture position), so
    rollout index k*M + m shares schedule k whatever order the eval dataloader delivered them in."""
    return [member for _, member in sorted(enumerate(members), key=lambda im: (im[1][2]["schedule_seed"], im[0]))]


def assert_schedule_layout(schedule_seeds, k_schedules, m_samples, task_id) -> None:
    """Hard error unless the ordered seeds are `k_schedules` consecutive blocks of `m_samples`, each block one seed
    and the block seeds distinct."""
    if len(schedule_seeds) != k_schedules * m_samples:
        raise AssertionError(f"transition task {task_id}: {len(schedule_seeds)} members != {k_schedules}x{m_samples}")
    block_seeds = []
    for b in range(k_schedules):
        block = set(schedule_seeds[b * m_samples:(b + 1) * m_samples])
        if len(block) != 1:
            raise AssertionError(f"transition task {task_id}: block {b} spans schedule seeds {block}")
        block_seeds.append(next(iter(block)))
    if len(set(block_seeds)) != k_schedules:
        raise AssertionError(f"transition task {task_id}: schedule seeds not distinct {block_seeds}")


# ---- the RunSpec the trainer is built with ----------------------------------

def _probe_run_spec(spec: ProbeSpec, design: dict, adapter_path):
    from pairedrl.train.runner import RunSpec

    return RunSpec(
        run_id=f"{spec.probe_id}-step", model=spec.model, condition=CONDITION, arm=ARM, p=spec.p, q=0.0,
        seed=spec.seed, eval_only=True, adapter_path=str(adapter_path) if adapter_path else None,
        diagnostic_schedules=design["schedules"], diagnostic_samples=design["samples"], fault_weights=spec.fault_weights,
        max_completion_length=spec.max_completion_length, max_tool_calling_iterations=spec.max_tool_calling_iterations,
        vllm_max_model_length=spec.vllm_max_model_length, vllm_gpu_memory_utilization=spec.vllm_gpu_memory_utilization,
        per_device_eval_batch_size=spec.per_device_eval_batch_size, logprob_chunk=spec.logprob_chunk,
    )


# ---- one checkpoint (the subprocess unit) -----------------------------------

def _probe_step(spec: ProbeSpec, step: int, runs_root, data_dir, out_dir, git_commit, deployed_commit, provider) -> dict:
    """Generate, score and accumulate one checkpoint; write step<N>.json and step<N>_evidence.json.gz atomically.
    Returns the step summary payload."""
    import torch
    import transformers
    from datasets import Dataset

    from pairedrl.noise.config import NoiseConfig
    from pairedrl.train.dataset import build_eval_rows
    from pairedrl.train.runner import assemble_diagnostic_rows, build_trainer, load_task_splits

    runs_root, out_dir = pathlib.Path(runs_root), pathlib.Path(out_dir)
    design = spec.design()
    adapter_path = None if step == 0 else runs_root / spec.trajectory / "trainer" / f"checkpoint-{step}"
    if adapter_path is not None and not (adapter_path / ADAPTER_CONFIG).exists():
        raise FileNotFoundError(f"probe {spec.probe_id}: checkpoint absent at {adapter_path}")
    adapter_sha = _sha256_file(adapter_path / "adapter_model.safetensors") if adapter_path else None

    if step == 0:
        transformers.set_seed(spec.seed)
    step_dir = out_dir / f"step{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    run_spec = _probe_run_spec(spec, design, adapter_path)
    splits = load_task_splits(data_dir)
    trainer, _, _ = build_trainer(run_spec, data_dir, step_dir / "trainer", step_dir / "episodes.jsonl")
    enable_lora_grads(trainer.model)

    diag_tasks = splits["diagnostic"][:design["diagnostic_tasks"]]
    diag_rows = assemble_diagnostic_rows(run_spec, diag_tasks)
    clean_rows = build_eval_rows(splits["heldout"][:design["clean_tasks"]], NoiseConfig.clean(), "probe:clean",
                                 list(range(design["clean_rollouts"])))
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    gen0 = time.perf_counter()
    with capture_batches(trainer) as diag_captured:
        trainer.evaluate(eval_dataset=Dataset.from_list(diag_rows), metric_key_prefix="probe_diag")
    with capture_batches(trainer) as clean_captured:
        trainer.evaluate(eval_dataset=Dataset.from_list(clean_rows), metric_key_prefix="probe_clean")
    generation_seconds = round(time.perf_counter() - gen0, 1)
    peak_generation = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    assert_capture_identities(diag_captured, clean_captured, design)

    trainer.model.train()
    _enable_gradient_checkpointing(trainer.model)
    trainer.model.config.use_cache = False
    coords, base_fingerprint, p = _establish_coordinates(trainer, clean_captured, step)
    fingerprint_before = lora_fingerprint(trainer.model, coords)
    t_ref = _t_ref(diag_captured, clean_captured)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    score0 = time.perf_counter()
    clean = _accumulate_outcome(spec, clean_captured, trainer, coords, p, t_ref)
    diag = _accumulate_transition(spec, design, step, diag_captured, trainer, coords, p, t_ref)
    scoring_seconds = round(time.perf_counter() - score0, 1)
    peak_scoring = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    loss_validation = _maybe_validate(spec, clean, trainer, coords)
    fingerprint_after = lora_fingerprint(trainer.model, coords)
    if fingerprint_after != fingerprint_before:
        raise AssertionError(f"LoRA weights changed during scoring: {fingerprint_before[:12]} -> {fingerprint_after[:12]}")
    trainer.model.eval()

    accumulators = {"step": step, "q": spec.q, "outcome": clean["accumulators"], "transition": diag["accumulators"]}
    summary = pm.checkpoint_summary(accumulators)
    provenance = _provenance(spec, adapter_sha, base_fingerprint, git_commit, design)
    payload = {
        "step": step, "trajectory": spec.trajectory, "adapter_path": str(adapter_path) if adapter_path else None,
        "provenance": provenance, "P": p, "coordinate_names_sha256": _names_sha(coords),
        "lora_fingerprint": fingerprint_after, "t_ref": t_ref, "summary": summary,
        "counts": {"clean_rollouts": clean["rollouts"], "clean_groups": clean["groups"],
                   "diagnostic_tasks": len(diag_tasks), "diagnostic_rollouts": diag["rollouts"],
                   "groups_per_design": diag["groups_per_design"], "rho_zero_fraction": clean["rho_zero_fraction"],
                   "rho_nonzero_min_mean_max": clean["rho_stats"]},
        "clean_task_table": clean["task_table"], "diagnostic_task_table": diag["task_table"],
        "loss_validation": loss_validation, "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds,
        "peak_mem_generation_gb": round(peak_generation / 1e9, 2), "peak_mem_scoring_gb": round(peak_scoring / 1e9, 2),
        "git_commit": git_commit, "deployed_commit": deployed_commit, "provider": provider,
    }
    evidence = {"provenance": provenance, "t_ref": t_ref, "P": p, "q": spec.q, "step": step,
                "coordinate_names": coords, "outcome": clean["evidence"], "transition": diag["evidence"]}
    atomic_write_gzip(step_dir.parent / f"step{step}_evidence.json.gz", evidence)
    atomic_write_json(step_dir.parent / f"step{step}.json", payload)
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return payload


def _establish_coordinates(trainer, clean_captured, step):
    """One backward on the first captured rollout fixes the coordinate set (LoRA params with a gradient), P, and the
    base fingerprint at step 0."""
    output, i, _ = next(iter(_by_task(clean_captured).values()))[0]
    _first_backward(trainer, output, i)
    coords = select_coordinates(_named_has_grad(trainer.model))
    params = dict(trainer.model.named_parameters())
    p = int(sum(params[name].numel() for name in coords))
    trainer.model.zero_grad(set_to_none=True)
    base_fingerprint = lora_fingerprint(trainer.model, coords) if step == 0 else None
    return coords, base_fingerprint, p


def _first_backward(trainer, output, i):
    import torch

    completion_mask = output["completion_mask"]
    tool_mask = output.get("tool_mask")
    mask_i = (completion_mask if tool_mask is None else completion_mask * tool_mask)[i:i + 1]
    input_ids = torch.cat([output["prompt_ids"][i:i + 1], output["completion_ids"][i:i + 1]], dim=1)
    attention_mask = torch.cat([output["prompt_mask"][i:i + 1], completion_mask[i:i + 1]], dim=1)
    trainer.model.zero_grad(set_to_none=True)
    logps, _, _ = trainer._get_per_token_logps_and_entropies(trainer.model, input_ids, attention_mask, output["completion_ids"].size(1))
    (logps * mask_i).sum().backward()


def _names_sha(coords) -> str:
    return hashlib.sha256("\n".join(coords).encode("utf-8")).hexdigest()


def _t_ref(diag_captured, clean_captured) -> float:
    """T_ref = TRAIN_GROUPS x TRAIN_GENERATIONS x mean masked tokens per rollout over the step's rollouts (TRL's
    per-step DAPO normalizer, one constant that cancels from paired-versus-independent ratios)."""
    totals = []
    for captured in (diag_captured, clean_captured):
        for output, episodes in captured:
            mask, _ = _mask_and_ratio(output)
            per = mask.sum(dim=1).float().tolist()
            totals.extend(per[:len(episodes)])
    mean_tokens = float(np.mean(totals)) if totals else 1.0
    return TRAIN_GROUPS * TRAIN_GENERATIONS * mean_tokens


def _score_task(trainer, members, coords, p):
    """Score a task's rollouts one at a time into a CPU matrix (n x P), returning the fp64 Gram, the fp32 score
    matrix, token counts and rho."""
    import torch

    n = len(members)
    scores = torch.empty((n, p), dtype=torch.float32, pin_memory=torch.cuda.is_available())
    tokens, rho = np.empty(n), np.empty(n)
    for row, (output, i, _) in enumerate(members):
        vec, tok, r = score_rollout(trainer, output, i, coords)
        scores[row].copy_(vec)
        tokens[row], rho[row] = tok, r
        del vec
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gram = gram_chunked(scores, _chunk_rows(p))
    return gram, scores, tokens, rho


def _accumulate_outcome(spec, captured, trainer, coords, p, t_ref):
    tasks = _by_task(captured)
    designs = pm.OUTCOME_DESIGNS
    sum_sq = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    within = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    running = {est: {d: np.zeros(p) for d in designs} for est in pm.ESTIMATORS}
    contrast = {d: 0.0 for d in designs}
    by_k = defaultdict(lambda: {"count": 0, "lhs_sum": 0.0, "rhs_sum": 0.0, "ratio_sum": 0.0})
    lhs_sum = rhs_sum = success = 0.0
    n_groups = rollouts = 0
    rho_zero = rho_all = 0
    rho_nonzero = []
    task_table, evidence = [], []
    validation = None
    for tid, members in tasks.items():
        gram, scores, tokens, rho = _score_task(trainer, members, coords, p)
        true = np.array([float(e["true_success"]) for _, _, e in members])
        res = pm.outcome_noise_group(gram, true, rho, t_ref, spec.q)
        if validation is None and len(set(true.tolist())) >= 2 and len({id(o) for o, _, _ in members}) == 1:
            validation = {"output": members[0][0], "indices": [j for _, j, _ in members],
                          "scores": scores.clone() if hasattr(scores, "clone") else np.array(scores),
                          "rho": rho.copy(), "true": true.copy()}
        lhs_sum += res["lhs"]
        rhs_sum += res["rhs"]
        n_groups += 1
        rollouts += len(members)
        success += float(true.sum())
        rho_all += len(rho)
        rho_zero += int(np.count_nonzero(rho == 0.0))
        rho_nonzero.extend(float(x) for x in rho[rho != 0.0])
        k = res["k_success"]
        ratio = res["lhs"] / res["rhs"] if res["rhs"] > 0 else 0.0
        bucket = by_k[k]
        bucket["count"] += 1
        bucket["lhs_sum"] += res["lhs"]
        bucket["rhs_sum"] += res["rhs"]
        bucket["ratio_sum"] += ratio
        task_table.append({"task_id": tid, "lhs": res["lhs"], "rhs": res["rhs"], "k_success": k})
        for est in pm.ESTIMATORS:
            for d in designs:
                blk = res[est][d]
                ea = np.asarray(blk["expected_weights"])
                sum_sq[est][d] += blk["expected_sq_norm"]
                within[est][d] += blk["expected_sq_norm"] - pm.quadratic(ea, gram)
                running[est][d] += weighted_sum_chunked(ea, scores, _chunk_rows(p))
                if est == "mean_centered":
                    contrast[d] += blk["expected_contrast_variance"]
        evidence.append({"task_id": tid, "gram": gram.tolist(), "true": true.tolist(), "tokens": tokens.tolist(),
                         "rho": rho.tolist(), "identities": [_identity(e) for _, _, e in members]})
        del scores
    accumulators = _finish_blocks(sum_sq, within, running, {d: contrast[d] / max(n_groups, 1) for d in designs},
                                  n_groups, n_groups, designs, "independent")
    accumulators.update({"n_groups": n_groups, "lhs_sum": lhs_sum, "rhs_sum": rhs_sum, "by_k": dict(by_k)})
    stats = (min(rho_nonzero), float(np.mean(rho_nonzero)), max(rho_nonzero)) if rho_nonzero else (0.0, 0.0, 0.0)
    return {"accumulators": accumulators, "rollouts": rollouts, "groups": n_groups, "task_table": task_table,
            "evidence": {"groups": evidence, "running": _running_norms(accumulators, designs)},
            "success_rate": success / max(rollouts, 1), "validation": validation,
            "rho_zero_fraction": rho_zero / max(rho_all, 1), "rho_stats": list(stats)}


def _accumulate_transition(spec, design, step, captured, trainer, coords, p, t_ref):
    tasks = _by_task(captured)
    designs = pm.TRANSITION_DESIGNS
    rng = np.random.default_rng(spec.seed * 1000 + step)
    sum_sq = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    within = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    running = {est: {d: np.zeros(p) for d in designs} for est in pm.ESTIMATORS}
    reward_var = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    n_groups = {est: {d: 0 for d in designs} for est in pm.ESTIMATORS}
    n_tasks = 0
    task_table, evidence = [], []
    rollouts = 0
    k, m = design["schedules"], design["samples"]
    for tid, unordered in tasks.items():
        members = order_transition_members(unordered)
        schedule_order = [int(episode["schedule_seed"]) for _, _, episode in members]
        assert_schedule_layout(schedule_order, k, m, tid)
        gram, scores, tokens, rho = _score_task(trainer, members, coords, p)
        rewards = np.array([float(e["true_success"]) for _, _, e in members])
        index_lists = pm.transition_designs(k, m, design["resamples"], rng)
        rollouts += len(members)
        n_tasks += 1
        task_row = {"task_id": tid}
        for d in designs:
            terms = pm.group_terms(gram, rewards, rho, t_ref, index_lists[d])
            for est in pm.ESTIMATORS:
                blk = terms[est]
                sum_sq[est][d] += blk["sum_sq_norm"]
                within[est][d] += within_task_trace_transition(blk["sum_sq_norm"], blk["summed_weights"], gram, blk["n_groups"])
                reward_var[est][d] += 2.0 * blk["reward_var_sum"] / max(blk["n_groups"], 1)
                n_groups[est][d] += blk["n_groups"]
                running[est][d] += weighted_sum_chunked(blk["summed_weights"], scores, _chunk_rows(p))
            task_row[d] = {"sum_sq_norm": terms["mean_centered"]["sum_sq_norm"]}
        task_table.append(task_row)
        evidence.append({"task_id": tid, "gram": gram.tolist(), "rewards": rewards.tolist(), "tokens": tokens.tolist(),
                         "rho": rho.tolist(), "index_lists": index_lists, "schedule_order": schedule_order,
                         "identities": [_identity(e) for _, _, e in members]})
        del scores
    accumulators = {}
    for est in pm.ESTIMATORS:
        block = {"cross_vec": float(running[est]["paired"] @ running[est]["independent_resampled"])}
        for d in designs:
            ng = n_groups[est][d]
            block[d] = {"sum_sq_norm": sum_sq[est][d], "vec_sq_norm": float(running[est][d] @ running[est][d]),
                        "n_groups": ng, "n_within": n_tasks, "within_sum": within[est][d],
                        "reward_variance": reward_var[est][d] / max(n_tasks, 1)}
        accumulators[est] = block
    groups_per_design = {d: n_groups["mean_centered"][d] for d in designs}
    return {"accumulators": accumulators, "rollouts": rollouts, "groups_per_design": groups_per_design,
            "task_table": task_table, "evidence": {"tasks": evidence, "running": _running_norms(accumulators, designs)}}


def _finish_blocks(sum_sq, within, running, reward_variance, n_groups, n_within, designs, primary_independent) -> dict:
    out = {}
    for est in pm.ESTIMATORS:
        block = {"cross_vec": float(running[est]["paired"] @ running[est][primary_independent])}
        for d in designs:
            block[d] = {"sum_sq_norm": sum_sq[est][d], "vec_sq_norm": float(running[est][d] @ running[est][d]),
                        "n_groups": n_groups, "n_within": n_within, "within_sum": within[est][d],
                        "reward_variance": reward_variance[d]}
        out[est] = block
    return out


def _running_norms(accumulators, designs) -> dict:
    """The GPU-computed running-vector norms and cross terms (vec_sq_norm, cross_vec) per estimator and design for the
    evidence file; recompute reads them because per-task Grams cannot rebuild the cross-task inner products."""
    return {est: {**{d: accumulators[est][d]["vec_sq_norm"] for d in designs}, "cross_vec": accumulators[est]["cross_vec"]}
            for est in pm.ESTIMATORS}


def _identity(episode) -> dict:
    return {k: episode.get(k) for k in ("task_id", "schedule_seed", "resolved_seed", "slot", "episode_index")}


def _maybe_validate(spec, clean, trainer, coords):
    """Run the loss-gradient validation on the first mixed clean group when smoke or --validate-loss is on."""
    if not (spec.validate_loss or spec.smoke):
        return None
    validation = clean.get("validation")
    return validate_loss_gradient(trainer, validation, coords) if validation is not None else None


# ---- run_probe: one process per step ----------------------------------------

def _default_step_runner(spec, step, runs_root, data_dir, out_dir, git_commit, deployed_commit, provider, usd_per_hour):
    cmd = [sys.executable, "-m", "pairedrl.train.probe", "--spec-json", json.dumps(spec.to_dict()), "--step", str(step),
           "--runs-root", str(runs_root), "--data-dir", str(data_dir), "--out-dir", str(out_dir),
           "--git-commit", git_commit, "--deployed-commit", deployed_commit, "--provider", provider,
           "--usd-per-hour", str(usd_per_hour)]
    return subprocess.run(cmd, check=False).returncode


def run_probe(spec: ProbeSpec, runs_root, data_dir, out_dir, git_commit: str, deployed_commit: str, provider: str,
              usd_per_hour: float, commit_fn=None, reraise_attempts: int = 1, step_runner=None) -> dict:
    """Probe every step of one trajectory, one child process per step (memory is released between checkpoints).
    Refuses on a provenance mismatch. Resumes a step only if its recorded provenance matches. A child's non-zero
    exit or a missing/mismatched step file marks the manifest FAILED and raises for attempts at most
    `reraise_attempts` (so Modal's retry re-enters and resumes the finished steps)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = step_runner or (lambda step: _default_step_runner(
        spec, step, runs_root, data_dir, out_dir, git_commit, deployed_commit, provider, usd_per_hour))
    design = spec.design()
    attempt = _attempt_number(out_dir)
    t0 = time.perf_counter()
    versions = {p: package_version(p) for p in VERSION_PACKAGES}
    provenance_ok = git_commit == deployed_commit and not git_commit.endswith("-dirty")

    def manifest(status, steps_done, episodes, cost, extra=None):
        m = {"run_id": spec.probe_id, "probe_id": spec.probe_id, "trajectory": spec.trajectory, "spec": spec.to_dict(),
             "steps": spec.steps, "steps_done": steps_done, "episodes_logged": episodes, "status": status,
             "git_commit": git_commit, "deployed_commit": deployed_commit, "provider": provider, "attempt": attempt,
             "reraise_attempts": reraise_attempts, "usd_per_hour": usd_per_hour, "versions": versions,
             "hostname": socket.gethostname(), "progress_utc": utc_now(), "wall_time_s": round(time.perf_counter() - t0, 1),
             "estimated_cost_usd": round(cost, 2)}
        if extra:
            m.update(extra)
        atomic_write_json(out_dir / "run_manifest.json", m)
        if commit_fn is not None:
            commit_fn()
        return m

    if not provenance_ok:
        reason = f"provenance: launched with commit {git_commit!r} but the code is at {deployed_commit!r}"
        return {"manifest": manifest("REFUSED", [], 0, 0.0, {"error": reason}), "summary": {}}

    steps_done, episodes, cost = [], 0, 0.0
    manifest("RUNNING", steps_done, episodes, cost)
    for step in spec.steps:
        step_file = out_dir / f"step{step}.json"
        if _resume_ok(step_file, spec, step, runs_root, git_commit, design):
            steps_done.append(step)
            episodes, cost = _reduce_steps(out_dir, steps_done, usd_per_hour)
            manifest("RUNNING", steps_done, episodes, cost)
            continue
        code = runner(step)
        if code != 0 or not step_file.exists():
            error = f"step {step} child exit {code}, step file present {step_file.exists()}"
            manifest("FAILED", steps_done, episodes, cost, {"error": error})
            if attempt <= reraise_attempts:
                raise RuntimeError(error)
            return {"manifest": manifest("FAILED", steps_done, episodes, cost, {"error": error}), "summary": {}}
        steps_done.append(step)
        episodes, cost = _reduce_steps(out_dir, steps_done, usd_per_hour)
        manifest("RUNNING", steps_done, episodes, cost)
    return {"manifest": manifest("COMPLETE", steps_done, episodes, cost), "summary": {"steps": steps_done}}


def _resume_ok(step_file, spec, step, runs_root, git_commit, design) -> bool:
    if not pathlib.Path(step_file).exists():
        return False
    payload = json.loads(pathlib.Path(step_file).read_text(encoding="utf-8"))
    adapter_path = None if step == 0 else pathlib.Path(runs_root) / spec.trajectory / "trainer" / f"checkpoint-{step}"
    adapter_sha = _sha256_file(adapter_path / "adapter_model.safetensors") if adapter_path else None
    base_fp = payload.get("provenance", {}).get("base_fingerprint")
    ok, reason = resume_matches(payload, _provenance(spec, adapter_sha, base_fp, git_commit, design))
    if not ok:
        raise SystemExit(f"probe {spec.probe_id} step {step}: existing step file does not match ({reason}); a repaired rerun needs a new probe id")
    return True


def _reduce_steps(out_dir, steps_done, usd_per_hour):
    episodes = cost = 0.0
    for step in steps_done:
        payload = json.loads((pathlib.Path(out_dir) / f"step{step}.json").read_text(encoding="utf-8"))
        counts = payload.get("counts", {})
        episodes += counts.get("clean_rollouts", 0) + counts.get("diagnostic_rollouts", 0)
        cost += (payload.get("generation_seconds", 0) + payload.get("scoring_seconds", 0)) / 3600 * usd_per_hour
    return int(episodes), cost


def _attempt_number(out_dir) -> int:
    marker = pathlib.Path(out_dir) / "attempts.txt"
    n = int(marker.read_text().strip()) + 1 if marker.exists() else 1
    marker.write_text(str(n), encoding="utf-8")
    return n


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--deployed-commit", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--usd-per-hour", type=float, default=0.0)
    args = parser.parse_args()
    spec = ProbeSpec.from_dict(json.loads(args.spec_json))
    step_file = pathlib.Path(args.out_dir) / f"step{args.step}.json"
    if step_file.exists():
        return 0
    _probe_step(spec, args.step, args.runs_root, args.data_dir, args.out_dir, args.git_commit, args.deployed_commit, args.provider)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

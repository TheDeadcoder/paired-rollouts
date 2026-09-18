"""The gradient probe GPU job (H1(b), amendment A1): ProbeSpec, the Modal-free ledger-row helper, and run_probe.
Torch and TRL are imported lazily inside run_probe and the scoring helpers so this module and ProbeSpec import on
any machine. The arithmetic lives in pairedrl.analysis.probe; here we generate rollouts with the trainer's own vLLM
path at a frozen checkpoint, score each rollout by one backward of (per_token_logps * mask).sum() over the LoRA
parameters, and accumulate the Gram-based terms."""

import dataclasses
import json
import pathlib
import socket
import time
from dataclasses import dataclass

import numpy as np

from pairedrl.analysis import probe as pm
from pairedrl.ops.job import ADAPTER_CONFIG, VERSION_PACKAGES, package_version, utc_now

SMOKE_DESIGN = {"diagnostic_tasks": 2, "schedules": 2, "samples": 2, "clean_tasks": 4, "clean_rollouts": 2, "resamples": 4}


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
        """The effective design; the smoke flag shrinks it to a ten-minute end-to-end test."""
        base = {"diagnostic_tasks": self.diagnostic_tasks, "schedules": self.schedules, "samples": self.samples,
                "clean_tasks": self.clean_tasks, "clean_rollouts": self.clean_rollouts, "resamples": self.resamples}
        return dict(SMOKE_DESIGN) if self.smoke else base


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


def _probe_run_spec(spec: ProbeSpec, design: dict, adapter_path, condition: str, arm: str):
    from pairedrl.train.runner import RunSpec

    return RunSpec(
        run_id=f"{spec.probe_id}-step",
        model=spec.model,
        condition=condition,
        arm=arm,
        p=spec.p,
        q=0.0,
        seed=spec.seed,
        eval_only=True,
        adapter_path=str(adapter_path) if adapter_path else None,
        diagnostic_schedules=design["schedules"],
        diagnostic_samples=design["samples"],
        fault_weights=spec.fault_weights,
        max_completion_length=spec.max_completion_length,
        max_tool_calling_iterations=spec.max_tool_calling_iterations,
        vllm_max_model_length=spec.vllm_max_model_length,
        vllm_gpu_memory_utilization=spec.vllm_gpu_memory_utilization,
        per_device_eval_batch_size=spec.per_device_eval_batch_size,
        logprob_chunk=spec.logprob_chunk,
    )


def _capture_generation(trainer) -> list:
    """Wrap the instance's _generate_and_score_completions so each eval batch's generation output and the matching
    environments' last_episode records (in input order, the training_group_records pattern) are captured for
    scoring. PhasedTrainer only logs group records while training, so an instance wrapper is used here rather than a
    subclass, to reuse build_trainer unchanged (recorded in docs/PROBE.md)."""
    captured: list = []
    inner = trainer._generate_and_score_completions

    def wrapper(inputs):
        output = inner(inputs)
        n = len(inputs)
        episodes = [env.last_episode for env in trainer.environments[:n]]
        captured.append((output, episodes))
        return output

    trainer._generate_and_score_completions = wrapper
    return captured


def _lora_parameters(model):
    return [p for _, p in sorted(model.named_parameters(), key=lambda kv: kv[0]) if p.requires_grad]


def _score_batch(trainer, output, lora_params):
    """One backward per rollout of (per_token_logps * mask).sum() over the LoRA parameters, reproducing the loss's
    forward (input_ids = cat(prompt, completion), mask = completion_mask * tool_mask). Returns the fp32 score
    matrix (rollouts x P) on the GPU and the per-rollout token counts T_i = mask.sum()."""
    import torch

    prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
    completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
    tool_mask = output.get("tool_mask")
    mask_full = completion_mask if tool_mask is None else completion_mask * tool_mask
    logits_to_keep = completion_ids.size(1)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    vectors = []
    tokens = []
    for i in range(input_ids.size(0)):
        trainer.model.zero_grad(set_to_none=True)
        logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids[i:i + 1], attention_mask[i:i + 1], logits_to_keep)
        mask_i = mask_full[i:i + 1]
        (logps * mask_i).sum().backward()
        grad = torch.cat([p.grad.reshape(-1) for p in lora_params]).float().detach()
        vectors.append(grad)
        tokens.append(float(mask_i.sum().item()))
    trainer.model.zero_grad(set_to_none=True)
    return torch.stack(vectors), tokens


def _episode_reward(episode: dict) -> float:
    return float(episode["true_success"])


def run_probe(spec: ProbeSpec, runs_root, data_dir, out_dir, git_commit: str, deployed_commit: str, provider: str,
              usd_per_hour: float, commit_fn=None) -> dict:
    """Probe every requested step of one trajectory. Refuses on a provenance mismatch (as run_job does). Writes a
    run_manifest.json that scripts/collect_modal.py prints unchanged and one step<N>.json per checkpoint (resumable:
    an existing step file is skipped). The base model is step 0 (adapter_path None); step N is the trajectory's
    trainer/checkpoint-N, a hard error when absent."""
    import torch

    from pairedrl.train.env_adapter import BackOfficeEnv

    runs_root = pathlib.Path(runs_root)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    t0 = time.perf_counter()
    versions = {p: package_version(p) for p in VERSION_PACKAGES}

    def write_manifest(status, steps_done, episodes, extra=None):
        manifest = {
            "run_id": spec.probe_id, "probe_id": spec.probe_id, "trajectory": spec.trajectory,
            "spec": spec.to_dict(), "steps": spec.steps, "steps_done": steps_done, "episodes_logged": episodes,
            "status": status, "git_commit": git_commit, "deployed_commit": deployed_commit, "provider": provider,
            "usd_per_hour": usd_per_hour, "versions": versions, "hostname": socket.gethostname(),
            "progress_utc": utc_now(), "wall_time_s": round(time.perf_counter() - t0, 1),
            "estimated_cost_usd": round((time.perf_counter() - t0) / 3600 * usd_per_hour, 2),
        }
        if extra:
            manifest.update(extra)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if commit_fn is not None:
            commit_fn()
        return manifest

    if git_commit != deployed_commit or git_commit.endswith("-dirty"):
        reason = f"provenance: launched with commit {git_commit!r} but the code is at {deployed_commit!r}"
        return {"manifest": write_manifest("REFUSED", [], 0, {"error": reason}), "summary": {}}

    design = spec.design()
    condition, arm = "C2", "paired"
    write_manifest("RUNNING", [], 0)
    steps_done: list[int] = []
    episodes_total = 0
    try:
        for step in spec.steps:
            step_file = out_dir / f"step{step}.json"
            if step_file.exists():
                steps_done.append(step)
                write_manifest("RUNNING", steps_done, episodes_total)
                continue
            adapter_path = None
            if step != 0:
                adapter_path = runs_root / spec.trajectory / "trainer" / f"checkpoint-{step}"
                if not (adapter_path / ADAPTER_CONFIG).exists():
                    raise FileNotFoundError(f"probe {spec.probe_id}: checkpoint absent at {adapter_path}")
            episodes_total += _probe_step(spec, design, step, adapter_path, condition, arm, runs_root, data_dir,
                                          out_dir, git_commit, deployed_commit, provider, BackOfficeEnv, torch)
            steps_done.append(step)
            write_manifest("RUNNING", steps_done, episodes_total)
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        import traceback

        return {"manifest": write_manifest("FAILED", steps_done, episodes_total,
                                           {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-8000:]}),
                "summary": {}}
    return {"manifest": write_manifest(status, steps_done, episodes_total), "summary": {"steps": steps_done}}


def _probe_step(spec, design, step, adapter_path, condition, arm, runs_root, data_dir, out_dir, git_commit,
                deployed_commit, provider, back_office_env, torch) -> int:
    """Generate and score one checkpoint's rollouts, accumulate the outcome-noise (clean groups) and transition-noise
    (diagnostic tasks) terms, and write step<N>.json. Returns the episode count of the step."""
    from datasets import Dataset

    from pairedrl.noise.config import NoiseConfig
    from pairedrl.train.dataset import build_eval_rows
    from pairedrl.train.runner import assemble_diagnostic_rows, build_trainer, load_task_splits

    step_dir = out_dir / f"step{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    run_spec = _probe_run_spec(spec, design, adapter_path, condition, arm)
    splits = load_task_splits(data_dir)
    trainer, _, _ = build_trainer(run_spec, data_dir, step_dir / "trainer", step_dir / "episodes.jsonl")
    lora_params = _lora_parameters(trainer.model)
    lora_param_count = int(sum(p.numel() for p in lora_params))

    diagnostic_tasks = splits["diagnostic"][:design["diagnostic_tasks"]]
    diag_rows = assemble_diagnostic_rows(run_spec, diagnostic_tasks)
    clean_rows = build_eval_rows(splits["heldout"][:design["clean_tasks"]], NoiseConfig.clean(), "probe:clean",
                                 list(range(design["clean_rollouts"])))

    gen0 = time.perf_counter()
    diag_captured = _capture_generation(trainer)
    back_office_env.phase = "probe:diag"
    trainer.evaluate(eval_dataset=Dataset.from_list(diag_rows), metric_key_prefix="probe_diag")
    clean_captured = _capture_generation(trainer)
    back_office_env.phase = "probe:clean"
    trainer.evaluate(eval_dataset=Dataset.from_list(clean_rows), metric_key_prefix="probe_clean")
    generation_seconds = round(time.perf_counter() - gen0, 1)

    score0 = time.perf_counter()
    diag = _accumulate_transition(spec, design, step, diag_captured, trainer, lora_params, torch)
    clean = _accumulate_outcome(spec, clean_captured, trainer, lora_params, torch)
    scoring_seconds = round(time.perf_counter() - score0, 1)

    accumulators = {"step": step, "q": spec.q, "outcome": clean["accumulators"], "transition": diag["accumulators"]}
    summary = pm.checkpoint_summary(accumulators)
    payload = {
        "step": step, "adapter_path": str(adapter_path) if adapter_path else None, "summary": summary,
        "counts": {"clean_rollouts": clean["rollouts"], "clean_groups": clean["groups"],
                   "diagnostic_tasks": len(diagnostic_tasks), "diagnostic_rollouts": diag["rollouts"],
                   "groups_per_design": diag["groups_per_design"]},
        "clean_task_table": clean["task_table"], "diagnostic_task_table": diag["task_table"],
        "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds,
        "lora_parameter_count": lora_param_count, "success_rates": {"clean": clean["success_rate"], "diagnostic": diag["success_rate"]},
        "git_commit": git_commit, "deployed_commit": deployed_commit, "provider": provider,
    }
    (out_dir / f"step{step}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return clean["rollouts"] + diag["rollouts"]


def _by_task(captured, key):
    """Group captured (output, episodes) rollouts by task id, preserving order; `key` maps an episode to its task."""
    from collections import OrderedDict

    tasks = OrderedDict()
    for output, episodes in captured:
        for i, episode in enumerate(episodes):
            tasks.setdefault(key(episode), []).append((output, i, episode))
    return tasks


def _zero_vectors(dim, designs, torch):
    return {est: {d: torch.zeros(dim, dtype=torch.float64) for d in designs} for est in pm.ESTIMATORS}


def _accumulate_outcome(spec, captured, trainer, lora_params, torch):
    """Outcome noise over the clean groups: one group per task (its clean_rollouts rollouts). Accumulate per
    estimator and design E||g||^2, the running expected-gradient vector and the contrast variance."""
    tasks = _by_task(captured, lambda e: e["task_id"])
    designs = pm.OUTCOME_DESIGNS
    sum_sq = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    contrast = {d: 0.0 for d in designs}
    running = None
    lhs_sum = rhs_sum = 0.0
    task_table = []
    n_groups = rollouts = 0
    success = 0.0
    batch_scores = {}
    for output, _ in captured:
        mat, toks = _score_batch(trainer, output, lora_params)
        batch_scores[id(output)] = (mat, toks)
    for task_id, members in tasks.items():
        scores = torch.stack([batch_scores[id(o)][0][i] for o, i, _ in members]).double()
        tokens = np.array([batch_scores[id(o)][1][i] for o, i, _ in members], dtype=float)
        true_success = np.array([_episode_reward(e) for _, _, e in members], dtype=float)
        success += float(true_success.sum())
        rollouts += len(members)
        gram = (scores @ scores.T).cpu().numpy()
        res = pm.outcome_noise_group(gram, true_success, tokens, spec.q)
        lhs_sum += res["lhs"]
        rhs_sum += res["rhs"]
        n_groups += 1
        task_table.append({"task_id": task_id, "lhs": res["lhs"], "rhs": res["rhs"], "k_success": res["k_success"]})
        if running is None:
            running = _zero_vectors(scores.shape[1], designs, torch)
        for est in pm.ESTIMATORS:
            for d in designs:
                blk = res[est][d]
                sum_sq[est][d] += blk["expected_sq_norm"]
                contrast_weights = torch.tensor(blk["expected_weights"], dtype=torch.float64, device=scores.device)
                running[est][d] += (contrast_weights @ scores).cpu()
                if est == "mean_centered":
                    contrast[d] += blk["expected_contrast_variance"]
    accumulators = _finish(sum_sq, running, {d: contrast[d] / max(n_groups, 1) for d in designs}, n_groups, designs, "independent", torch)
    accumulators["n_groups"] = n_groups
    accumulators["lhs_sum"] = lhs_sum
    accumulators["rhs_sum"] = rhs_sum
    return {"accumulators": accumulators, "rollouts": rollouts, "groups": n_groups,
            "task_table": task_table, "success_rate": success / max(rollouts, 1)}


def _accumulate_transition(spec, design, step, captured, trainer, lora_params, torch):
    """Transition noise over the diagnostic tasks: K*M rollouts per task grouped under the three designs. Accumulate
    per estimator and design the sum of ||g||^2, the running gradient vector and the within-group reward variance."""
    tasks = _by_task(captured, lambda e: e["task_id"])
    designs = pm.TRANSITION_DESIGNS
    rng = np.random.default_rng(spec.seed * 1000 + step)
    sum_sq = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    reward_var_sum = {est: {d: 0.0 for d in designs} for est in pm.ESTIMATORS}
    n_groups = {est: {d: 0 for d in designs} for est in pm.ESTIMATORS}
    running = None
    task_table = []
    rollouts = 0
    success = 0.0
    k, m = design["schedules"], design["samples"]
    batch_scores = {}
    for output, _ in captured:
        mat, toks = _score_batch(trainer, output, lora_params)
        batch_scores[id(output)] = (mat, toks)
    for task_id, members in tasks.items():
        scores = torch.stack([batch_scores[id(o)][0][i] for o, i, _ in members]).double()
        tokens = np.array([batch_scores[id(o)][1][i] for o, i, _ in members], dtype=float)
        rewards = np.array([_episode_reward(e) for _, _, e in members], dtype=float)
        success += float(rewards.sum())
        rollouts += len(members)
        gram = (scores @ scores.T).cpu().numpy()
        index_lists = pm.transition_designs(k, m, design["resamples"], rng)
        if running is None:
            running = _zero_vectors(scores.shape[1], designs, torch)
        task_row = {"task_id": task_id}
        for d in designs:
            terms = pm.group_terms(gram, rewards, tokens, index_lists[d])
            for est in pm.ESTIMATORS:
                blk = terms[est]
                sum_sq[est][d] += blk["sum_sq_norm"]
                reward_var_sum[est][d] += blk["reward_var_sum"]
                n_groups[est][d] += blk["n_groups"]
                weights = torch.tensor(blk["summed_weights"], dtype=torch.float64, device=scores.device)
                running[est][d] += (weights @ scores).cpu()
            task_row[d] = {"trace_var_paired_terms": terms["mean_centered"]["sum_sq_norm"]}
        task_table.append(task_row)
    accumulators = {}
    for est in pm.ESTIMATORS:
        block = {"cross_vec": float(running[est]["paired"] @ running[est]["independent_resampled"])}
        for d in designs:
            n = n_groups[est][d]
            block[d] = {"sum_sq_norm": sum_sq[est][d], "vec_sq_norm": float(running[est][d] @ running[est][d]),
                        "n_groups": n, "reward_variance": 2.0 * reward_var_sum[est][d] / max(n, 1)}
        accumulators[est] = block
    groups_per_design = {d: n_groups["mean_centered"][d] for d in designs}
    return {"accumulators": accumulators, "rollouts": rollouts, "groups_per_design": groups_per_design,
            "task_table": task_table, "success_rate": success / max(rollouts, 1)}


def _finish(sum_sq, running, reward_variance, n_groups, designs, primary_independent, torch):
    """Assemble one noise type's per-estimator accumulator block (sum_sq_norm, vec_sq_norm, n_groups, reward
    variance, and the paired/primary-independent cross term for the cosine)."""
    out = {}
    for est in pm.ESTIMATORS:
        block = {"cross_vec": float(running[est]["paired"] @ running[est][primary_independent]) if running else 0.0}
        for d in designs:
            vec_sq = float(running[est][d] @ running[est][d]) if running else 0.0
            block[d] = {"sum_sq_norm": sum_sq[est][d], "vec_sq_norm": vec_sq, "n_groups": n_groups,
                        "reward_variance": reward_variance[d]}
        out[est] = block
    return out

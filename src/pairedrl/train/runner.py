"""Run specification, dataset assembly, trainer construction and evaluation orchestration."""

import dataclasses
import hashlib
import json
import pathlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pairedrl.env.backoffice.tasks import Task, read_jsonl
from pairedrl.noise.config import DEFAULT_WEIGHTS, TRANSITION_TYPES, NoiseConfig
from pairedrl.train.dataset import (
    build_blocking_rows,
    build_diagnostic_rows,
    build_eval_rows,
    build_training_rows,
)

ARMS = ("paired", "independent", "clean", "blocking", "blocking_paired")
EVAL_SCHEDULES_PERIODIC = [1, 2]
EVAL_SCHEDULES_FINAL = [1, 2, 3, 4]
EVAL_SCHEDULES_CHALLENGE = [1]
FINAL_TRANSITION_LEVELS = (0.10, 0.25)


@dataclass
class RunSpec:
    run_id: str
    model: str
    condition: str
    arm: str
    p: float = 0.0
    q: float = 0.0
    seed: int = 0
    steps: int = 100
    prompts_per_step: int = 24
    num_generations: int = 8
    lora_r: int = 32
    learning_rate: float = 1e-5
    max_completion_length: int = 6144
    max_tool_calling_iterations: int = 24
    vllm_max_model_length: int = 12288
    fault_weights: dict[str, float] | None = None
    scale_rewards: str = "group"
    loss_type: str = "dapo"
    eval_every: int = 20
    checkpoint_steps: int = 20
    eval_tasks: int = 100
    final_eval_tasks: int = 200
    diagnostic_steps: list[int] = field(default_factory=lambda: [0, 50, 100])
    diagnostic_schedules: int = 8
    diagnostic_samples: int = 8
    train_only: bool = False
    eval_only: bool = False
    adapter_path: str | None = None
    vllm_gpu_memory_utilization: float = 0.35
    vllm_enable_prefix_caching: bool | None = None
    vllm_max_num_batched_tokens: int | None = None
    per_device_eval_batch_size: int = 128
    micro_batch: int = 2
    logprob_chunk: int = 1
    extend_previous: bool = False
    relaunch_from_commit: str | None = None
    notes: str = ""

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.arm == "clean" and (self.p > 0 or self.q > 0):
            raise ValueError("the clean arm has p = q = 0")
        if self.arm != "clean" and self.p == 0 and self.q == 0:
            raise ValueError("a noisy arm needs p > 0 or q > 0")
        if self.fault_weights is not None:
            unknown = set(self.fault_weights) - set(TRANSITION_TYPES)
            if unknown or any(w < 0 for w in self.fault_weights.values()) or sum(self.fault_weights.values()) <= 0:
                raise ValueError(f"fault_weights must be non-negative over {TRANSITION_TYPES} with a positive sum")
        if self.scale_rewards not in ("group", "none"):
            raise ValueError("scale_rewards must be 'group' or 'none'")
        if self.eval_only and self.train_only:
            raise ValueError("eval_only and train_only are mutually exclusive")
        if (self.prompts_per_step * self.num_generations) % self.micro_batch != 0:
            raise ValueError("prompts_per_step * num_generations must be divisible by micro_batch")
        if self.per_device_eval_batch_size % self.micro_batch != 0:
            raise ValueError("per_device_eval_batch_size must be divisible by micro_batch")
        if self.logprob_chunk < 1:
            raise ValueError("logprob_chunk must be at least 1")
        if self.vllm_max_num_batched_tokens is not None and self.vllm_max_num_batched_tokens < 1024:
            raise ValueError("vllm_max_num_batched_tokens must be at least 1024")
        if self.max_tool_calling_iterations < 2:
            raise ValueError("max_tool_calling_iterations must be at least 2")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunSpec":
        return cls(**data)

    def weights(self) -> dict[str, float]:
        """Training fault mixture: the calibrated default, or the run's own (for instance outage-free)."""
        return dict(self.fault_weights or DEFAULT_WEIGHTS)

    def training_noise(self) -> NoiseConfig:
        if self.arm == "clean":
            return NoiseConfig.clean()
        mode = {"blocking": "independent", "blocking_paired": "paired"}.get(self.arm, self.arm)
        return NoiseConfig(p=self.p, q=self.q, weights=self.weights(), mode=mode)

    def eval_noise(self) -> NoiseConfig:
        """Evaluation always pairs: fixed schedules shared by every arm and checkpoint."""
        if self.arm == "clean":
            return NoiseConfig.clean()
        return NoiseConfig(p=self.p, q=self.q, weights=self.weights(), mode="paired")


def assemble_training_rows(spec: RunSpec, tasks: list[Task]) -> list[dict]:
    n_rows = spec.steps * spec.prompts_per_step
    config = spec.training_noise()
    if spec.arm in ("blocking", "blocking_paired"):
        return build_blocking_rows(tasks, config, spec.condition, n_rows, spec.seed)
    return build_training_rows(tasks, config, spec.condition, n_rows, spec.seed)


def assemble_eval_sets(spec: RunSpec, tasks: list[Task], final: bool) -> dict[str, list[dict]]:
    """Periodic sets (validation tasks): clean, at-training-noise and the matched challenge (every write fails
    once; deterministic, so one schedule). Final sets (test tasks): clean, every transition level at the default
    mixture, the held-out fault types, the challenge (all identical for every arm and condition), plus the run's own
    training noise when it is not already one of those (outcome noise, outage-free mixtures)."""
    n = spec.final_eval_tasks if final else spec.eval_tasks
    indices = EVAL_SCHEDULES_FINAL if final else EVAL_SCHEDULES_PERIODIC
    tasks = tasks[:n]
    sets = {"clean": build_eval_rows(tasks, NoiseConfig.clean(), "eval:clean", indices)}
    challenge = build_eval_rows(tasks, NoiseConfig.challenge_writes(), "eval:challenge", EVAL_SCHEDULES_CHALLENGE)
    if not final:
        noise = spec.eval_noise() if spec.arm != "clean" else NoiseConfig.transition(0.25, "paired")
        sets["noisy"] = build_eval_rows(tasks, noise, "eval:noisy", indices)
        sets["challenge"] = challenge
        return sets
    for name, config in shared_final_configs(spec).items():
        sets[name] = build_eval_rows(tasks, config, f"eval:{name}", indices)
    sets["challenge"] = challenge
    return sets


def shared_final_configs(spec: RunSpec) -> dict[str, NoiseConfig]:
    """The final noisy sets every run is evaluated on, plus the run's own training noise when it is none of them."""
    shared = {f"noisy_p{round(level * 100):03d}": NoiseConfig.transition(level, "paired") for level in FINAL_TRANSITION_LEVELS}
    shared["heldout_types"] = NoiseConfig.heldout_types(0.25, "paired")
    own = spec.eval_noise()
    if spec.arm != "clean" and own not in shared.values():
        shared["at_training"] = own
    return shared


def expected_eval_sizes(spec: RunSpec) -> dict[str, dict[str, int]]:
    """Episode counts a complete run shows in every periodic set (per evaluated step) and every final set, from
    the spec alone; `assemble_eval_sets` on the real pools produces exactly these sizes."""
    per_task = len(EVAL_SCHEDULES_PERIODIC)
    periodic = {
        "clean": spec.eval_tasks * per_task,
        "noisy": spec.eval_tasks * per_task,
        "challenge": spec.eval_tasks * len(EVAL_SCHEDULES_CHALLENGE),
    }
    final = {"clean": spec.final_eval_tasks * len(EVAL_SCHEDULES_FINAL)}
    for name in shared_final_configs(spec):
        final[name] = spec.final_eval_tasks * len(EVAL_SCHEDULES_FINAL)
    final["challenge"] = spec.final_eval_tasks * len(EVAL_SCHEDULES_CHALLENGE)
    return {"periodic": periodic, "final": final}


def periodic_steps(spec: RunSpec) -> list[int]:
    """Steps at which the periodic sets are evaluated: 0, every `eval_every`, and the last step."""
    if spec.eval_only or spec.train_only:
        return []
    steps = {0, spec.steps} | set(range(spec.eval_every, spec.steps + 1, spec.eval_every))
    return sorted(steps)


def assemble_diagnostic_rows(spec: RunSpec, diagnostic: list[Task]) -> list[dict]:
    """K schedules x M samples per task; identical rows share a schedule seed, so M samples pair exactly."""
    p = spec.p if spec.p > 0 else 0.25
    config = NoiseConfig.transition(p, "paired", spec.weights())
    base = build_diagnostic_rows(diagnostic, config, "diag", spec.diagnostic_schedules, base_seed=spec.seed)
    return [dict(row) for row in base for _ in range(spec.diagnostic_samples)]


SPLITS = ("train", "heldout", "diagnostic", "test")


def load_task_splits(data_dir) -> dict[str, list[Task]]:
    """train: training tasks; heldout: validation (periodic curves, calibration); test: final evaluation only;
    diagnostic: the luck-share tasks."""
    data_dir = pathlib.Path(data_dir)
    return {name: read_jsonl(data_dir / f"{name}.jsonl") for name in SPLITS}


def task_pool_digests(data_dir) -> dict[str, dict]:
    """SHA-256 of every split file against data/tasks/MANIFEST.json: the pools a run trained and evaluated on
    are the frozen ones exactly when every `match` is true."""
    data_dir = pathlib.Path(data_dir)
    manifest_path = data_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"] if manifest_path.exists() else {}
    out = {}
    for name in SPLITS:
        path = data_dir / f"{name}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        expected = manifest.get(name, {}).get("sha256")
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0
        out[name] = {"count": count, "sha256": digest, "manifest_sha256": expected, "match": digest is not None and digest == expected}
    return out


def final_step(spec: RunSpec) -> int:
    """The trainer step stamped on the final evaluation phases: the last training step, or 0 for eval-only runs."""
    return 0 if spec.eval_only else spec.steps


def identity_counts(rows: list[dict]) -> Counter:
    """Multiset of (task_id, schedule_seed, condition) over rows or episode records."""
    return Counter((r["task_id"], int(r["schedule_seed"]), r.get("condition")) for r in rows)


def expected_evidence(spec: RunSpec, splits: dict[str, list[Task]]) -> dict:
    """The exact identities a complete run's logs must show, from the spec and the frozen pools: for every
    periodic set the (task, schedule, condition) multiset evaluated at each registered step, for every final set
    the multiset at the final step, for each diagnostic step the K schedules per diagnostic task with M records
    each, and for every training step its `prompts_per_step` groups in trainer order (the dataset is not shuffled
    and one generation batch is one optimizer step, so step s draws rows s * P to (s + 1) * P - 1)."""
    periodic = assemble_eval_sets(spec, splits["heldout"], final=False) if periodic_steps(spec) else {}
    final = assemble_eval_sets(spec, splits["test"], final=True) if not spec.train_only else {}
    diag_steps = [s for s in spec.diagnostic_steps if s == 0 or not spec.eval_only] if not spec.train_only else []
    diagnostic = assemble_diagnostic_rows(spec, splits["diagnostic"]) if diag_steps else []
    train = assemble_training_rows(spec, splits["train"]) if not spec.eval_only else []
    per_step = spec.prompts_per_step
    return {
        "periodic_steps": periodic_steps(spec),
        "periodic": {name: identity_counts(rows) for name, rows in periodic.items()},
        "final_step": final_step(spec),
        "final": {name: identity_counts(rows) for name, rows in final.items()},
        "diagnostic_steps": diag_steps,
        "diagnostic": identity_counts(diagnostic),
        "diagnostic_tasks": len(splits["diagnostic"]) if diag_steps else 0,
        "training": [
            [(r["task_id"], int(r["schedule_seed"]), r.get("condition")) for r in train[s * per_step:(s + 1) * per_step]]
            for s in range(spec.steps if train else 0)
        ],
    }


def summarize_episodes(records: list[dict]) -> dict:
    """Aggregate an episode log by (phase, condition): success, recovery success, calls, exposure."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r.get("phase", "train"), r.get("condition"))].append(r)
    out = {}
    for (phase, condition), recs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        n = len(recs)
        exposed = [r for r in recs if r.get("exposed")]
        out[f"{phase}|{condition}"] = {
            "episodes": n,
            "true_success": sum(r["true_success"] for r in recs) / n,
            "observed_reward": sum(r["observed_reward"] for r in recs) / n,
            "flipped_frac": sum(r["flipped"] for r in recs) / n,
            "exposed_frac": len(exposed) / n,
            "recovery_success": (sum(r["true_success"] for r in exposed) / len(exposed)) if exposed else None,
            "mean_calls": sum(r["calls"] for r in recs) / n,
            "budget_exceeded_frac": sum(r["budget_exceeded"] for r in recs) / n,
            "finished_frac": sum(r["finished"] for r in recs) / n,
        }
    return out


def luck_share_tables_by_phase(
    records: list[dict], field: str = "true_success", schedules: int | None = None, samples: int | None = None
) -> dict[str, list[list[list[float]]]]:
    """K x M reward tables per diagnostic task, separately for every diagnostic phase (checkpoint).

    Cells are never truncated or merged across checkpoints: when `schedules` and `samples` are given, every task
    in a phase must have exactly that many schedules and samples per schedule, otherwise a ValueError names the
    offending (phase, task)."""
    by_phase: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        if r.get("condition") == "diag":
            by_phase[r.get("phase", "diag")][r["task_id"]][r["schedule_seed"]].append(float(r[field]))
    out = {}
    for phase in sorted(by_phase):
        tables = []
        for task_id in sorted(by_phase[phase]):
            rows = [cell for _, cell in sorted(by_phase[phase][task_id].items())]
            if schedules is not None and len(rows) != schedules:
                raise ValueError(f"{phase}/{task_id}: {len(rows)} schedules, expected {schedules}")
            if samples is not None and any(len(cell) != samples for cell in rows):
                raise ValueError(f"{phase}/{task_id}: samples per schedule {[len(c) for c in rows]}, expected {samples}")
            if len(rows) >= 2 and min(len(cell) for cell in rows) >= 2:
                tables.append(rows)
        out[phase] = tables
    return out


def luck_share_tables(records: list[dict], field: str = "true_success", phase: str | None = None) -> list[list[list[float]]]:
    """Tables for one diagnostic phase (the only phase when `phase` is None; an error if several exist)."""
    by_phase = luck_share_tables_by_phase(records, field)
    if phase is None:
        if len(by_phase) > 1:
            raise ValueError(f"several diagnostic phases present, pass one of {sorted(by_phase)}")
        return next(iter(by_phase.values()), [])
    return by_phase.get(phase, [])


def training_group_records(
    step: int, attempt: int, rows: list[dict], episodes: list[dict], rewards, advantages, num_generations: int,
    scale: str = "group",
) -> list[dict]:
    """One record per generation group of the trainer's own batch: row i of the batch is served by environment i,
    whose episode record was written by get_reward. Rows, episodes, rewards and advantages must line up exactly;
    a mismatch between a row and its episode means the linkage is broken and raises rather than being logged."""
    from pairedrl.analysis.diagnostics import summarize_group

    n = len(rows)
    if not (len(episodes) == len(rewards) == len(advantages) == n) or n % num_generations != 0:
        raise ValueError(f"group register: {n} rows, {len(episodes)} episodes, {len(rewards)} rewards, "
                         f"{len(advantages)} advantages, group size {num_generations}")
    out = []
    for g in range(n // num_generations):
        idx = range(g * num_generations, (g + 1) * num_generations)
        members = [episodes[i] for i in idx]
        for i, e in zip(idx, members):
            if e is None or e["task_id"] != rows[i]["task_id"] or abs(e["observed_reward"] - float(rewards[i])) > 1e-9:
                raise ValueError(f"group register: row {i} ({rows[i]['task_id']}) does not match its episode")
        observed = [float(rewards[i]) for i in idx]
        true = [bool(e["true_success"]) for e in members]
        summary = summarize_group(observed, true, scale=scale).to_dict()
        out.append({
            "step": step, "attempt": attempt, "group": g, "task_id": rows[idx[0]]["task_id"],
            "condition": rows[idx[0]].get("condition"), "schedule_seed": int(rows[idx[0]].get("schedule_seed", 0)),
            "resolved_seeds": [e.get("resolved_seed") for e in members],
            "slots": [e.get("slot") for e in members], "episode_indices": [e.get("episode_index") for e in members],
            "true_success": true, "observed_reward": observed, "flipped": [bool(e.get("flipped")) for e in members],
            "exposed": [bool(e.get("exposed")) for e in members], "calls": [e.get("calls") for e in members],
            "advantages": [float(advantages[i]) for i in idx],
            "zero_variance": summary["zero_variance"], "spurious": summary["spurious"],
            "max_abs_advantage": max(abs(float(advantages[i])) for i in idx),
        })
    return out


def summarize_groups(records: list[dict]) -> dict:
    """Training-group register totals: groups, zero-variance groups, all-correct groups and the spurious rate
    among them (the quantity H1 compares with 1 - (1 - q)^G - q^G), per condition and overall."""
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_condition[str(r.get("condition"))].append(r)
    out = {}
    for condition, recs in sorted(by_condition.items()):
        all_correct = [r for r in recs if all(r["true_success"])]
        out[condition] = {
            "groups": len(recs),
            "steps": sorted({r["step"] for r in recs}),
            "zero_variance_frac": sum(r["zero_variance"] for r in recs) / len(recs),
            "all_correct_groups": len(all_correct),
            "spurious_among_all_correct": (
                sum(r["spurious"] for r in all_correct) / len(all_correct) if all_correct else None
            ),
            "mean_max_abs_advantage": sum(r["max_abs_advantage"] for r in recs) / len(recs),
        }
    return out


def vllm_engine_overrides(spec: RunSpec) -> dict:
    overrides = {}
    if spec.vllm_enable_prefix_caching is not None:
        overrides["enable_prefix_caching"] = spec.vllm_enable_prefix_caching
    if spec.vllm_max_num_batched_tokens is not None:
        overrides["max_num_batched_tokens"] = spec.vllm_max_num_batched_tokens
    return overrides


def patch_vllm_engine(overrides: dict) -> None:
    """TRL builds its colocated vLLM engine with fixed kwargs (no prefix caching for hybrid models,
    max_num_batched_tokens 4096); this wraps the LLM class it instantiates so run specs can override them."""
    import trl.generation.vllm_generation as vg

    base = getattr(vg, "_pairedrl_base_llm", None) or vg.LLM
    vg._pairedrl_base_llm = base
    if not overrides:
        vg.LLM = base
        return

    class PatchedLLM(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, **overrides})

    vg.LLM = PatchedLLM


def build_trainer(spec: RunSpec, data_dir, out_dir, log_path, group_log_path=None):
    """Construct the TRL trainer. Imports TRL lazily so the rest of the module stays importable on any machine."""
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.utils import create_model_from_path

    from pairedrl.train.env_adapter import BackOfficeEnv, make_env_factory

    patch_vllm_engine(vllm_engine_overrides(spec))
    splits = load_task_splits(data_dir)
    task_paths = [str(pathlib.Path(data_dir) / f"{n}.jsonl") for n in SPLITS]
    train_rows = assemble_training_rows(spec, splits["train"]) if not spec.eval_only else []
    periodic = assemble_eval_sets(spec, splits["heldout"], final=False)
    per_device = spec.prompts_per_step * spec.num_generations
    config = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=spec.steps if not spec.eval_only else 1,
        per_device_train_batch_size=spec.micro_batch,
        generation_batch_size=per_device,
        num_generations=spec.num_generations,
        num_generations_eval=1,
        per_device_eval_batch_size=spec.per_device_eval_batch_size,
        gradient_accumulation_steps=per_device // spec.micro_batch,
        gradient_checkpointing=True,
        learning_rate=spec.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no" if spec.eval_only else "steps",
        save_steps=spec.checkpoint_steps,
        save_total_limit=None,
        eval_strategy="no",
        report_to="none",
        max_completion_length=spec.max_completion_length,
        max_tool_calling_iterations=spec.max_tool_calling_iterations,
        temperature=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=spec.vllm_gpu_memory_utilization,
        vllm_max_model_length=spec.vllm_max_model_length,
        scale_rewards=spec.scale_rewards,
        loss_type=spec.loss_type,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": "bfloat16"},
        seed=spec.seed,
        shuffle_dataset=False,
    )
    peft_config = None
    model = spec.model
    if spec.adapter_path:
        base = create_model_from_path(spec.model, dtype="bfloat16")
        model = PeftModel.from_pretrained(base, spec.adapter_path)
    else:
        peft_config = LoraConfig(
            r=spec.lora_r, lora_alpha=2 * spec.lora_r, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )
    train_dataset = Dataset.from_list(train_rows) if train_rows else Dataset.from_list(periodic["clean"][:per_device])

    class PhasedTrainer(GRPOTrainer):
        """Stamps the phase onto episode records, chunks log-prob passes, and skips the loss in evaluation.

        A log-prob pass materializes (chunk, max_completion_length, vocab) logits at least twice: 3 GB per
        sequence in bf16 with a 248k vocabulary and a 6144-token cap, which is why the chunk is small.
        """

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            BackOfficeEnv.phase = f"{metric_key_prefix}:step{self.state.global_step}"
            try:
                return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            finally:
                BackOfficeEnv.phase = "train"

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kwargs):
            chunk = min(batch_size or input_ids.size(0), spec.logprob_chunk)
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, chunk, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch

            self._prepare_inputs(inputs)
            return torch.zeros((), device=self.accelerator.device), None, None

        def _generate_and_score_completions(self, inputs):
            t0 = time.perf_counter()
            output = super()._generate_and_score_completions(inputs)
            if self.model.training:
                self.generation_seconds = round(time.perf_counter() - t0, 1)
            if group_log_path is not None and self.model.training and self.environments:
                n = len(inputs)
                name = self.reward_func_names[0]
                records = training_group_records(
                    self.state.global_step, BackOfficeEnv.attempt, list(inputs),
                    [env.last_episode for env in self.environments[:n]],
                    list(self._logs["rewards"][name])[-n:], output["advantages"].tolist(),
                    spec.num_generations, scale=spec.scale_rewards,
                )
                with open(group_log_path, "a", encoding="utf-8") as f:
                    f.writelines(json.dumps(record, sort_keys=True) + "\n" for record in records)
            return output

    trainer = PhasedTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        environment_factory=make_env_factory(task_paths, log_path=log_path),
        peft_config=peft_config,
    )
    return trainer, splits, periodic


class EvalSchedule:
    """Runs the periodic evaluations and diagnostics at the requested steps from a TrainerCallback.

    `on_checkpoint` is called after every evaluation phase and every `checkpoint_every` training steps so the
    caller can persist partial outputs; a run that dies later still leaves everything up to the last checkpoint.
    """

    def __init__(self, spec: RunSpec, trainer, splits, periodic, on_checkpoint=None, checkpoint_every: int = 10):
        from datasets import Dataset
        from transformers import TrainerCallback

        from pairedrl.train.env_adapter import BackOfficeEnv

        self.spec = spec
        self.trainer = trainer
        self.periodic = {name: Dataset.from_list(rows) for name, rows in periodic.items()}
        self.diagnostic = Dataset.from_list(assemble_diagnostic_rows(spec, splits["diagnostic"]))
        self.timings: list[dict] = []
        self.step_timings: list[dict] = []
        self.on_checkpoint = on_checkpoint
        self.checkpoint_every = checkpoint_every
        runner = self
        clock = {"step_begin": None, "step_end": None}

        class Callback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                BackOfficeEnv.phase = f"train:step{state.global_step}"
                clock["step_begin"] = time.perf_counter()
                runner.trainer.generation_seconds = None

            def on_step_end(self, args, state, control, **kwargs):
                step = state.global_step
                now = time.perf_counter()
                if clock["step_begin"] is not None:
                    runner.step_timings.append({
                        "step": step, "seconds": round(now - clock["step_begin"], 1),
                        "generation_s": runner.trainer.generation_seconds,
                    })
                # Evaluations log through the trainer, which clears the flags the flow callback just set for
                # this step and leaves the model in eval mode; both are restored so the step's loss and train
                # metrics are logged under the right mode and its checkpoint is saved.
                should_log, should_save = control.should_log, control.should_save
                was_training = runner.trainer.model.training
                if not runner.spec.train_only:
                    if step % runner.spec.eval_every == 0 or step == runner.spec.steps:
                        runner.run_periodic(step)
                    if step in runner.spec.diagnostic_steps and step < runner.spec.steps:
                        runner.run_diagnostic(step)
                control.should_log, control.should_save = should_log, should_save
                if was_training:
                    runner.trainer.model.train()
                BackOfficeEnv.phase = f"train:step{step}"
                clock["step_end"] = time.perf_counter()
                if step % runner.checkpoint_every == 0:
                    runner.checkpoint()

            def on_save(self, args, state, control, **kwargs):
                if runner.step_timings and clock["step_end"] is not None:
                    runner.step_timings[-1]["save_s"] = round(time.perf_counter() - clock["step_end"], 1)

        self.callback = Callback()

    def checkpoint(self) -> None:
        if self.on_checkpoint is not None:
            self.on_checkpoint()

    def _evaluate(self, step: int, what: str, dataset) -> None:
        t0 = time.perf_counter()
        self.trainer.evaluate(eval_dataset=dataset, metric_key_prefix=what)
        self.timings.append({"step": step, "what": what, "seconds": round(time.perf_counter() - t0, 1)})
        self.checkpoint()

    def run_periodic(self, step: int) -> None:
        for name, ds in self.periodic.items():
            self._evaluate(step, f"eval_{name}", ds)

    def run_diagnostic(self, step: int) -> None:
        self._evaluate(step, "diag", self.diagnostic)

    def run_final(self, splits) -> None:
        from datasets import Dataset

        for name, rows in assemble_eval_sets(self.spec, splits["test"], final=True).items():
            if rows:
                self._evaluate(self.spec.steps, f"final_{name}", Dataset.from_list(rows))


def read_episode_log(path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

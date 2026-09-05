"""Run specification, dataset assembly, trainer construction and evaluation orchestration."""

import dataclasses
import json
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass, field

from pairedrl.env.backoffice.tasks import Task, read_jsonl
from pairedrl.noise.config import NoiseConfig
from pairedrl.train.dataset import (
    build_blocking_rows,
    build_diagnostic_rows,
    build_eval_rows,
    build_training_rows,
)

ARMS = ("paired", "independent", "clean", "blocking")
EVAL_SEEDS_PERIODIC = [1, 2]
EVAL_SEEDS_FINAL = [1, 2, 3, 4]


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
    scale_rewards: str = "group"
    loss_type: str = "dapo"
    eval_every: int = 20
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
    notes: str = ""

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.arm == "clean" and (self.p > 0 or self.q > 0):
            raise ValueError("the clean arm has p = q = 0")
        if self.arm != "clean" and self.p == 0 and self.q == 0:
            raise ValueError("a noisy arm needs p > 0 or q > 0")
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

    def training_noise(self) -> NoiseConfig:
        if self.arm == "clean":
            return NoiseConfig.clean()
        mode = "independent" if self.arm == "blocking" else self.arm
        return NoiseConfig(p=self.p, q=self.q, mode=mode)

    def eval_noise(self) -> NoiseConfig:
        """Evaluation always pairs: fixed schedules shared by every arm and checkpoint."""
        return NoiseConfig(p=self.p, q=self.q, mode="paired") if self.arm != "clean" else NoiseConfig.clean()


def assemble_training_rows(spec: RunSpec, tasks: list[Task]) -> list[dict]:
    n_rows = spec.steps * spec.prompts_per_step
    config = spec.training_noise()
    if spec.arm == "blocking":
        return build_blocking_rows(tasks, config, spec.condition, n_rows, spec.seed)
    return build_training_rows(tasks, config, spec.condition, n_rows, spec.seed)


def assemble_eval_sets(spec: RunSpec, heldout: list[Task], final: bool) -> dict[str, list[dict]]:
    n = spec.final_eval_tasks if final else spec.eval_tasks
    seeds = EVAL_SEEDS_FINAL if final else EVAL_SEEDS_PERIODIC
    tasks = heldout[:n]
    sets = {"clean": build_eval_rows(tasks, NoiseConfig.clean(), "eval:clean", seeds)}
    if spec.arm != "clean":
        sets["noisy"] = build_eval_rows(tasks, spec.eval_noise(), "eval:noisy", seeds)
    else:
        sets["noisy"] = build_eval_rows(tasks, NoiseConfig.transition(0.25, "paired"), "eval:noisy", seeds)
    if final:
        sets["heldout_types"] = build_eval_rows(
            tasks, NoiseConfig.heldout_types(max(spec.p, 0.25), "paired"), "eval:heldout_types", seeds
        )
    return sets


def assemble_diagnostic_rows(spec: RunSpec, diagnostic: list[Task]) -> list[dict]:
    """K schedules x M samples per task; identical rows share a schedule seed, so M samples pair exactly."""
    p = spec.p if spec.p > 0 else 0.25
    config = NoiseConfig.transition(p, "paired")
    base = build_diagnostic_rows(diagnostic, config, "diag", spec.diagnostic_schedules, base_seed=spec.seed)
    return [dict(row) for row in base for _ in range(spec.diagnostic_samples)]


def load_task_splits(data_dir) -> dict[str, list[Task]]:
    data_dir = pathlib.Path(data_dir)
    return {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ("train", "heldout", "diagnostic")}


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


def luck_share_tables(records: list[dict], field: str = "true_success") -> list[list[list[float]]]:
    """Group diagnostic episodes into K x M reward tables per task, keyed by (task_id, schedule_seed)."""
    by_task: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("condition") == "diag":
            by_task[r["task_id"]][r["schedule_seed"]].append(float(r[field]))
    tables = []
    for task_id in sorted(by_task):
        rows = [samples for _, samples in sorted(by_task[task_id].items())]
        m = min(len(s) for s in rows)
        if len(rows) >= 2 and m >= 2:
            tables.append([s[:m] for s in rows])
    return tables


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


def build_trainer(spec: RunSpec, data_dir, out_dir, log_path):
    """Construct the TRL trainer. Imports TRL lazily so the rest of the module stays importable on any machine."""
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.utils import create_model_from_path

    from pairedrl.train.env_adapter import BackOfficeEnv, make_env_factory

    patch_vllm_engine(vllm_engine_overrides(spec))
    splits = load_task_splits(data_dir)
    task_paths = [str(pathlib.Path(data_dir) / f"{n}.jsonl") for n in ("train", "heldout", "diagnostic")]
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
        save_strategy="no",
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

        self.spec = spec
        self.trainer = trainer
        self.periodic = {name: Dataset.from_list(rows) for name, rows in periodic.items()}
        self.diagnostic = Dataset.from_list(assemble_diagnostic_rows(spec, splits["diagnostic"]))
        self.timings: list[dict] = []
        self.on_checkpoint = on_checkpoint
        self.checkpoint_every = checkpoint_every
        runner = self

        class Callback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                step = state.global_step
                if step % runner.spec.eval_every == 0 and step < runner.spec.steps:
                    runner.run_periodic(step)
                if step in runner.spec.diagnostic_steps and step < runner.spec.steps:
                    runner.run_diagnostic(step)
                if step % runner.checkpoint_every == 0:
                    runner.checkpoint()

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

        for name, rows in assemble_eval_sets(self.spec, splits["heldout"], final=True).items():
            self._evaluate(self.spec.steps, f"final_{name}", Dataset.from_list(rows))


def read_episode_log(path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

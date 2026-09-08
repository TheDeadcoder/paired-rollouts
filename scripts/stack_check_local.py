"""Stack check on the machine it runs on (a DigitalOcean MI300X droplet, a local GPU box): the same checks as
`stack_check_modal.py` (TRL GRPOTrainer + environment_factory + vLLM colocate + LoRA on a toy counter
environment) plus the kernel availability, and optionally a real back-office evaluation batch whose timings
are comparable with the Modal speed checks. Writes `registers/stack_check_<provider>_<model>.json`.

    python scripts/stack_check_local.py --provider digitalocean --model Qwen/Qwen3.5-2B --real-eval 32
"""

import argparse
import datetime as dt
import importlib.metadata as md
import importlib.util
import json
import pathlib
import platform
import time
import traceback
from typing import ClassVar

SENTINEL = "TOOLRESULT_7f3a"
PACKAGES = ["torch", "transformers", "trl", "vllm", "peft", "accelerate", "datasets", "flash-linear-attention"]


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def toy_check(model_id: str, max_steps: int, out_dir: pathlib.Path) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    t_start = time.perf_counter()
    result = {"model": model_id, "max_steps": max_steps, "errors": [], "adaptations": []}

    class CounterEnv:
        reset_kwargs_seen: ClassVar[set[str]] = set()
        reset_calls: ClassVar[int] = 0

        def reset(self, **kwargs):
            CounterEnv.reset_kwargs_seen.update(kwargs.keys())
            CounterEnv.reset_calls += 1
            self.counter = 0
            self.target = int(kwargs.get("target", 3))

        def increment(self, step: int) -> str:
            """Add an integer amount to the counter and return the new counter value.

            Args:
                step: The integer amount to add to the counter.
            """
            self.counter += int(step)
            return f"{SENTINEL} counter={self.counter}"

        def read_counter(self) -> str:
            """Return the current value of the counter."""
            return f"{SENTINEL} counter={self.counter}"

        def get_reward(self) -> float:
            return 1.0 if self.counter == self.target else 0.0

    system = (
        "You control a counter that starts at 0. Use the tools to make the counter exactly equal "
        "to the target, then reply with the single word DONE."
    )
    targets = [2, 3, 4, 5] * 4
    rows = [
        {
            "prompt": [{"role": "system", "content": system}, {"role": "user", "content": f"Target: {t}"}],
            "task_id": f"toy-{i:03d}",
            "target": t,
            "schedule_seed": 1000 + i,
        }
        for i, t in enumerate(targets)
    ]
    dataset = Dataset.from_list(rows)
    captured = {}

    class CheckedTrainer(GRPOTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not captured and "completion_ids" in inputs:
                captured["completion_ids"] = inputs["completion_ids"].detach().cpu()
                captured["completion_mask"] = inputs["completion_mask"].detach().cpu()
                tm = inputs.get("tool_mask")
                captured["tool_mask"] = tm.detach().cpu() if tm is not None else None
                captured["input_keys"] = sorted(str(k) for k in inputs)
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.times = []

        def on_step_begin(self, args, state, control, **kwargs):
            self.t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            self.times.append(round(time.perf_counter() - self.t0, 2))

    config = GRPOConfig(
        output_dir=str(out_dir / "toy"),
        max_steps=max_steps,
        per_device_train_batch_size=16,
        generation_batch_size=16,
        num_generations=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_completion_length=256,
        max_tool_calling_iterations=4,
        temperature=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        scale_rewards="group",
        loss_type="dapo",
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": "bfloat16"},
        seed=0,
    )
    result["config"] = {
        k: getattr(config, k)
        for k in [
            "per_device_train_batch_size", "generation_batch_size", "num_generations",
            "max_completion_length", "max_tool_calling_iterations", "vllm_mode",
            "vllm_gpu_memory_utilization", "scale_rewards", "loss_type", "chat_template_kwargs",
        ]
    }
    peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    timer = StepTimer()
    tokenizer = None
    try:
        trainer = CheckedTrainer(
            model=model_id, args=config, train_dataset=dataset, environment_factory=CounterEnv,
            peft_config=peft_config, callbacks=[timer],
        )
        pc = trainer.processing_class
        tokenizer = getattr(pc, "tokenizer", pc)
        trainer.train()
        result["log_history"] = trainer.state.log_history
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"{type(e).__name__}: {e}")
        result["traceback"] = traceback.format_exc()[-6000:]

    result["reset_kwargs_keys"] = sorted(CounterEnv.reset_kwargs_seen)
    result["reset_calls"] = CounterEnv.reset_calls
    result["step_times_s"] = timer.times
    result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    hist = result.get("log_history", [])
    result["tools_call_frequency"] = [h.get("tools/call_frequency") for h in hist if "tools/call_frequency" in h]
    result["tools_failure_frequency"] = [h.get("tools/failure_frequency") for h in hist if "tools/failure_frequency" in h]
    result["reward_per_step"] = [h.get("reward") for h in hist if "reward" in h]

    mask = {"input_keys": captured.get("input_keys"), "tool_mask_present": captured.get("tool_mask") is not None}
    if captured and captured.get("tool_mask") is not None and tokenizer is not None:
        ids, cm, tm = captured["completion_ids"], captured["completion_mask"].bool(), captured["tool_mask"]
        tool_tok = cm & (tm == 0)
        model_tok = cm & (tm == 1)
        tool_text = " ".join(tokenizer.decode(row[m]) for row, m in zip(ids, tool_tok))
        model_text = " ".join(tokenizer.decode(row[m]) for row, m in zip(ids, model_tok))
        mask["tool_result_tokens_total"] = int(tool_tok.sum())
        mask["model_tokens_total"] = int(model_tok.sum())
        mask["sentinel_count_tool_tokens"] = tool_text.count(SENTINEL)
        mask["sentinel_count_model_tokens"] = model_text.count(SENTINEL)
        mask["tool_text_sample"] = tool_text[:300]
        mask["pass"] = (
            mask["tool_result_tokens_total"] > 0 and mask["sentinel_count_tool_tokens"] > 0
            and mask["sentinel_count_model_tokens"] == 0
        )
    else:
        mask["pass"] = False
    result["mask_check"] = mask
    expected = {"prompt", "task_id", "target", "schedule_seed"}
    result["checks"] = {
        "reset_receives_row_fields": expected.issubset(set(result["reset_kwargs_keys"])),
        "tool_calls_parsed": bool(result["tools_call_frequency"]) and max(result["tools_call_frequency"]) >= 0.5,
        "tool_mask_applied": mask["pass"],
        "trained_without_error": not result["errors"] and len(timer.times) == max_steps,
        "step_time_under_60s": bool(timer.times) and max(timer.times) < 60,
    }
    result["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    return result


def real_eval_check(model_id: str, n_tasks: int, out_dir: pathlib.Path, data_dir: str) -> dict:
    """One final evaluation of the actual back-office runner on `n_tasks` test tasks: the per-set seconds are
    comparable with the Modal speed checks and calibrations (same code path, per_device_eval_batch_size 128)."""
    from pairedrl.train.runner import (
        EvalSchedule,
        RunSpec,
        build_trainer,
        read_episode_log,
        summarize_episodes,
    )

    t_start = time.perf_counter()
    spec = RunSpec(
        run_id="stack-check-real", model=model_id, condition="C2", arm="paired", p=0.25, eval_only=True,
        final_eval_tasks=n_tasks, diagnostic_steps=[], notes="stack check, real evaluation batch",
    )
    result = {"spec": spec.to_dict(), "errors": []}
    log_path = out_dir / "real" / "episodes.jsonl"
    try:
        trainer, splits, periodic = build_trainer(spec, data_dir, out_dir / "real" / "trainer", log_path)
        schedule = EvalSchedule(spec, trainer, splits, periodic)
        schedule.run_final(splits)
        result["eval_timings"] = schedule.timings
        records = read_episode_log(log_path)
        result["episodes"] = len(records)
        result["by_phase_condition"] = summarize_episodes(records)
        total = sum(t["seconds"] for t in schedule.timings)
        result["episodes_per_second"] = round(len(records) / total, 3) if total else None
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"{type(e).__name__}: {e}")
        result["traceback"] = traceback.format_exc()[-6000:]
    result["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, help="short provider name used in the register file name")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--real-eval", type=int, default=0, help="test tasks for the real evaluation batch (0 skips it)")
    parser.add_argument("--data-dir", default="data/tasks")
    parser.add_argument("--out-dir", default="outputs/stack_check")
    parser.add_argument("--register-dir", default="registers")
    args = parser.parse_args()

    import torch

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provider": args.provider,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_backend": "hip" if getattr(torch.version, "hip", None) else "cuda",
        "torch_backend_version": getattr(torch.version, "hip", None) or torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: package_version(p) for p in PACKAGES},
        "fla_importable": importlib.util.find_spec("fla") is not None,
    }
    result["toy"] = toy_check(args.model, args.max_steps, out_dir)
    if args.real_eval > 0:
        result["real_eval"] = real_eval_check(args.model, args.real_eval, out_dir, args.data_dir)
    slug = args.model.split("/")[-1].lower()
    path = pathlib.Path(args.register_dir) / f"stack_check_{args.provider}_{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    summary = {
        "provider": args.provider, "gpu": result["gpu"], "backend": result["torch_backend_version"],
        "versions": result["versions"], "fla_importable": result["fla_importable"],
        "toy_checks": result["toy"]["checks"], "toy_step_times_s": result["toy"]["step_times_s"],
        "toy_errors": result["toy"]["errors"],
    }
    if "real_eval" in result:
        real = result["real_eval"]
        summary["real_eval"] = {
            "episodes": real.get("episodes"), "eval_timings": real.get("eval_timings"),
            "episodes_per_second": real.get("episodes_per_second"), "errors": real["errors"],
        }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

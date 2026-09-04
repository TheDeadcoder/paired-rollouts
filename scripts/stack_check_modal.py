"""Day-1 stack check on Modal: TRL GRPOTrainer + environment_factory + vLLM colocate + LoRA.

Runs a few GRPO steps on a toy counter environment and writes one register per model with
versions, reset kwargs, tool-call statistics, the tool-mask check, per-step timings and memory.
"""

import datetime as dt
import json
import pathlib
import time
from typing import ClassVar

import modal

APP_NAME = "pairedrl-stack-check"
HF_CACHE = "/cache/hf"
SENTINEL = "TOOLRESULT_7f3a"
MODELS = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3-1.7B"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.27.1",
        "trl[vllm,peft]==1.12.0",
        "accelerate>=1.14.0",
        "datasets>=5.0.0",
    )
    .env({"HF_HOME": HF_CACHE, "TRL_EXPERIMENTAL_SILENCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("pairedrl-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=3600, volumes={HF_CACHE: hf_cache})
def stack_check(model_id: str, max_steps: int = 3) -> dict:
    import importlib.metadata as md
    import platform
    import traceback

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    t_start = time.perf_counter()
    result = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provider": "modal",
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "versions": {
            p: md.version(p)
            for p in ["torch", "transformers", "trl", "vllm", "peft", "accelerate", "datasets"]
        },
        "model": model_id,
        "max_steps": max_steps,
        "errors": [],
        "adaptations": [],
    }

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
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Target: {t}"},
            ],
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
        output_dir="/tmp/stack_check",
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
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    timer = StepTimer()
    tokenizer = None
    try:
        trainer = CheckedTrainer(
            model=model_id,
            args=config,
            train_dataset=dataset,
            environment_factory=CounterEnv,
            peft_config=peft_config,
            callbacks=[timer],
        )
        pc = trainer.processing_class
        tokenizer = getattr(pc, "tokenizer", pc)
        result["response_template_set"] = getattr(tokenizer, "response_template", None) is not None
        result["response_schema_set"] = getattr(tokenizer, "response_schema", None) is not None
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
    result["tools_call_frequency"] = [
        h.get("tools/call_frequency") for h in hist if "tools/call_frequency" in h
    ]
    result["tools_failure_frequency"] = [
        h.get("tools/failure_frequency") for h in hist if "tools/failure_frequency" in h
    ]
    result["reward_per_step"] = [h.get("reward") for h in hist if "reward" in h]

    mask = {
        "input_keys": captured.get("input_keys"),
        "tool_mask_present": captured.get("tool_mask") is not None,
    }
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
        mask["pass"] = mask["tool_result_tokens_total"] > 0 and mask["sentinel_count_tool_tokens"] > 0
    else:
        mask["pass"] = False
    result["mask_check"] = mask

    expected = {"prompt", "task_id", "target", "schedule_seed"}
    result["checks"] = {
        "reset_receives_row_fields": expected.issubset(set(result["reset_kwargs_keys"])),
        "tool_calls_parsed": bool(result["tools_call_frequency"]) and max(result["tools_call_frequency"]) >= 0.5,
        "tool_mask_applied": mask["pass"],
        "trained_without_error": not result["errors"] and len(timer.times) == max_steps,
        "step_time_under_30s": bool(timer.times) and max(timer.times) < 30,
    }
    result["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    return result


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), max_steps: int = 3):
    out_dir = pathlib.Path("registers")
    out_dir.mkdir(exist_ok=True)
    for model_id in models.split(","):
        res = stack_check.remote(model_id.strip(), max_steps)
        slug = model_id.strip().split("/")[-1].lower()
        path = out_dir / f"stack_check_modal_{slug}.json"
        path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        print(json.dumps({"model": res["model"], "checks": res["checks"], "versions": res["versions"],
                          "step_times_s": res["step_times_s"], "errors": res["errors"]}, indent=2))

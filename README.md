# paired-rollouts

Paired rollouts for group-relative RL of LLM agents in stochastic environments.

Group-relative policy optimization (GRPO and its critic-free relatives) scores every rollout against its siblings from the same prompt. In agentic environments each sibling normally draws its own environment randomness: tool calls fail, reads go stale, graders flake. This repository studies what that does to the group-relative estimator and tests the fix: all rollouts in a group share one pre-drawn noise schedule (common random numbers), while different groups get different schedules.

Status: scaffold. The pre-registration lives in `docs/PREREGISTRATION.md` and is frozen with RFC 3161 timestamps before any pre-registered run starts.

## Layout

- `src/pairedrl/env/` back-office tool environment (state, tools, tasks, grader)
- `src/pairedrl/noise/` seeded noise schedules and noise types
- `src/pairedrl/train/` TRL GRPO integration and dataset builders
- `src/pairedrl/evaluation/` paired evaluation harness and metrics
- `src/pairedrl/analysis/` register builders and figures
- `src/pairedrl/census/` flaky-test census over SWE-smith instances
- `src/pairedrl/bfcl/` BFCL v3 multi-turn turn-episode environment
- `scripts/` entry points for training, evaluation and census jobs
- `configs/` run configurations
- `registers/` generated registers; every number in the paper is emitted here
- `docs/` pre-registration, daily log, deviations, run ledger

## Development setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The `train` extra (torch, transformers, trl, peft, accelerate, datasets, vllm) is installed only on GPU machines and is pinned in `pyproject.toml` after the stack check.

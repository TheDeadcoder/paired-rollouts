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

## Running on Modal

Runs are spawned onto a deployed Modal app and never depend on the launching machine staying online.

```
modal deploy scripts/run_modal.py
python scripts/launch_modal.py --specs configs/a.json,configs/b.json --label <name>
python scripts/collect_modal.py
python scripts/collect_modal.py --cancel
```

`deploy` is run once per commit and bakes the source, the task data and the commit hash into the image; a run whose launch commit differs from the deployed one refuses to start. `launch` returns in seconds, refuses a tree with uncommitted changes, writes `registers/launches/<utc>_<label>.json` (run ids, Modal call ids, commit) and appends one `LAUNCHED` row per run to `docs/RUN_LEDGER.md`. `collect` can be run at any time from any machine with Modal credentials: it prints each run's state and progress, downloads finished runs to `outputs/runs/<run_id>/` (not committed; the same files stay on the `pairedrl-runs` volume) and fills in the ledger status. `--cancel` stops every still-running run of the newest launch and terminates its containers. Runs checkpoint their manifest, trainer log and episode log to the volume after every evaluation phase and every ten training steps, so a run that dies leaves everything up to its last checkpoint.

## Development setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The `train` extra (torch, transformers, trl, peft, accelerate, datasets, vllm) is installed only on GPU machines and is pinned in `pyproject.toml` after the stack check.

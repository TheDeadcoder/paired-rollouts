# paired-rollouts

Paired rollouts for group-relative RL of LLM agents in stochastic environments.

Group-relative policy optimization (GRPO and its critic-free relatives) scores every rollout against its siblings from the same prompt. In agentic environments each sibling normally draws its own environment randomness: tool calls fail, reads go stale, graders flake. This repository studies what that does to the group-relative estimator and tests the fix: all rollouts in a group share one pre-drawn noise schedule (common random numbers), while different groups get different schedules.

Status: calibration and exploratory runs; the pre-registration (`docs/PREREGISTRATION.md`, draft v0.2) is frozen with RFC 3161 timestamps before any pre-registered run starts. The theory notes are in `docs/THEORY.md`; departures from the plan are in `docs/DEVIATIONS.md`.

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
python scripts/launch_modal.py --specs configs/a.json,configs/b.json --label <name> [--preregistration v1]
python scripts/collect_modal.py
python scripts/collect_modal.py --cancel
python scripts/verify_run.py outputs/runs/<run_id> [--weights]
```

`deploy` is run once per commit and bakes the source, the task data and the commit hash into the image; a run whose launch commit differs from the deployed one refuses to start. `launch` returns in seconds, refuses a tree with uncommitted changes (the ledger and the launch registers excepted, since the previous launch of a batch writes them), refuses a run id whose newest ledger row is still open or that the ledger records as COMPLETE (`--relaunch` overrides the former for a job known to be dead), writes `registers/launches/<utc>_<label>.json` (run ids, Modal call ids, commit, pre-registration label) and appends one `LAUNCHED` row per run to `docs/RUN_LEDGER.md`; `--preregistration <label>` marks the rows and the register as pre-registered and is accepted only when every spec's notes carry `pre-registered <label>`, otherwise the rows say `not pre-registered`. `collect` can be run at any time from any machine with Modal credentials: it prints each run's state and progress, downloads finished runs to `outputs/runs/<run_id>/` (not committed; the same files stay on the `pairedrl-runs` volume; trainer checkpoints and the final adapter are skipped unless `--with-weights` is given) and fills in the ledger status. `--cancel` stops every still-running run of the newest launch and terminates its containers. Runs checkpoint their manifest, trainer log, episode log (`episodes.jsonl`) and training-group register (`groups.jsonl`, the trainer's own groups with true outcomes, observed rewards and advantages) to the volume after every evaluation phase and every ten training steps, and save the trainer state (adapter, optimizer, scheduler, RNG) every `checkpoint_steps` steps. A container that Modal retries (preemption, or the one re-raise of a failed first attempt) moves the previous attempt's files to `attempt<n>/` and resumes from the newest complete trainer checkpoint; a relaunch with a different spec or of a COMPLETE run is refused and recorded under `refused/`. `verify_run` rebuilds the run register along the attempt lineage and prints ACCEPT or REJECT against the evidence the spec and the frozen task pools prescribe: every periodic set at every registered step and every final set at the final step with exactly the expected task and schedule identities, the sixteen frozen diagnostic tasks with K x M records each at every diagnostic step, every training step's groups on their rows in trainer order with `num_generations` members, no evaluation or training phase outside that plan, exact episode and group totals, pool files whose SHA-256 match `data/tasks/MANIFEST.json`, one code commit across attempts (a relaunch at another commit must name the previous one in the spec's `relaunch_from_commit`, and the run job refuses it otherwise), one model snapshot and one set of pools; with `--weights` every expected checkpoint must hold the adapter with its config, `optimizer.pt`, `scheduler.pt`, the RNG state and a `trainer_state.json` whose `global_step` is the checkpoint's step, plus `adapter_final/`. A checkpoint missing any of those is never resumed from either. `scripts/decide.py` applies the pre-registered rules only to runs that pass this check and records the pool hashes with the decision.

## Running on DigitalOcean (AMD MI300X)

The same job body runs on a droplet through `scripts/run_local.py`; `scripts/launch_remote.py` starts specs over SSH in a detached shell and `scripts/collect_remote.py` rsyncs the results back. See `docs/DIGITALOCEAN.md`.

## Development setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cloud]"
pytest -q
```

The `cloud` extra (modal) is needed by the launch and collect scripts and by their tests, which are skipped without it. The `train` extra (torch, transformers, trl, peft, accelerate, datasets, vllm) is installed only on GPU machines and is pinned in `pyproject.toml` after the stack check.

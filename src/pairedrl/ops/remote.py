"""Commands for running specs on a remote GPU host over SSH (a DigitalOcean droplet) and collecting the results."""

import shlex

ENV = "HF_HOME={hf_home} TRL_EXPERIMENTAL_SILENCE=1 TOKENIZERS_PARALLELISM=false"


def ssh_command(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", host, command]


def checkout_command(remote_repo: str, commit: str) -> str:
    """Move the remote checkout to `commit`. The checkout on a GPU host is a deployment target, never edited there,
    but scripts run from it can leave files behind (a stack-check register later committed from the laptop, for
    instance); `-f` lets the committed version replace an untracked file that is in the way instead of aborting.
    Modified tracked files are listed first so the log shows what was discarded."""
    repo = shlex.quote(remote_repo)
    return (
        f"cd {repo} && git fetch -q origin && git status --porcelain && git checkout -q -f {shlex.quote(commit)} "
        f"&& git rev-parse HEAD"
    )


def in_container(container: str | None, command: str, detached: bool = False) -> str:
    """Wrap a shell command for `docker exec` when the stack lives in a container, else run it on the host."""
    if container is None:
        return command
    flag = "-d " if detached else ""
    return f"docker exec {flag}{shlex.quote(container)} bash -c {shlex.quote(command)}"


def install_command(repo: str, container: str | None) -> str:
    return in_container(container, f"cd {shlex.quote(repo)} && pip install -q -e .")


CHAIN_MARKER = "scripts/run_local.py"
QUEUE_MARKER = "scripts/run_queue.py"


def _pgrep(marker: str) -> str:
    """pgrep for a script; the bracketed first letter keeps pgrep from matching its own command line."""
    return f"pgrep -f {shlex.quote(f'[{marker[0]}]{marker[1:]}')} > /dev/null"


def chain_status_command(container: str | None) -> str:
    """Prints RUNNING when a job chain or a queue runner is alive where the jobs run, IDLE otherwise (a chain never
    shares the GPU with anything)."""
    return in_container(container, f"({_pgrep(CHAIN_MARKER)} || {_pgrep(QUEUE_MARKER)}) && echo RUNNING || echo IDLE")


def queue_host_status_command(container: str | None) -> str:
    """Prints QUEUE when the droplet queue's runner is alive, CHAIN when only jobs outside a queue run, IDLE otherwise."""
    return in_container(container, f"{_pgrep(QUEUE_MARKER)} && echo QUEUE || ({_pgrep(CHAIN_MARKER)} && echo CHAIN || echo IDLE)")


def ensure_checkout_command(remote_repo: str, checkouts_dir: str, commit: str) -> str:
    """A checkout of `commit` of its own under checkouts_dir (a local clone of the host's repository, so git works
    inside the container too): queued entries run from the checkout of their commit, so entries appended at a later
    commit never change the code of jobs already queued or running."""
    repo, base = shlex.quote(remote_repo), shlex.quote(checkouts_dir)
    target = shlex.quote(f"{checkouts_dir.rstrip('/')}/{commit}")
    return (
        f"cd {repo} && git fetch -q origin && mkdir -p {base} && (test -d {target}/.git || git clone -q {repo} {target}) "
        f"&& cd {target} && git checkout -q -f {shlex.quote(commit)} && git rev-parse HEAD"
    )


def queue_append_command(container: str | None, checkout: str, queue_dir: str, entry: dict) -> str:
    """Append one entry to the droplet queue; prints APPENDED <id> RUNNER_ALIVE or RUNNER_IDLE."""
    import json

    ckout = shlex.quote(checkout)
    return in_container(container, (
        f"cd {ckout} && PYTHONPATH={ckout}/src python {QUEUE_MARKER} append --queue-dir {shlex.quote(queue_dir)} "
        f"--entry-json {shlex.quote(json.dumps(entry, sort_keys=True))}"
    ))


def queue_run_command(
    container: str | None, checkout: str, queue_dir: str, runs_dir: str, provider: str, usd_per_hour: float, hf_home: str,
) -> str:
    """Start the droplet queue's runner detached (docker exec -d, or nohup on a pod), so it survives the laptop going
    offline; a second runner on the same queue exits at once."""
    env = ENV.format(hf_home=shlex.quote(hf_home))
    ckout, queue = shlex.quote(checkout), shlex.quote(queue_dir)
    inner = (
        f"cd {ckout} && mkdir -p {queue} && {env} PYTHONPATH={ckout}/src python {QUEUE_MARKER} run --queue-dir {queue} "
        f"--runs-dir {shlex.quote(runs_dir)} --provider {shlex.quote(provider)} --usd-per-hour {usd_per_hour} "
        f">> {queue}/runner.log 2>&1"
    )
    if container is None:
        return f"nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 &"
    return in_container(container, inner, detached=True)


def read_files_command(*paths: str) -> str:
    """Print each file (or nothing when it is absent), separated by a marker line, on the host."""
    return " ; echo '----8<----' ; ".join(f"cat {shlex.quote(p)} 2>/dev/null" for p in paths)


def run_chain_command(
    repo: str, spec_paths: list[str], runs_dir: str, provider: str, usd_per_hour: float, commit: str,
    hf_home: str, container: str | None,
) -> str:
    """One detached shell that runs the specs one after another, each logging to <runs_dir>/<spec stem>.log. A
    spec whose job does not end COMPLETE is run a second time at once (the relaunch resumes from the newest complete
    checkpoint; `attempt<n>/` keeps the first attempt), then the chain moves on whatever the outcome: the remote
    host has no platform retries, so this is the whole retry policy, and `collect_remote` shows what happened."""
    env = ENV.format(hf_home=shlex.quote(hf_home))
    parts = []
    for path in spec_paths:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        job = (
            f"{env} python {CHAIN_MARKER} --spec {shlex.quote(path)} --runs-dir {shlex.quote(runs_dir)} "
            f"--provider {shlex.quote(provider)} --usd-per-hour {usd_per_hour} --commit {shlex.quote(commit)}"
        )
        log = f"{shlex.quote(runs_dir)}/{stem}.log"
        parts.append(f"{job} > {log} 2>&1 || {job} >> {log} 2>&1")
    chain = "; ".join(parts)
    inner = f"cd {shlex.quote(repo)} && mkdir -p {shlex.quote(runs_dir)} && ({chain})"
    if container is None:
        return f"nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 &"
    return in_container(container, inner, detached=True)


def rsync_command(host: str, remote_run_dir: str, local_run_dir: str, with_weights: bool = False) -> list[str]:
    cmd = ["rsync", "-az", "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"]
    if not with_weights:
        cmd += ["--exclude", "trainer/", "--exclude", "adapter_final/"]
    cmd += [f"{host}:{remote_run_dir.rstrip('/')}/", f"{local_run_dir.rstrip('/')}/"]
    return cmd

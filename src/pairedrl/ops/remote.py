"""Commands for running specs on a remote GPU host over SSH (a DigitalOcean droplet) and collecting the results."""

import shlex

ENV = "HF_HOME={hf_home} TRL_EXPERIMENTAL_SILENCE=1 TOKENIZERS_PARALLELISM=false"


def ssh_command(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", host, command]


def checkout_command(remote_repo: str, commit: str) -> str:
    return f"cd {shlex.quote(remote_repo)} && git fetch -q origin && git checkout -q {shlex.quote(commit)} && git rev-parse HEAD"


def in_container(container: str | None, command: str, detached: bool = False) -> str:
    """Wrap a shell command for `docker exec` when the stack lives in a container, else run it on the host."""
    if container is None:
        return command
    flag = "-d " if detached else ""
    return f"docker exec {flag}{shlex.quote(container)} bash -c {shlex.quote(command)}"


def install_command(repo: str, container: str | None) -> str:
    return in_container(container, f"cd {shlex.quote(repo)} && pip install -q -e .")


def run_chain_command(
    repo: str, spec_paths: list[str], runs_dir: str, provider: str, usd_per_hour: float, commit: str,
    hf_home: str, container: str | None,
) -> str:
    """One detached shell that runs the specs one after another, each logging to <runs_dir>/<spec stem>.log."""
    env = ENV.format(hf_home=shlex.quote(hf_home))
    parts = []
    for path in spec_paths:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        parts.append(
            f"{env} python scripts/run_local.py --spec {shlex.quote(path)} --runs-dir {shlex.quote(runs_dir)} "
            f"--provider {shlex.quote(provider)} --usd-per-hour {usd_per_hour} --commit {shlex.quote(commit)} "
            f"> {shlex.quote(runs_dir)}/{stem}.log 2>&1"
        )
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

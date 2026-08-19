"""Docker container control on an operator-owned machine, over SSH.

Backs the dashboard's "ssh" worker slots: each slot is one container of the
worker image on a machine the operator owns (a spare laptop on the LAN, a home
server), started/stopped/probed here. The host string is handed to `ssh`
verbatim, so "user@host" and ~/.ssh/config aliases both work; key-based auth
must already be set up (BatchMode forbids prompts, so a missing key fails fast
instead of hanging the dashboard).

An unreachable machine is a normal condition (powered off, lid closed), not an
error: container_state() reports it as the "unreachable" probe state and the
caller decides what to do. Mutating calls raise SshMachineError.

Results come back the same way control goes out -- over this ssh connection,
read out of the container with docker exec (see read_from_container). A worker
on an operator's own machine therefore needs no bucket and no inbound network
path to the controller.

Stopping is not the only way to idle a container: pause/unpause suspend and
resume its processes in place, keeping the unpacked bundle and the in-flight
work. That is what the dashboard uses for a scheduler gate, which parks a
worker many times an hour (docker stop, then a start that re-runs the image's
bootstrap, costs a minute of every cycle and discards the chunk in flight).
"""

import shlex
import subprocess
from pathlib import Path

# ConnectTimeout bounds how long an unreachable host can stall a probe;
# ControlMaster/ControlPersist multiplex every call onto one shared connection,
# so the dashboard's frequent status probes cost a round-trip, not a handshake.
_SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/scz-ssh-%C",
    "-o", "ControlPersist=60",
]  # fmt: skip

# ssh(1) reserves exit status 255 for its own failures (unreachable host, auth
# refused); anything else is the remote command's own status.
_SSH_FAILED = 255

_PROBE_TIMEOUT = 15
# Long enough for `docker stop`'s in-container SIGTERM grace (below) plus the
# worker's final uploads.
_MUTATE_TIMEOUT = 90

# A first pull of the worker image moves a gigabyte or so of NVIDIA runtime
# over a home connection.
_PULL_TIMEOUT = 1800

# How `docker cp` says the path is not in the container, which is a legitimate
# empty answer rather than a failure. Two spellings because the message was
# reworded: the first is what Docker 29 emits (checked against the fleet), the
# second what older versions did. Matching too narrowly here would turn "this
# worker produced nothing" into an error that blocks its replacement forever.
_PATH_NOT_IN_CONTAINER = ("Could not find the file", "No such container:path")

# Sending a file into a container. Long because what goes this way is a model
# (tens of megabytes) over a home network, and a push that gives up leaves the
# slot with nothing to work on until the next pass tries again.
_WRITE_TIMEOUT = 600

# Reading a container's output directory out of it. Generous because it is the
# last look at a container about to be destroyed and there is no way to ask how
# much there is first -- and because failing means the replacement is cancelled
# and the whole copy is repeated next pass.
_COPY_TIMEOUT = 1800


class SshMachineError(Exception):
    """An SSH machine operation failed (host unreachable, docker error)."""


def classify_probe(returncode: int, stdout: str, stderr: str) -> str:
    """Map a `docker inspect -f {{.State.Status}}` result onto the probe
    states: "running" | "paused" | "stopped" | "missing" | "unreachable". Only
    a definite "No such object" counts as missing; any other failure (ssh
    itself, a down docker daemon) is "unreachable", so the caller never
    recreates a container it merely could not see."""
    if returncode != 0:
        if returncode != _SSH_FAILED and "No such object" in stderr:
            return "missing"
        return "unreachable"
    status = stdout.strip()
    if status in ("running", "paused"):
        return status
    return "stopped"  # created / exited / dead: nothing of it is executing


def env_file(env: dict[str, str]) -> str:
    """`env` in docker --env-file format. The format is line-based with no
    quoting, so a newline inside a value would silently truncate it -- refuse
    rather than corrupt."""
    assert not any("\n" in v for v in env.values()), "env values must not contain newlines"
    return "".join(f"{k}={v}\n" for k, v in env.items())


class SshMachine:
    def __init__(self, host: str):
        self.host = host

    def argv(self, command: list[str]) -> list[str]:
        """The local ssh invocation for `command` on the machine. The remote
        side runs a shell, so each argument is quoted for it."""
        remote = " ".join(shlex.quote(a) for a in command)
        return ["ssh", *_SSH_OPTIONS, self.host, remote]

    def _run(self, command: list[str], *, timeout: int, stdin_text: str | None = None):
        try:
            return subprocess.run(
                self.argv(command),
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(command, _SSH_FAILED, "", "ssh timed out")

    def exec_in_container(self, name: str, command: list[str]):
        """Run `command` inside container `name`. Raises SshMachineError on
        failure, like every other mutating call."""
        self._mutate(["docker", "exec", name, *command])

    def _exec(self, argv: list[str], *, timeout: int, doing: str, stdin=None) -> bytes:
        """Run one `docker exec` and return its stdout as bytes. Binary
        throughout: what these calls carry is a tar stream or a model, which
        text decoding would corrupt. `doing` names the operation in the error
        a timeout raises."""
        try:
            res = subprocess.run(self.argv(argv), stdin=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SshMachineError(f"{self.host}: {doing} timed out") from None
        if res.returncode != 0:
            raise SshMachineError(f"{self.host}: {res.stderr.decode(errors='replace').strip()}")
        return res.stdout

    def read_from_container(self, name: str, command: list[str]) -> bytes:
        """Run `command` inside container `name` and return its stdout -- how
        results are read out of a worker (a tar stream)."""
        return self._exec(
            ["docker", "exec", name, *command],
            timeout=_MUTATE_TIMEOUT,
            doing=f"reading from {name}",
        )

    def write_to_container(self, name: str, command: list[str], src: Path):
        """Run `command` inside container `name` with the bytes of `src` on its
        stdin -- how a file goes the other way, from the controller into a
        worker (cloud/ssh_transfer.py's push). Streamed from the file rather
        than read into memory: what travels this way is a model, tens of
        megabytes of it, and the dashboard holds no copy of it."""
        with open(src, "rb") as f:
            self._exec(
                ["docker", "exec", "-i", name, *command],
                timeout=_WRITE_TIMEOUT,
                doing=f"writing to {name}",
                stdin=f,
            )

    def _mutate(self, command: list[str], stdin_text: str | None = None):
        res = self._run(command, timeout=_MUTATE_TIMEOUT, stdin_text=stdin_text)
        if res.returncode != 0:
            what = "unreachable over ssh" if res.returncode == _SSH_FAILED else "command failed"
            raise SshMachineError(f"{self.host}: {what}: {res.stderr.strip() or ' '.join(command)}")

    def container_state(self, name: str) -> str:
        """Probe state of container `name`; see classify_probe."""
        res = self._run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name], timeout=_PROBE_TIMEOUT
        )
        return classify_probe(res.returncode, res.stdout, res.stderr)

    def pull_image(self, image: str):
        """Fetch the newest `image` onto the machine. Containers start with
        --pull=never, so this is where a machine picks up a rebuilt worker
        image -- otherwise every machine would need a hand-run `docker pull`
        after every image change. A private repo needs `docker login` here
        (a one-time setup step), and this reports its failure as any other."""
        res = self._run(["docker", "pull", image], timeout=_PULL_TIMEOUT)
        if res.returncode != 0:
            raise SshMachineError(f"{self.host}: pulling {image} failed: {res.stderr.strip()}")

    def copy_from_container(self, name: str, path: str, dest: Path) -> bool:
        """`path` out of container `name`, as a tar stream whose member names
        are relative to its parent. Unlike exec, this works on a container that
        is stopped -- the only way to read what a worker flushed on its way
        down. A path the container does not have is empty, not an error: a
        worker that died before producing anything never made its output
        directory, and that must not be mistaken for a failure to read it.

        Anything else raises, which is the safe direction: the caller sweeps a
        container in order to destroy it, so a read that failed for a reason
        nobody recognised must stop that, not look like "nothing there".

        The stream is written to `dest` rather than returned, because its size
        is the container's whole output directory and holding that in the
        dashboard's memory is not something this can promise about a machine it
        does not control.

        Uncompressed, unlike a collection: piping through gzip would put the
        pipeline's exit status in gzip's hands rather than docker's, and gzip
        exits happily on the empty stream a failed `docker cp` hands it -- so a
        real error would arrive here as "nothing there", which is the one
        answer that lets the caller destroy the container. `set -o pipefail`
        would settle it, but the fleet's /bin/sh is dash, which has no such
        option. What is not compressed is one worker's dying gasp; see
        ssh_transfer.sweep_stopped. Returns whether anything was written."""
        with open(dest, "wb") as out:
            try:
                res = subprocess.run(
                    self.argv(["docker", "cp", f"{name}:{path}", "-"]),
                    stdout=out,
                    stderr=subprocess.PIPE,
                    timeout=_COPY_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                raise SshMachineError(
                    f"{self.host}: copying {path} from {name} timed out"
                ) from None
        if res.returncode == 0:
            return dest.stat().st_size > 0
        stderr = res.stderr.decode(errors="replace")
        if any(phrase in stderr for phrase in _PATH_NOT_IN_CONTAINER):
            return False
        raise SshMachineError(f"{self.host}: {stderr.strip()}")

    def run_container(self, name: str, image: str, env: dict[str, str], *, gpus: bool = False):
        """Create + start container `name` from `image`. The environment
        (which includes bucket credentials) travels on the ssh pipe as an
        --env-file rather than on the remote command line, where it would be
        visible in the machine's process list. --pull=never keeps a missing
        image an instant, actionable error instead of a multi-minute pull
        blocking the dashboard: pulling is a one-time manual setup step (the
        image repo is private, so it needs a docker login anyway).

        `gpus` gives the container the machine's GPUs, for a role that runs
        one (match eval plays a neural agent). It needs the NVIDIA container
        toolkit installed there; without it `docker run` fails immediately and
        says so, which is the honest answer for a machine that cannot serve
        the role."""
        self._mutate(
            [
                "docker",
                "run",
                "--detach",
                "--pull=never",
                *(["--gpus", "all"] if gpus else []),
                "--name",
                name,
                "--env-file",
                "/dev/stdin",
                image,
            ],
            stdin_text=env_file(env),
        )

    def container_exit(self, name: str) -> str:
        """Why container `name` is not running: its exit code and the last
        thing it said, as one line. Empty when the container is gone or the
        machine cannot be reached -- a slot's failure reason is a nicety, and
        never worth failing a status pass over."""
        res = self._run(
            [
                "sh",
                "-c",
                f"printf 'exit %s: ' \"$(docker inspect -f '{{{{.State.ExitCode}}}}' {name})\"; "
                f"docker logs --tail 20 {name} 2>&1 | grep -v '^$' | tail -1",
            ],
            timeout=_PROBE_TIMEOUT,
        )
        return res.stdout.strip() if res.returncode == 0 else ""

    def start_container(self, name: str):
        self._mutate(["docker", "start", name])

    def pause_container(self, name: str):
        """Suspend the container's processes (SIGSTOP-like, via the freezer
        cgroup). Nothing is lost and nothing restarts on resume -- the point of
        using this for a gate rather than stop/start."""
        self._mutate(["docker", "pause", name])

    def unpause_container(self, name: str):
        self._mutate(["docker", "unpause", name])

    def stop_container(self, name: str):
        # SIGTERM with a grace period long enough to flush completed output
        # (the worker loses at most its in-flight cycle), then SIGKILL.
        self._mutate(["docker", "stop", "-t", "60", name])

    def remove_container(self, name: str):
        self._mutate(["docker", "rm", name])

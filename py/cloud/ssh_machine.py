"""Docker container control on a remote machine, over ssh.

Backs the dashboard's "ssh" worker slots: each slot is one container of the
worker image on either a machine the operator owns (a spare laptop, a home
server) or one the dashboard rented (cloud/providers/). The host string goes
to `ssh` verbatim, so "user@host" and ~/.ssh/config aliases both work.
Key-based auth must already work: BatchMode forbids prompts, so a missing key
fails fast instead of hanging the dashboard.

An unreachable machine (powered off, lid closed) is a normal condition, not an
error: probes report it as "unreachable" and the caller decides what to do.
Mutating calls raise SshMachineError.

Pausing a container (pause_container) suspends its processes in place,
keeping the unpacked bundle and the work in flight. The dashboard's scheduler
gates use it because they park workers many times an hour, and a stop/start
cycle would re-run the image's bootstrap (about a minute) and discard the
chunk in flight each time.
"""

import shlex
import subprocess
from pathlib import Path
from typing import IO

# ConnectTimeout bounds how long an unreachable host can stall a probe.
# ControlMaster/ControlPersist multiplex every call onto one shared connection,
# so the dashboard's frequent probes cost a round trip, not a handshake.
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
# probe()'s exit status for "the readiness marker is not there yet".
_NOT_READY = 3

_PROBE_TIMEOUT = 15
# Covers `docker stop`'s 60 s SIGTERM grace (stop_container) with room to
# spare.
_MUTATE_TIMEOUT = 90

# Commands that read from the machine without changing it: a docker exec
# reading a worker's listings or result archives (cloud/ssh_transfer.py), and
# detect_arch's throwaway container. Above ssh_transfer's
# COLLECT_TIMEOUT_SECONDS, so an overrunning in-container tar hits its own
# limit first.
_READ_TIMEOUT = 90

# A first pull of the worker image moves a gigabyte or so of NVIDIA runtime
# over a home connection.
_PULL_TIMEOUT = 1800

# How `docker cp` reports a path the container lacks: a legitimate empty
# answer, not a failure. Docker 29 emits the first form, older versions the
# second. Missing a spelling would turn "this worker produced nothing" into an
# error that blocks the container's replacement indefinitely.
_PATH_NOT_IN_CONTAINER = ("Could not find the file", "No such container:path")

# Sending a file into a container: a model of tens of megabytes, possibly over
# a home network. A push that gives up leaves the slot idle until the next
# pass retries, so the limit is generous.
_WRITE_TIMEOUT = 600

# Copying a stopped container's output out before it is destroyed. Generous
# because the size cannot be asked first, and a failure cancels the
# replacement and repeats the whole copy next pass.
_COPY_TIMEOUT = 1800


class SshMachineError(Exception):
    """An SSH machine operation failed (host unreachable, docker error)."""


def classify_probe(returncode: int, stdout: str, stderr: str) -> str:
    """Map a `docker inspect -f {{.State.Status}}` result to a probe state:
    "running" | "paused" | "stopped" | "missing" | "unreachable". Only a
    definite "no such object" counts as missing. Any other failure (ssh
    itself, a down docker daemon) is "unreachable", so the caller never
    recreates a container it merely could not see.

    The match is case-insensitive because Docker 28 prints "Error: No such
    object" and Docker 29 "error: no such object". Misreading a missing
    container as unreachable is costly: removal refuses an unreachable slot,
    and the host's shared reachability cache then stalls every other slot on
    that host."""
    if returncode != 0:
        if returncode != _SSH_FAILED and "no such object" in stderr.lower():
            return "missing"
        return "unreachable"
    status = stdout.strip()
    if status in ("running", "paused"):
        return status
    return "stopped"  # created / exited / dead: nothing of it is executing


def env_file(env: dict[str, str]) -> str:
    """`env` in docker --env-file format. The format has no quoting, so a
    value containing a newline is refused rather than silently truncated."""
    assert not any("\n" in v for v in env.values()), "env values must not contain newlines"
    return "".join(f"{k}={v}\n" for k, v in env.items())


class SshMachine:
    def __init__(
        self, host: str, identity_file: str | None = None, known_hosts_file: str | None = None
    ):
        """`host` goes to ssh verbatim. A rented machine passes its own key
        and known_hosts file: its host key is unknown until first contact, and
        providers reuse addresses, so entries in the shared file would go
        stale. An operator's own machine leaves both None and uses the dev
        container's ssh defaults."""
        self.host = host
        self.identity_file = identity_file
        self.known_hosts_file = known_hosts_file

    def argv(self, command: list[str]) -> list[str]:
        """The local ssh argv that runs `command` on the machine. ssh hands
        the remote side one shell string, so each argument is quoted."""
        remote = " ".join(shlex.quote(a) for a in command)
        options = list(_SSH_OPTIONS)
        if self.identity_file:
            options += ["-i", self.identity_file, "-o", "IdentitiesOnly=yes"]
        if self.known_hosts_file:
            options += [
                "-o", f"UserKnownHostsFile={self.known_hosts_file}",
                "-o", "StrictHostKeyChecking=accept-new",
            ]  # fmt: skip
        return ["ssh", *options, self.host, remote]

    def probe(self, ready_file: str | None = None) -> str:
        """Whether the machine can host containers right now: "up" (ssh
        answers and Docker serves), "no docker" (ssh answers but Docker does
        not: not installed, or the user is not in the docker group), or
        "unreachable".

        A rented machine passes `ready_file`, which its first-boot script
        writes after pulling the worker images. sshd comes up well before
        that, so until the file exists the answer is "preparing".
        """
        if ready_file is None:
            command = ["docker", "info", "--format", "{{.ServerVersion}}"]
        else:
            command = [
                "sh",
                "-c",
                f"test -f {shlex.quote(ready_file)} || exit {_NOT_READY}; "
                "docker info --format '{{.ServerVersion}}'",
            ]
        res = self._run(command, timeout=_PROBE_TIMEOUT)
        if res.returncode == _SSH_FAILED:
            return "unreachable"
        if res.returncode == _NOT_READY:
            return "preparing"
        return "up" if res.returncode == 0 else "no docker"

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
        """Run `command` inside container `name`; raises SshMachineError on
        failure."""
        self._mutate(["docker", "exec", name, *command])

    def _exec(
        self, argv: list[str], *, timeout: int, doing: str, stdin: IO[bytes] | None = None
    ) -> bytes:
        """Run one remote command and return its stdout. Binary throughout,
        since it carries tar streams and models. `doing` describes the
        operation in the timeout error."""
        try:
            res = subprocess.run(self.argv(argv), stdin=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SshMachineError(f"{self.host}: {doing} timed out") from None
        if res.returncode != 0:
            raise SshMachineError(f"{self.host}: {res.stderr.decode(errors='replace').strip()}")
        return res.stdout

    def read_from_container(self, name: str, command: list[str]) -> bytes:
        """Run `command` inside container `name` and return its stdout; how
        cloud/ssh_transfer.py reads results out of a worker."""
        return self._exec(
            ["docker", "exec", name, *command],
            timeout=_READ_TIMEOUT,
            doing=f"reading from {name}",
        )

    def write_to_container(self, name: str, command: list[str], src: Path):
        """Run `command` inside container `name` with `src` streamed to its
        stdin; how cloud/ssh_transfer.py pushes a file into a worker. Streamed
        rather than loaded, since it is typically a model of tens of
        megabytes."""
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

    def detect_arch(self, image: str) -> str:
        """The machine's CPU microarchitecture as a GCC -march value. It asks
        the compiler inside the worker `image`, as the container's bootstrap
        does at startup, so the answer names the tarball the bootstrap will
        pick. The image must already be on the machine (pull_image)."""
        res = self._run(
            ["docker", "run", "--rm", "--pull=never", "--entrypoint", "g++", image,
             "-march=native", "-Q", "--help=target"],
            timeout=_READ_TIMEOUT,
        )  # fmt: skip
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("-march="):
                arch = line.split("=", 1)[1].strip()
                if arch and arch != "native":
                    return arch
        raise SshMachineError(f"{self.host}: could not detect its arch: {res.stderr.strip()}")

    def pull_image(self, image: str):
        """Fetch the newest `image` onto the machine. Containers start with
        --pull=never, so this is how a machine picks up a rebuilt worker
        image; the dashboard calls it before creating each container. The
        repo is private, so the machine needs a one-time `docker login` (a
        rented machine's first-boot script does it)."""
        res = self._run(["docker", "pull", image], timeout=_PULL_TIMEOUT)
        if res.returncode != 0:
            raise SshMachineError(f"{self.host}: pulling {image} failed: {res.stderr.strip()}")

    def copy_from_container(self, name: str, path: str, dest: Path) -> bool:
        """Copy `path` out of container `name` into `dest` as a tar whose
        member names are relative to the path's parent. Returns whether
        anything was written. Unlike exec, this works on a stopped container,
        which makes it the way to read what a worker flushed as it stopped.

        A path the container lacks returns False: a worker that died before
        producing anything never created its output directory. Any other
        failure raises. The caller is sweeping the container in order to
        destroy it, so an unrecognized failure must stop that rather than
        read as "nothing there".

        The stream goes to a file rather than memory because its size is
        whatever the remote container holds. It is not compressed: piping
        through gzip would make the pipeline's exit status gzip's, and gzip
        succeeds on the empty stream a failed `docker cp` produces, turning an
        error into "nothing there". `set -o pipefail` would fix that, but the
        machines' /bin/sh is dash, which lacks it."""
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
        """Create and start container `name` from `image`. The environment,
        which includes bucket credentials, travels over the ssh pipe as an
        --env-file rather than on the remote command line, where the machine's
        process list would show it. --pull=never makes a missing image an
        immediate error rather than a long pull under the dashboard; pulling
        is pull_image's job.

        `gpus` gives the container the machine's GPUs, for roles that use one
        (e.g. match eval's neural agents). Without the NVIDIA container toolkit
        on the machine, `docker run` fails immediately with a clear error."""
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
        """Why container `name` is not running: its exit code and last log
        line, as one line. Empty when the container is gone or the machine is
        unreachable; a failure reason is informational and never worth
        failing a status pass over."""
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
        """Freeze the container's processes (via the freezer cgroup). Nothing
        is lost, and unpause resumes them where they were."""
        self._mutate(["docker", "pause", name])

    def unpause_container(self, name: str):
        self._mutate(["docker", "unpause", name])

    def stop_container(self, name: str):
        # SIGTERM, then SIGKILL after 60 s: long enough for the worker to flush
        # completed output.
        self._mutate(["docker", "stop", "-t", "60", name])

    def remove_container(self, name: str):
        self._mutate(["docker", "rm", name])

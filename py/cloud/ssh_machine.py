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

    def read_from_container(self, name: str, command: list[str]) -> bytes:
        """Run `command` inside container `name` and return its stdout as
        bytes -- how results are read out of a worker (a tar stream). Text
        decoding would corrupt them, so this path stays binary."""
        try:
            res = subprocess.run(
                self.argv(["docker", "exec", name, *command]),
                capture_output=True,
                timeout=_MUTATE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise SshMachineError(f"{self.host}: reading from {name} timed out") from None
        if res.returncode != 0:
            raise SshMachineError(f"{self.host}: {res.stderr.decode(errors='replace').strip()}")
        return res.stdout

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

    def run_container(self, name: str, image: str, env: dict[str, str]):
        """Create + start container `name` from `image`. The environment
        (which includes bucket credentials) travels on the ssh pipe as an
        --env-file rather than on the remote command line, where it would be
        visible in the machine's process list. --pull=never keeps a missing
        image an instant, actionable error instead of a multi-minute pull
        blocking the dashboard: pulling is a one-time manual setup step (the
        image repo is private, so it needs a docker login anyway)."""
        self._mutate(
            [
                "docker",
                "run",
                "--detach",
                "--pull=never",
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

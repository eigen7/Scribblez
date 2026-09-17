"""What the dashboard asks of a cloud provider: a catalog of machine types,
and launch / describe / stop / start / terminate of instances it tags as its
own. A provider is one module implementing `Provider` plus a section in the
credentials file; the dashboard names none of them outside the machine forms.

An instance here is a machine in the sense of docs/cloud_machines_plan.md:
ssh-reachable, running Docker, hosting the existing ssh-kind slots. The
provider's job ends at the address; everything after is the ssh kind's.
"""

from dataclasses import dataclass
from typing import Protocol


class ProviderError(Exception):
    """A provider call failed. `detail` is the provider's own words; the
    message is the operator-facing sentence (see Provider.refusal)."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail or message


@dataclass(frozen=True)
class MachineType:
    """One catalog row: what the operator picks from."""

    id: str  # the provider's instance type id
    vcpus: int
    gpu_count: int
    gpu: str  # "" for a CPU-only type
    arch: str  # the bundle arch its CPU family builds for (py/build.py SUPPORTED_ARCHS)
    cost_per_hr: float  # on-demand list price, dated in the catalog module


@dataclass(frozen=True)
class LaunchRequest:
    type_id: str
    # The value of the ownership tag: "<workload>/<tag>/<machine name>", what
    # the dashboard recognizes an instance by. An instance tagged with a value
    # no task's machines carry is an orphan.
    owner: str
    # Rent spare capacity at its market rate, with the provider free to
    # interrupt (stop) the instance when it wants the capacity back.
    spot: bool = False


@dataclass
class Instance:
    """An instance as the provider describes it. `state` is the provider's own
    lifecycle word, normalized: pending | running | stopping | stopped |
    terminated."""

    id: str
    state: str
    type_id: str
    owner: str | None  # the ownership tag, None on an instance that lacks it
    address: str | None  # public address while it has one
    launched_at: float | None
    spot: bool = False
    cost_per_hr: float | None = None  # a spot instance's rate at launch; None: the catalog's


class Provider(Protocol):
    name: str
    ssh_user: str  # the login the image gives ssh
    identity_file: str  # the private key every instance is launched with
    ready_file: str  # the marker the first-boot script writes last

    def catalog(self) -> list[MachineType]: ...

    def spot_prices(self) -> dict[str, float]: ...  # current spot rate by catalog type id

    def account(self) -> str: ...  # who and where machines are rented as, for the form

    def prepare(self): ...  # the account-side one-time setup, idempotent

    def launch(self, request: LaunchRequest) -> Instance: ...

    def describe(self) -> dict[str, Instance]: ...  # every instance this provider tagged, by id

    def stop(self, instance_id: str): ...

    def start(self, instance_id: str): ...

    def terminate(self, instance_id: str): ...

    def refusal(self, error: ProviderError, type_id: str) -> str: ...

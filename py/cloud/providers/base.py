"""The interface the dashboard needs from a cloud provider.

A provider offers a catalog of machine types and can launch, describe, stop,
start and terminate the instances it has tagged as the dashboard's. Adding a
provider means one module implementing `Provider` plus a section in the
credentials file; outside the machine forms, the dashboard never names a
specific provider.

The provider's job ends once an instance has an address. From there the
instance is an ordinary ssh machine (cloud/ssh_machine.py) hosting ssh-kind
worker slots.
"""

from dataclasses import dataclass
from typing import Protocol


class ProviderError(Exception):
    """A provider call failed. The message is the provider's error code;
    `detail` is its own explanation. Provider.refusal turns both into a
    sentence for the operator."""

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
    arch: str  # GCC -march value for its CPU family; selects the bundle it runs
    cost_per_hr: float  # on-demand list price, dated in the provider module


@dataclass(frozen=True)
class LaunchRequest:
    type_id: str
    # The ownership tag's value, "<workload>/<tag>/<machine name>": how the
    # dashboard recognizes its instances. One whose value matches no task's
    # machine is an orphan.
    owner: str
    # Rent spare capacity at the market rate. The provider may interrupt
    # (stop) the instance when it wants the capacity back.
    spot: bool = False


@dataclass
class Instance:
    """An instance as the provider describes it. `state` is normalized to one
    of pending | running | stopping | stopped | terminated."""

    id: str
    state: str
    type_id: str
    owner: str | None  # the ownership tag, None on an instance that lacks it
    address: str | None  # public address while it has one
    launched_at: float | None
    spot: bool = False
    cost_per_hr: float | None = None  # spot rate at launch; None means the catalog price


class Provider(Protocol):
    name: str
    ssh_user: str  # the login the machine image provides
    identity_file: str  # the private key every instance is launched with
    ready_file: str  # written last by the first-boot script; see SshMachine.probe

    def catalog(self) -> list[MachineType]: ...

    def spot_prices(self) -> dict[str, float]: ...  # current spot rate by catalog type id

    def account(self) -> str: ...  # which account and region, for display in the form

    def prepare(self): ...  # one-time account-side setup; idempotent

    def launch(self, request: LaunchRequest) -> Instance: ...

    def describe(self) -> dict[str, Instance]: ...  # every instance this provider tagged, by id

    def stop(self, instance_id: str): ...

    def start(self, instance_id: str): ...

    def terminate(self, instance_id: str): ...

    def refusal(self, error: ProviderError, type_id: str) -> str: ...

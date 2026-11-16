"""What a worker on the engine image imports: the entrypoint, with the whole
workload registry behind it, and each of that image's roles' runner and
deps. The image has the standard library and numpy
(docker-setup/worker/Dockerfile) and the bundle adds py/, so an import of
anything else there stops every worker on the image at start-up, whatever its
workload; torch and onnxruntime come only with the torch image."""

import subprocess
import sys

# Run in a fresh interpreter, the only place a blocked import is not already
# in sys.modules.
SCRIPT = """
import importlib.abc
import sys
from pathlib import Path

import cloud

PY = Path(cloud.__file__).resolve().parents[1]
ON_THE_IMAGE = set(sys.stdlib_module_names) | {"numpy"} | {p.stem for p in PY.iterdir()}


class EngineImageOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] not in ON_THE_IMAGE:
            raise ModuleNotFoundError(f"the engine worker image has no {name}", name=name)


sys.meta_path.insert(0, EngineImageOnly())
import cloud.worker_entrypoint
from cloud.runtime_abi import RUNTIME_ENGINE
from scribblez import workloads

for spec in workloads.WORKLOADS.values():
    for role in spec.roles:
        if role.runtime == RUNTIME_ENGINE:
            for dotted in filter(None, (role.runner, role.deps)):
                workloads.resolve(dotted)
"""


def test_an_engine_image_worker_imports_nothing_the_image_lacks():
    done = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]

"""py/tools/hook_finder.py: words that stay words with the hook letter removed."""

import importlib.util

from scribblez.paths import REPO_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "hook_finder", REPO_ROOT / "py" / "tools" / "hook_finder.py"
)
hook_finder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hook_finder)


def test_front_and_back_hooks():
    words = ["SEA", "SEAS", "AT", "SAT", "ATS", "CAT", "CATS", "SOS"]
    # SEAS: "EAS" is no word but "SEA" is, so the back strip must be tried too.
    assert hook_finder.find_hooks(words, "S") == ["CATS", "SEAS", "ATS", "SAT"]

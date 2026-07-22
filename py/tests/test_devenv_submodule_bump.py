"""Tests for the shared submodule-bump logic (submodule_bump.py) and the
[submodules] config knob (config.py).

Pure config parsing, the lenient mode reader, and the devenv.toml write-back --
no git, no Gitea. The bump offer/commit paths that need real repos are exercised
in test_devenv_publish.py (publish side) and test_devenv_submodule_guard.py
(hook side).
"""

import sys
import tomllib
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import submodule_bump  # noqa: E402
from submodules.devenv_utils.config import Submodules, load_config  # noqa: E402


def write_toml(tmp_path: Path, body: str) -> Path:
    (tmp_path / "devenv.toml").write_text(body)
    return tmp_path


# ---- [submodules] pull_update parsing ------------------------------------


def test_pull_update_defaults_to_prompt(tmp_path: Path):
    cfg = load_config(write_toml(tmp_path, 'name = "proj"\n'))
    assert cfg.submodules == Submodules(pull_update="prompt")


@pytest.mark.parametrize("mode", ["prompt", "never", "always"])
def test_pull_update_accepts_each_mode(tmp_path: Path, mode: str):
    cfg = load_config(
        write_toml(tmp_path, f'name = "proj"\n[submodules]\npull_update = "{mode}"\n')
    )
    assert cfg.submodules.pull_update == mode


def test_pull_update_rejects_invalid_value(tmp_path: Path):
    with pytest.raises(ValueError, match="pull_update"):
        load_config(
            write_toml(tmp_path, 'name = "proj"\n[submodules]\npull_update = "sometimes"\n')
        )


# ---- lenient mode reader (used by the hook) ------------------------------


def test_pull_update_mode_reads_configured_value(tmp_path: Path):
    write_toml(tmp_path, 'name = "proj"\n[submodules]\npull_update = "always"\n')
    assert submodule_bump.pull_update_mode(tmp_path) == "always"


def test_pull_update_mode_falls_back_when_file_missing(tmp_path: Path):
    assert submodule_bump.pull_update_mode(tmp_path) == "prompt"


def test_pull_update_mode_falls_back_on_invalid_value(tmp_path: Path):
    write_toml(tmp_path, 'name = "proj"\n[submodules]\npull_update = "nope"\n')
    assert submodule_bump.pull_update_mode(tmp_path) == "prompt"


# ---- devenv.toml write-back ----------------------------------------------


def test_save_never_appends_new_table_not_inside_last_table(tmp_path: Path):
    # A trailing [services] table: a bare appended key would land inside it.
    toml = write_toml(tmp_path, 'name = "proj"\n[services]\nweb = 5173\n') / "devenv.toml"
    submodule_bump.save_pull_update_never(toml)

    parsed = tomllib.loads(toml.read_text())
    assert parsed["submodules"] == {"pull_update": "never"}
    assert parsed["services"] == {"web": 5173}  # not swallowed into [services]


def test_save_never_rewrites_existing_key_in_place(tmp_path: Path):
    toml = (
        write_toml(tmp_path, 'name = "proj"\n[submodules]\npull_update = "prompt"\n')
        / "devenv.toml"
    )
    submodule_bump.save_pull_update_never(toml)
    assert tomllib.loads(toml.read_text())["submodules"] == {"pull_update": "never"}


def test_save_never_inserts_key_under_bare_table(tmp_path: Path):
    toml = write_toml(tmp_path, 'name = "proj"\n[submodules]\n') / "devenv.toml"
    submodule_bump.save_pull_update_never(toml)
    assert tomllib.loads(toml.read_text())["submodules"] == {"pull_update": "never"}


def test_save_never_preserves_other_content(tmp_path: Path):
    body = 'name = "proj"\ndocker_context = "docker-setup"\n\n[services]\nweb = 5173\n'
    toml = write_toml(tmp_path, body) / "devenv.toml"
    submodule_bump.save_pull_update_never(toml)

    text = toml.read_text()
    assert body in text  # every original byte is preserved verbatim
    parsed = tomllib.loads(text)
    assert parsed["name"] == "proj"
    assert parsed["docker_context"] == "docker-setup"
    assert parsed["submodules"]["pull_update"] == "never"

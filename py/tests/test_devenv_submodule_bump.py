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


# ---- devenv.local.toml overlay -------------------------------------------


def test_local_overlay_replaces_tracked_table(tmp_path: Path):
    write_toml(tmp_path, 'name = "proj"\n[submodules]\npull_update = "prompt"\n')
    (tmp_path / "devenv.local.toml").write_text('[submodules]\npull_update = "never"\n')
    assert load_config(tmp_path).submodules.pull_update == "never"


def test_local_overlay_adds_local_only_keys(tmp_path: Path):
    write_toml(tmp_path, 'name = "proj"\n')
    (tmp_path / "devenv.local.toml").write_text('[submodules]\npull_update = "always"\n')
    assert load_config(tmp_path).submodules.pull_update == "always"


def test_absent_local_file_leaves_tracked_config(tmp_path: Path):
    write_toml(tmp_path, 'name = "proj"\n[submodules]\npull_update = "always"\n')
    assert load_config(tmp_path).submodules.pull_update == "always"


# ---- devenv.local.toml write-back ----------------------------------------
#
# save_pull_update_never targets the untracked devenv.local.toml, creating it
# when absent and using the same table-aware rewrite within an existing file.


def test_save_never_creates_local_file_from_scratch(tmp_path: Path):
    local = tmp_path / "devenv.local.toml"
    submodule_bump.save_pull_update_never(local)
    assert local.exists()
    assert tomllib.loads(local.read_text()) == {"submodules": {"pull_update": "never"}}


def test_save_never_appends_new_table_not_inside_last_table(tmp_path: Path):
    # A trailing [services] table: a bare appended key would land inside it.
    local = tmp_path / "devenv.local.toml"
    local.write_text('name = "proj"\n[services]\nweb = 5173\n')
    submodule_bump.save_pull_update_never(local)

    parsed = tomllib.loads(local.read_text())
    assert parsed["submodules"] == {"pull_update": "never"}
    assert parsed["services"] == {"web": 5173}  # not swallowed into [services]


def test_save_never_rewrites_existing_key_in_place(tmp_path: Path):
    local = tmp_path / "devenv.local.toml"
    local.write_text('[submodules]\npull_update = "prompt"\n')
    submodule_bump.save_pull_update_never(local)
    assert tomllib.loads(local.read_text())["submodules"] == {"pull_update": "never"}


def test_save_never_inserts_key_under_bare_table(tmp_path: Path):
    local = tmp_path / "devenv.local.toml"
    local.write_text('name = "proj"\n[submodules]\n')
    submodule_bump.save_pull_update_never(local)
    assert tomllib.loads(local.read_text())["submodules"] == {"pull_update": "never"}


def test_save_never_preserves_other_content(tmp_path: Path):
    body = 'name = "proj"\ndocker_context = "docker-setup"\n\n[services]\nweb = 5173\n'
    local = tmp_path / "devenv.local.toml"
    local.write_text(body)
    submodule_bump.save_pull_update_never(local)

    text = local.read_text()
    assert body in text  # every original byte is preserved verbatim
    parsed = tomllib.loads(text)
    assert parsed["name"] == "proj"
    assert parsed["docker_context"] == "docker-setup"
    assert parsed["submodules"]["pull_update"] == "never"

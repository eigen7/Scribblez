#!/usr/bin/env python3
"""Interactive first-time setup for Scribblez.

Run this *outside* the Docker container. It:
  1. Picks a persistent host directory ("mount dir") to be bind-mounted into
     the container at /workspace/mount. Build artifacts that need to outlive a
     single container, plus large data files (Macondo, lexica), live there.
  2. Verifies you can run `docker` without sudo.
  3. Downloads .kwg lexicon files from the public Woogles/liwords repo into
     <mount>/lexica/, and symlinks them into Macondo's own data dir so the
     macondo subprocess can find them too. The KWG files are not redistributed
     by Scribblez; we just automate the same fetch-from-upstream the user
     would do by hand. It also copies the repo's committed phony lexica
     (phonies/*.kwg) into <mount>/lexica/.
  4. Writes a per-container VS Code config so that "Dev Containers: Attach
     to Running Container" connects as devuser instead of root.
  5. Builds the Docker image, then validates GPU access inside Docker.

The generic steps live in `submodules/devenv_utils`; this script only adds the
Scribblez-specific lexica step and wires the steps together.

The Macondo checkout and binary are managed by py/build.py, not this wizard.

Re-run the wizard any time you want to install additional lexica, refresh the
VS Code attach config, or rebuild the image.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setup_common import (
    DEFAULT_LEXICA,
    LIWORDS_KWG_URL_TEMPLATE,
    make_config,
)
from submodules.devenv_utils import (
    SetupException,
    SetupWizardTool,
    download,
    in_docker_container,
    print_green,
    print_red,
)

PHONY_LEXICA_DIR = Path(__file__).resolve().parent / "phonies"

# Mirrors the TEMPLATE in py/cloud/credentials.py. Duplicated rather than
# imported: setup_wizard.py runs on the host, where py/ is not on the import
# path (it only lands on PYTHONPATH inside the Docker container), so the
# host-side wizard can't import the container-side cloud.credentials module.
CLOUD_CREDENTIALS_TEMPLATE = {
    "runpod": {
        "api_key": "FILL_ME",
        "container_registry_auth_id": "FILL_ME",
    },
    "registry": {
        "worker_image": "FILL_ME",
    },
    "r2": {
        "account_id": "FILL_ME",
        "access_key_id": "FILL_ME",
        "secret_access_key": "FILL_ME",
        "bucket": "scribblez",
    },
}


class ScribblezSetupWizard(SetupWizardTool):
    """Scribblez setup: the generic steps plus lexicon installation."""

    def setup_git_config(self):
        """Apply the git settings that keep the devenv_utils submodule in sync.

        - submodule.recurse=true: `git pull` / `git checkout` update the
          submodule working tree to match the commit the superproject records,
          so a checkout can't silently go stale.
        - push.recurseSubmodules=check: git refuses to push a commit whose
          submodule pointer references a commit absent from the submodule's
          remote, which would break every other clone.

        Also clears core.hooksPath: this repo wires no custom git hooks, so
        git should use the default .git/hooks.
        """
        repo_root = self.config.repo_root
        for key, value in [
            ("submodule.recurse", "true"),
            ("push.recurseSubmodules", "check"),
        ]:
            subprocess.run(["git", "config", key, value], cwd=repo_root, check=True)
        subprocess.run(["git", "config", "--unset", "core.hooksPath"], cwd=repo_root, check=False)
        print_green("Configured git for the devenv_utils submodule.")

    def installed_lexica(self) -> list[str]:
        lex_dir = self.mount_dir / "lexica"
        if not lex_dir.is_dir():
            return []
        return sorted(p.stem for p in lex_dir.glob("*.kwg"))

    def setup_lexica(self):
        """List installed lexica, prompt for additional ones, fetch them.

        Only populates <mount>/lexica/. The Macondo checkout's own
        data/lexica/gaddag/ symlinks are created by py/build.py *after* it clones
        Macondo -- pre-creating <mount>/macondo/ here would break that clone.
        """
        lex_dir = self.mount_dir / "lexica"
        lex_dir.mkdir(parents=True, exist_ok=True)

        have = self.installed_lexica()
        if have:
            print(f"Already installed lexica ({len(have)}): {', '.join(have)}")
        else:
            print("No lexica installed yet.")

        proposed = [name for name in DEFAULT_LEXICA if name not in have]
        default_csv = ",".join(proposed) if proposed else ""

        prompt = (
            "Lexica to install (comma-separated names; blank to skip)"
            f"{f' [{default_csv}]' if default_csv else ''}: "
        )
        ans = input(prompt).strip()
        if not ans:
            ans = default_csv
        if not ans:
            print("Skipping lexicon install.")
            return

        requested = [s.strip() for s in ans.split(",") if s.strip()]
        failed = []
        for name in requested:
            if name in self.installed_lexica():
                print(f"  {name}: already installed; skipping.")
                continue
            url = LIWORDS_KWG_URL_TEMPLATE.format(name=name)
            print(f"Fetching {name} ...")
            if not download(url, lex_dir / f"{name}.kwg"):
                print_red(f"  Failed to download {name}.")
                failed.append(name)

        print()
        have = self.installed_lexica()
        print_green(f"Installed lexica ({len(have)}): {', '.join(have) or '(none)'}")
        if failed:
            print_red(f"Failed: {', '.join(failed)}")

    def setup_cloud_credentials(self):
        """Write a placeholder cloud credentials file if one isn't there yet.

        Never overwrites an existing file, so a real, filled-in credentials.json
        is left untouched on re-runs. Fill in the FILL_ME values by hand, then
        validate with py/scripts/cloud_check_credentials.py.
        """
        path = self.mount_dir / "cloud" / "credentials.json"
        if path.exists():
            print_green(f"Cloud credentials already present at {path}.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(CLOUD_CREDENTIALS_TEMPLATE, indent=2) + "\n")
        path.chmod(0o600)
        print_green(f"Wrote placeholder cloud credentials to {path}.")
        print("Fill in each FILL_ME value, then verify with:")
        print("    ./py/scripts/cloud_check_credentials.py")

    def install_phony_lexica(self):
        """Copy the repo's committed phony lexica into <mount>/lexica/.

        Unlike the real lexica (downloaded from Woogles), the phony lexica are
        generated by tools/generate_phony_lexicon.py and committed under phonies/,
        so they are copied straight in. Overwrites on each run so the mount mirrors
        the repo. They load like any other lexicon (e.g. for the word-validity toy).
        """
        kwgs = sorted(PHONY_LEXICA_DIR.glob("*.kwg")) if PHONY_LEXICA_DIR.is_dir() else []
        if not kwgs:
            print("No phony lexica in phonies/ to install.")
            return
        lex_dir = self.mount_dir / "lexica"
        lex_dir.mkdir(parents=True, exist_ok=True)
        for src in kwgs:
            shutil.copyfile(src, lex_dir / src.name)
        print_green(f"Installed phony lexica: {', '.join(p.stem for p in kwgs)}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser.parse_args()


def main():
    assert not in_docker_container(), (
        "setup_wizard.py is intended to be run on the host, not inside the container."
    )
    get_args()  # for --help

    config = make_config()
    os.chdir(config.repo_root)
    tool = ScribblezSetupWizard(config)

    print("*" * 78)
    print("Scribblez setup wizard")
    print("*" * 78)

    try:
        tool.rm_target_on_major_bump()
        tool.rule()
        tool.setup_git_config()
        tool.rule()
        tool.setup_mount_dir()
        tool.rule()
        tool.validate_docker_permissions()
        tool.rule()
        tool.setup_lexica()
        tool.rule()
        tool.install_phony_lexica()
        tool.rule()
        tool.setup_cloud_credentials()
        tool.rule()
        tool.setup_vscode_attach_config()
        tool.rule()
        tool.setup_claude_trust()
        tool.rule()
        tool.build_docker_image()
        tool.rule()
        tool.setup_cdi()
        tool.rule()
        tool.validate_nvidia_installation()
        tool.rule()
        tool.commit()
        print_green("Setup complete.")
        print("Next: ./run_docker.py")
    except KeyboardInterrupt:
        print()
        print("Setup wizard interrupted. Re-run when ready.")
        sys.exit(1)
    except SetupException as e:
        for arg in e.args:
            print("*" * 78)
            print(arg)
        sys.exit(1)


if __name__ == "__main__":
    main()

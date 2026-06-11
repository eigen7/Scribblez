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
     would do by hand.
  4. Writes a per-container VS Code config so that "Dev Containers: Attach
     to Running Container" connects as devuser instead of root.
  5. Builds the Docker image, then validates GPU access inside Docker.

The generic steps live in `subtrees/devenv_utils`; this script only adds the
Scribblez-specific lexica step and wires the steps together.

The Macondo checkout and binary are managed by build.py, not this wizard.

Re-run the wizard any time you want to install additional lexica, refresh the
VS Code attach config, or rebuild the image.
"""

import argparse
import os
import sys

from setup_common import DEFAULT_LEXICA, LIWORDS_KWG_URL_TEMPLATE, make_config
from subtrees.devenv_utils import (
    SetupException,
    SetupWizardTool,
    download,
    in_docker_container,
    print_green,
    print_red,
)


class ScribblezSetupWizard(SetupWizardTool):
    """Scribblez setup: the generic steps plus lexicon installation."""

    def installed_lexica(self) -> list[str]:
        lex_dir = self.mount_dir / "lexica"
        if not lex_dir.is_dir():
            return []
        return sorted(p.stem for p in lex_dir.glob("*.kwg"))

    def setup_lexica(self):
        """List installed lexica, prompt for additional ones, fetch them.

        Only populates <mount>/lexica/. The Macondo checkout's own
        data/lexica/gaddag/ symlinks are created by build.py *after* it clones
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser.parse_args()


def main():
    assert not in_docker_container(), \
        "setup_wizard.py is intended to be run on the host, not inside the container."
    get_args()  # for --help

    config = make_config()
    os.chdir(config.repo_root)
    tool = ScribblezSetupWizard(config)

    print("*" * 78)
    print("Scribblez setup wizard")
    print("*" * 78)

    try:
        tool.setup_mount_dir()
        tool.rule()
        tool.validate_docker_permissions()
        tool.rule()
        tool.setup_lexica()
        tool.rule()
        tool.setup_vscode_attach_config()
        tool.rule()
        tool.build_docker_image()
        tool.rule()
        tool.validate_nvidia_installation()
        tool.rule()
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

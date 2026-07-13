#!/usr/bin/env python3
"""Scribblez entry point for the local Gitea stack used for PR review.

The implementation is shared across consumer projects; see
submodules/devenv_utils/gitea_serve.py.
"""

from setup_check import import_setup_common

setup_common = import_setup_common()

from submodules.devenv_utils import gitea_serve  # noqa: E402

if __name__ == "__main__":
    gitea_serve.main(setup_common.make_config())

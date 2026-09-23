"""Argparse help formatter that always shows defaults and keeps the
description's layout.

`argparse.ArgumentDefaultsHelpFormatter` appends "(default: ...)" to existing
help text, so an argument declared without `help=` never shows its default. The
formatter here fills in the default for those arguments too.

It also prints the description as written instead of reflowing it: scripts pass
their module docstring, already wrapped, whose example commands and lists
reflowing would run together.
"""

import argparse


class ArgumentDefaultsHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    def add_argument(self, action):
        if not action.help and action.default is not argparse.SUPPRESS:
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                action.help = "(default: %(default)s)"
        super().add_argument(action)

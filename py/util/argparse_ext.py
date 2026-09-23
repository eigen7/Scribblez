"""Argparse help formatter that always shows defaults.

`argparse.ArgumentDefaultsHelpFormatter` appends "(default: ...)" to existing
help text, so an argument declared without `help=` never shows its default. The
formatter here fills in the default for those arguments too.
"""

import argparse


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_argument(self, action):
        if not action.help and action.default is not argparse.SUPPRESS:
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                action.help = "(default: %(default)s)"
        super().add_argument(action)

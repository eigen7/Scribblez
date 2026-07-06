"""Argparse help formatters.

`argparse.ArgumentDefaultsHelpFormatter` only appends "(default: ...)" to an
argument's help text. If an argument has no `help=` at all, argparse skips the
help column entirely for that argument, so the default is never shown. This
module provides a drop-in replacement that shows the default in that case too.
"""

import argparse


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like `argparse.ArgumentDefaultsHelpFormatter`, but also shows the
    default value for arguments that have no `help=` text.
    """

    def add_argument(self, action):
        if not action.help and action.default is not argparse.SUPPRESS:
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                action.help = "(default: %(default)s)"
        super().add_argument(action)

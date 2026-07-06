"""Tests for the ArgumentDefaultsHelpFormatter in util/argparse_ext.py."""

import argparse

from util.argparse_ext import ArgumentDefaultsHelpFormatter


def _help_text(**parser_kwargs):
    parser = argparse.ArgumentParser(
        prog="prog", formatter_class=ArgumentDefaultsHelpFormatter, **parser_kwargs
    )
    return parser


def test_shows_default_without_help_string():
    parser = _help_text()
    parser.add_argument("--foo", default=42)
    assert "(default: 42)" in parser.format_help()


def test_appends_default_to_existing_help_string():
    parser = _help_text()
    parser.add_argument("--bar", default=7, help="some help")
    help_text = parser.format_help()
    assert "some help (default: 7)" in help_text


def test_store_true_flag_shows_default():
    parser = _help_text()
    parser.add_argument("--baz", action="store_true")
    assert "(default: False)" in parser.format_help()


def test_required_positional_without_nargs_omits_default():
    # Matches stock ArgumentDefaultsHelpFormatter behavior: a plain
    # required positional has no meaningful "default" to show.
    parser = _help_text()
    parser.add_argument("pos")
    assert "default" not in parser.format_help()


def test_optional_positional_shows_default():
    parser = _help_text()
    parser.add_argument("pos", nargs="?", default="posdefault")
    assert "(default: posdefault)" in parser.format_help()


def test_suppressed_default_not_shown():
    parser = _help_text()
    parser.add_argument("--quiet", default=argparse.SUPPRESS)
    help_text = parser.format_help()
    assert "default" not in help_text

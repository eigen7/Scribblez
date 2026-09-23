#!/usr/bin/env python3
"""List the words that are a front or back hook of another word, for one letter.

Prints every word that starts or ends with LETTER and is still a word with that
letter removed, longest first, then alphabetically. The dictionary is a plain
text word list, one word per line, in upper case:

    py/tools/hook_finder.py words.txt S
"""

import sys
from pathlib import Path


def is_hooked(word: str, letter: str, word_set: set[str]) -> bool:
    """Whether `word` is still a word with `letter` removed from its front or
    its back. A word both starting and ending with `letter` gets both tries."""
    return (word.startswith(letter) and word[1:] in word_set) or (
        word.endswith(letter) and word[:-1] in word_set
    )


def find_hooks(words: list[str], letter: str) -> list[str]:
    """The words of `words` hooked by `letter`, longest first, then alphabetically."""
    word_set = set(words)
    return sorted((w for w in words if is_hooked(w, letter, word_set)), key=lambda w: (-len(w), w))


def main():
    if len(sys.argv) != 3:
        print("Usage: python hook_finder.py <dictionary_file> <letter>")
        sys.exit(1)

    dictionary_file = sys.argv[1]
    letter = sys.argv[2].upper()

    if not letter.isalpha() or len(letter) != 1:
        print("Error: The second argument must be a single letter.")
        sys.exit(1)

    if not Path(dictionary_file).is_file():
        print(f"Error: The dictionary file '{dictionary_file}' does not exist.")
        sys.exit(1)

    with open(dictionary_file) as f:
        words = [line.strip() for line in f if line.strip()]

    for word in find_hooks(words, letter):
        print(word)


if __name__ == "__main__":
    main()

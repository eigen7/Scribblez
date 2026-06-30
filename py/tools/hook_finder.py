#!/usr/bin/env python3
"""
Reads a dictionary file and accepts a single letter.

Outputs all words that starts or ends with that letter, which have the property that if you remove
that letter from the word, you still have a valid word in the dictionary. Sorts the output by word
length (longest to shortest), then alphabetically.
"""

import sys
from pathlib import Path

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
    word_list = [line.strip() for line in f if line.strip()]

word_set = set(word_list)

output = []
for word in word_list:
    if word.startswith(letter):
        subword = word[1:]
    elif word.endswith(letter):
        subword = word[:-1]
    else:
        continue

    if subword in word_set:
        output.append(word)

output.sort(key=lambda x: (-len(x), x))
for word in output:
    print(word)

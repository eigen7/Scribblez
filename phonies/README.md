# Phonies

Generated via:

```
./py/tools/generate_phony_lexicon.py --real-lexicon ../mount/lexica/NWL23.kwg \
    --out-txt phonies/PHONY-NWL23.txt --out-kwg phonies/PHONY-NWL23.kwg
```

The `phonies/*.kwg` files are copied to the mount's `lexica/` directory by
`setup_wizard.py`.

These phonies are useful for neural network experiments.


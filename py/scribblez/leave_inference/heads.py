"""The leave inference model's output heads (model.py), by name. Kept free of
torch so the workload registry can offer them as a param's choices."""

HEAD_TILTED = "tilted"
HEAD_AUTOREGRESSIVE = "autoregressive"
HEADS = (HEAD_TILTED, HEAD_AUTOREGRESSIVE)

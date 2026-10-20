"""A dense, plain-op stand-in for the transfer_test reader's compiled
FlexAttention, so the reader's tests run on the CPU.

torch's FlexAttention has no CPU backward, and compiling it for the CPU costs
more than the tests it serves. use_dense_attention() swaps the reader's two
compiled entry points for eager equivalents; test_transfer_reader's
test_dense_attention_matches_flex_attention holds the swap to the real kernel.
"""

import math

import torch
from scribblez.transfer_test import reader as reader_mod
from torch.nn.attention.flex_attention import create_block_mask


def dense_flex_attention(q, k, v, score_mod, block_mask, enable_gqa):
    """flex_attention's result for the reader's call, from plain
    differentiable ops over the dense (B, heads, S, S) scores: the score_mod
    and the block mask's mask_mod are evaluated on broadcast index grids."""
    batch, heads, s, head_dim = q.shape
    group = heads // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    b = torch.arange(batch).view(-1, 1, 1, 1)
    h = torch.arange(heads).view(1, -1, 1, 1)
    qi = torch.arange(s).view(1, 1, -1, 1)
    kvi = torch.arange(s).view(1, 1, 1, -1)
    scores = score_mod(q @ k.transpose(-2, -1) / math.sqrt(head_dim), b, h, qi, kvi)
    scores = scores.masked_fill(~block_mask.mask_mod(b, h, qi, kvi), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def use_dense_attention(monkeypatch):
    monkeypatch.setattr(reader_mod, "_flex_attention", dense_flex_attention)
    monkeypatch.setattr(reader_mod, "_create_block_mask", create_block_mask)

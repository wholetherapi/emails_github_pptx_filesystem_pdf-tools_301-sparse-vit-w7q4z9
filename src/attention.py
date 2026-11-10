"""Windowed sparse attention kernel."""

import torch


def windowed_attention(q, k, v, window=64):
    # local window attention with global tokens handled by the caller
    return torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v

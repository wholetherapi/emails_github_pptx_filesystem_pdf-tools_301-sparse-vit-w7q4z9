"""Vision transformer integration."""

from src.attention import windowed_attention


class WindowedViT:
    def __init__(self, depth=12, window=64):
        self.depth = depth
        self.window = window

# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # [dim]
        freqs = freqs.repeat_interleave(2, 0)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor):
        """
        Args:
            x: [...]
        Returns:
            pos_enc: [..., dim]
        """
        # [..., dim]
        pos_enc = x.unsqueeze(-1) * self.freqs.view([1] * x.dim() + [-1])
        pos_enc = torch.cat([torch.cos(pos_enc[..., ::2]), torch.sin(pos_enc[..., 1::2])], dim=-1)
        return pos_enc


class PositionalEmbeddingRad(nn.Module):
    def __init__(self, dim: int):
        """
        if dim=2, then just [cos(theta), sin(theta)]
        """
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        # [dim]: [1,1,2,2,4,4,8,8]
        # freqs = 2 ** (torch.arange(0, dim // 2).float())
        # [dim]: [1,1,2,2,3,3,4,4]
        freqs = torch.arange(0, dim // 2) + 1.0
        freqs = freqs.repeat_interleave(2, 0)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor):
        """
        Args:
            x: [...], in rad
        Returns:
            pos_enc: [..., dim]
        """
        # [..., dim]
        pos_enc = x.unsqueeze(-1) * self.freqs.view([1] * x.dim() + [-1])
        pos_enc = torch.cat([torch.cos(pos_enc[..., ::2]), torch.sin(pos_enc[..., 1::2])], dim=-1)
        return pos_enc

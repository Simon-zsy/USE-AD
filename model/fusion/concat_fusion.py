"""Concatenation fusion - baseline strategy."""

import torch
import torch.nn as nn

from .base import BaseFusion, FusionRegistry


@FusionRegistry.register("concat")
class ConcatFusion(BaseFusion):
    """
    Simple concatenation fusion (baseline).

    This is the current implementation behavior - provided for comparison.
    Includes optional LayerNorm for pre-fusion normalization.
    """

    def __init__(
        self,
        ae_dim: int,
        tabpfn_dim: int,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__(ae_dim, tabpfn_dim)
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.norm_ae = nn.LayerNorm(ae_dim)
            self.norm_tab = nn.LayerNorm(tabpfn_dim)

    @property
    def output_dim(self) -> int:
        return self.ae_dim + self.tabpfn_dim

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.use_layer_norm:
            h = self.norm_ae(h)
            t = self.norm_tab(t)
        return torch.cat([h, t], dim=1)

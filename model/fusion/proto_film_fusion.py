"""Prototype-Conditioned FiLM Fusion - ICLAD, 2026."""

import torch
import torch.nn as nn

from .base import BaseFusion, FusionRegistry


@FusionRegistry.register("proto_film")
class ProtoFiLMFusion(BaseFusion):
    """
    Prototype-conditioned FiLM fusion (ICLAD, arXiv 2026).

    Uses labeled anomaly exemplars to derive a conditioning vector:
        delta = mean(emb[anomalies]) - mean(emb[normals])

    Applies Feature-wise Linear Modulation (FiLM) to the AE hidden h:
        h' = (1 + gamma(delta)) * h + beta(delta)

    where gamma, beta are tiny linear maps from tabpfn_dim → ae_dim.
    Both are zero-initialized, so at step 0: h' = h (identity), same as concat.

    The delta vector is fixed (computed once from training data in Stage 2,
    set via set_conditioning()). Only gamma/beta nets are learned — minimal
    parameters (~2 * tabpfn_dim * ae_dim ≈ 2 * 512 * 64 = 65K params).

    Reference:
        "ICLAD: In-Context Learning for Unified Tabular Anomaly Detection
         Across Supervision Regimes", arXiv 2026
    """

    def __init__(
        self,
        ae_dim: int,
        tabpfn_dim: int,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__(ae_dim, tabpfn_dim)
        self._output_dim = ae_dim + tabpfn_dim

        if use_layer_norm:
            self.norm_ae = nn.LayerNorm(ae_dim)
            self.norm_tab = nn.LayerNorm(tabpfn_dim)
        else:
            self.norm_ae = nn.Identity()
            self.norm_tab = nn.Identity()

        # FiLM parameter generators: delta → scale/shift for h
        self.gamma_net = nn.Linear(tabpfn_dim, ae_dim)
        self.beta_net = nn.Linear(tabpfn_dim, ae_dim)

        # Zero-init: identity modulation at step 0
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

        # Conditioning vector (set externally via set_conditioning)
        self.register_buffer("delta", torch.zeros(tabpfn_dim))

    def set_conditioning(self, delta: torch.Tensor) -> None:
        """Set the prototype difference vector (anomaly mean - normal mean).

        Called once after Stage 2 embedding extraction, before training.
        """
        self.delta = delta.detach().to(self.delta.device)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm_ae(h)
        t = self.norm_tab(t)

        delta = self.delta.unsqueeze(0)          # (1, tabpfn_dim)
        gamma = self.gamma_net(delta)            # (1, ae_dim)
        beta = self.beta_net(delta)              # (1, ae_dim)
        h_modulated = (1.0 + gamma) * h + beta  # FiLM: broadcast over batch

        return torch.cat([h_modulated, t], dim=1)

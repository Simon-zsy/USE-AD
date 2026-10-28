"""Dynamic Prototype-Conditioned FiLM Fusion — per-sample soft prototype assignment."""

import torch
import torch.nn as nn

from .base import BaseFusion, FusionRegistry


@FusionRegistry.register("proto_film_dynamic")
class DynamicProtoFiLMFusion(BaseFusion):
    """
    Per-sample dynamic ProtoFiLM fusion.

    Unlike ProtoFiLMFusion which applies a fixed conditioning vector
    delta = mean(anomaly) - mean(normal) uniformly to every sample,
    this module computes a per-sample conditioning vector via soft
    attention over the two prototypes:

        dist_n_i  = ||t_i - normal_proto||²
        dist_a_i  = ||t_i - anomaly_proto||²
        alpha_i   = softmax([-dist_n_i, -dist_a_i])     # (B, 2)
        delta_i   = alpha_i[0] * normal_proto
                  + alpha_i[1] * anomaly_proto           # (B, D)

    A sample close to the anomaly prototype gets conditioning that
    emphasises the anomaly direction; a borderline sample gets a
    mixture.  FiLM modulation is then applied to AE hidden h:

        gamma_i, beta_i = FiLM_net(delta_i)
        h'_i = (1 + gamma_i) * h_i + beta_i

    Zero-initialised FiLM nets ensure identity at step 0 (same as concat).
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

        self.norm_ae  = nn.LayerNorm(ae_dim)   if use_layer_norm else nn.Identity()
        self.norm_tab = nn.LayerNorm(tabpfn_dim) if use_layer_norm else nn.Identity()

        self.gamma_net = nn.Linear(tabpfn_dim, ae_dim)
        self.beta_net  = nn.Linear(tabpfn_dim, ae_dim)
        nn.init.zeros_(self.gamma_net.weight); nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight);  nn.init.zeros_(self.beta_net.bias)

        self.register_buffer("normal_proto",  torch.zeros(tabpfn_dim))
        self.register_buffer("anomaly_proto", torch.zeros(tabpfn_dim))
        self._fitted = False

    def set_prototypes(self, normal_proto: torch.Tensor, anomaly_proto: torch.Tensor) -> None:
        """Set normal and anomaly prototype vectors from training embeddings."""
        self.normal_proto  = normal_proto.detach().to(self.normal_proto.device)
        self.anomaly_proto = anomaly_proto.detach().to(self.anomaly_proto.device)
        self._fitted = True

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm_ae(h)
        t = self.norm_tab(t)

        if not self._fitted:
            return torch.cat([h, t], dim=1)

        n_proto = self.normal_proto.unsqueeze(0)   # (1, D)
        a_proto = self.anomaly_proto.unsqueeze(0)  # (1, D)

        dist_n = ((t - n_proto) ** 2).sum(dim=1, keepdim=True)   # (B, 1)
        dist_a = ((t - a_proto) ** 2).sum(dim=1, keepdim=True)   # (B, 1)

        alpha = torch.softmax(torch.cat([-dist_n, -dist_a], dim=1), dim=1)  # (B, 2)
        delta_i = alpha[:, 0:1] * n_proto + alpha[:, 1:2] * a_proto         # (B, D)

        gamma = self.gamma_net(delta_i)            # (B, ae_dim)
        beta  = self.beta_net(delta_i)             # (B, ae_dim)
        h_mod = (1.0 + gamma) * h + beta

        return torch.cat([h_mod, t], dim=1)

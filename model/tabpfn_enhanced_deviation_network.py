"""TabPFN-Enhanced Deviation Network that integrates TabPFN embeddings."""
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .autoencoder import AutoEncoder
from .fusion import BaseFusion, FusionRegistry
FUSION_STRATEGIES = Literal['concat', 'proto_film', 'proto_film_dynamic']

class TabPFNEnhancedDeviationNetwork(nn.Module):
    """
    TabPFN-Enhanced Deviation Network with pluggable fusion strategies.

    Extracts and fuses three types of embeddings:
    1. Hidden representation (h): AutoEncoder encoder output (64-dim)
    2. Reconstruction residual vector (r): Normalized(input - reconstruction)
    3. TabPFN embedding (t): Deep semantic features from pre-trained TabPFN

    Fusion strategies:
    - concat: Simple concatenation (baseline)
    - glu: Gated Linear Unit
    - gmu: Gated Multimodal Unit
    - grn: Gated Residual Network
    - film: Feature-wise Linear Modulation
    - cross_attention: Cross-attention mechanism
    - dcn: Deep & Cross Network v2

    Anomaly score architecture:
        [r, fused(h,t), e] -> Dense(256, relu) + [e] -> Dense(32, relu) + [e] -> Dense(1, linear)

    Critical: Reconstruction error (e) is concatenated at EVERY layer.

    Args:
        input_dim: Number of input features
        autoencoder: Pre-trained AutoEncoder
        tabpfn_dim: Dimension of TabPFN embeddings
        fusion_strategy: Name of fusion strategy to use (default: "concat")
        fusion_kwargs: Additional kwargs passed to fusion module
    """
    AE_HIDDEN_DIM = 64

    def __init__(self, input_dim: int, autoencoder: AutoEncoder, tabpfn_dim: int, fusion_strategy: FUSION_STRATEGIES='concat', fusion_kwargs: Optional[dict]=None):
        super().__init__()
        self.input_dim = input_dim
        self.tabpfn_dim = tabpfn_dim
        self.fusion_strategy = fusion_strategy
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = True
        fusion_kwargs = fusion_kwargs or {}
        fusion_class = FusionRegistry.get(fusion_strategy)
        self.fusion: BaseFusion = fusion_class(ae_dim=self.AE_HIDDEN_DIM, tabpfn_dim=tabpfn_dim, **fusion_kwargs)
        self.norm_residual = nn.Identity()
        concat_dim = input_dim + self.fusion.output_dim + 1
        self.score_layer1 = nn.Linear(concat_dim, 256)
        self.score_layer2 = nn.Linear(256 + 1, 32)
        self.score_layer3 = nn.Linear(32 + 1, 1)
        self._init_weights()

    def _init_weights(self):
        """Xavier normal initialization for score layers."""
        for module in [self.score_layer1, self.score_layer2, self.score_layer3]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, tabpfn_embedding: torch.Tensor, return_hidden: bool=False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through TabPFN-enhanced deviation network.

        Args:
            x: Input tensor (batch_size, input_dim)
            tabpfn_embedding: TabPFN embedding tensor (batch_size, tabpfn_dim)

        Returns:
            score: Anomaly score (batch_size, 1)
            reconstruction_error: L2 norm of residual (batch_size,)
            residual_normalized: Normalized residual vector (batch_size, input_dim)
        """
        reconstruction, h = self.autoencoder(x)
        residual = reconstruction - x
        reconstruction_error = torch.norm(residual, p=2, dim=1, keepdim=True)
        residual_normalized = residual / (reconstruction_error + 1e-08)
        residual_normalized = self.norm_residual(residual_normalized)
        fused_features = self.fusion(h, tabpfn_embedding)
        features = torch.cat([residual_normalized, fused_features, reconstruction_error], dim=1)
        out = F.relu(self.score_layer1(features))
        out = torch.cat([out, reconstruction_error], dim=1)
        penultimate = F.relu(self.score_layer2(out))
        out = torch.cat([penultimate, reconstruction_error], dim=1)
        score = self.score_layer3(out)
        if return_hidden:
            return (score, reconstruction_error.squeeze(1), residual_normalized, h, penultimate)
        return (score, reconstruction_error.squeeze(1), residual_normalized)

    def predict(self, x: torch.Tensor, tabpfn_embedding: torch.Tensor) -> torch.Tensor:
        """Inference mode: return only anomaly scores."""
        score, _, _ = self.forward(x, tabpfn_embedding)
        return score

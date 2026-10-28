"""Base class for fusion strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn


class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for all fusion strategies.

    All fusion modules must:
    1. Accept AE hidden features (h) and TabPFN embeddings (t)
    2. Return fused features tensor
    3. Report their output dimension via output_dim property
    """

    def __init__(self, ae_dim: int, tabpfn_dim: int):
        """
        Initialize fusion module.

        Args:
            ae_dim: Dimension of AutoEncoder hidden representation (typically 64)
            tabpfn_dim: Dimension of TabPFN embeddings (variable)
        """
        super().__init__()
        self.ae_dim = ae_dim
        self.tabpfn_dim = tabpfn_dim

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension of the fused features."""
        pass

    @abstractmethod
    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Fuse AE hidden representation with TabPFN embeddings.

        Args:
            h: AutoEncoder hidden representation (batch_size, ae_dim)
            t: TabPFN embedding (batch_size, tabpfn_dim)

        Returns:
            Fused features tensor (batch_size, output_dim)
        """
        pass


class FusionRegistry:
    """Registry for fusion strategy classes."""

    _registry: Dict[str, Type[BaseFusion]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a fusion class."""

        def decorator(fusion_class: Type[BaseFusion]) -> Type[BaseFusion]:
            cls._registry[name] = fusion_class
            return fusion_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseFusion]:
        """Get fusion class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown fusion strategy: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list:
        """List all registered fusion strategies."""
        return list(cls._registry.keys())

"""Fusion modules used by the paper and its component ablation."""
from .base import BaseFusion, FusionRegistry
from .concat_fusion import ConcatFusion
from .proto_film_fusion import ProtoFiLMFusion
from .dynamic_proto_film_fusion import DynamicProtoFiLMFusion

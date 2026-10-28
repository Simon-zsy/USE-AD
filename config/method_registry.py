"""Central registry for anomaly detection methods."""
from typing import Dict, Tuple, Type
from pydantic import BaseModel
from methods import BaseMethod, CatBoostConfig, CatBoostMethod, RandomForestConfig, RandomForestMethod, DevNetConfig, DevNetMethod, FEAWADConfig, FEAWADMethod, DeepSADConfig, DeepSADMethod, PReNetConfig, PReNetMethod, TabPFNConfig, TabPFNMethod, TabPFNFEAWADConfig, TabPFNFEAWADMethod

class MethodRegistry:
    """Central registry for all anomaly detection methods."""
    _registry: Dict[str, Tuple[Type[BaseModel], Type[BaseMethod]]] = {'catboost': (CatBoostConfig, CatBoostMethod), 'random_forest': (RandomForestConfig, RandomForestMethod), 'devnet': (DevNetConfig, DevNetMethod), 'feawad': (FEAWADConfig, FEAWADMethod), 'deepsad': (DeepSADConfig, DeepSADMethod), 'prenet': (PReNetConfig, PReNetMethod), 'tabpfn': (TabPFNConfig, TabPFNMethod), 'tabpfn-feawad': (TabPFNFEAWADConfig, TabPFNFEAWADMethod)}

    @classmethod
    def get_config_class(cls, method_name: str) -> Type[BaseModel]:
        """Get config class for a method."""
        if method_name not in cls._registry:
            raise ValueError(f'Unknown method: {method_name}')
        return cls._registry[method_name][0]

    @classmethod
    def get_method_class(cls, method_name: str) -> Type[BaseMethod]:
        """Get method class for a method."""
        if method_name not in cls._registry:
            raise ValueError(f'Unknown method: {method_name}')
        return cls._registry[method_name][1]

    @classmethod
    def create_method(cls, method_name: str, config: BaseModel) -> BaseMethod:
        """Factory method to create a method instance."""
        if method_name not in cls._registry:
            raise ValueError(f'Unknown method: {method_name}')
        _, method_class = cls._registry[method_name]
        return method_class(config)

    @classmethod
    def list_methods(cls) -> list[str]:
        """List all registered method names."""
        return list(cls._registry.keys())

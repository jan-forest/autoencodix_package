from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DefaultConfig:
    """
    Configuration container for models
    """
    model_type: str = 'vanilix'
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    trainer_params: Dict[str, Any] = field(default_factory=dict)
    predictor_params: Dict[str, Any] = field(default_factory=dict)
    visualization_params: Dict[str, Any] = field(default_factory=dict)
    evaluation_params: Dict[str, Any] = field(default_factory=dict)
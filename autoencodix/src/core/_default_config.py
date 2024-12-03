from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DefaultConfig:
    """
    Configuration container for models

    Attributes
    ----------
    model_type : str
        Type of model (e.g., 'vanilix').
    preprocessor_params : dict
        Parameters for the preprocessor.
    trainer_params : dict
        Parameters for the trainer.
    predictor_params : dict
        Parameters for the predictor.
    visualization_params : dict
        Parameters for visualization.
    evaluation_params : dict
        Parameters for evaluation.
    """

    model_type: str = "vanilix"
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    trainer_params: Dict[str, Any] = field(default_factory=dict)
    predictor_params: Dict[str, Any] = field(default_factory=dict)
    visualization_params: Dict[str, Any] = field(default_factory=dict)
    evaluation_params: Dict[str, Any] = field(default_factory=dict)

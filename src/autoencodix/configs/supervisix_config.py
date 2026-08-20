from .varix_config import VarixConfig
from pydantic import Field, model_validator


class SupervisixConfig(VarixConfig):
    """
    A specialized configuration for supervisix,inheriting from DefaultConfig.
    """
    # TODO find sensible defaults for Supervisix

    gamma_class_separation: float = Field(
        default=0.0,
        ge=0, # Given value must be >=
        description="Gamma weighting factor for class-based separation loss in supervisix VAE architecture"
    )

    delta_class_cohesion: float = Field(
        default=0.0,
        ge=0, # Given value must be >=
        description="Delta weighting factor for class-based cohesion loss in supervisix VAE architecture"
    )

    # TODO: Add @model_validator flag?
    @model_validator(mode="after")
    def _check_class_param_is_set(self):
        if self.class_param is None:
            raise ValueError("The class_param parameter must be set for Supervisix losses. " \
            "Please add class_param=<column name> to the config setup attribute" \
            " for a column that contains the class labels in your dataset's metadata.")
        return self

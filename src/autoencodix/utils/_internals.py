"""
Utility to Map Architectures to Trainers.
Only used for training sub datasets in XModalix as of now.
"""

from typing import Dict, Type

from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture
from autoencodix.modeling._ontix_architecture import OntixArchitecture
from autoencodix.modeling._vanillix_architecture import VanillixArchitecture
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.modeling._maskix_architecture import MaskixArchitectureVanilla
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.trainers._ontix_trainer import OntixTrainer
from autoencodix.trainers._maskix_trainer import MaskixTrainer

model_trainer_map: Dict[Type, Type] = {
    VarixArchitecture: GeneralTrainer,
    ImageVAEArchitecture: GeneralTrainer,
    VanillixArchitecture: GeneralTrainer,
    OntixArchitecture: OntixTrainer,
    MaskixArchitectureVanilla: MaskixTrainer,
}

from typing import Dict, Type

from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture
from autoencodix.modeling._ontix_architecture import OntixArchitecture
from autoencodix.modeling._vanillix_architecture import VanillixArchitecture
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.trainers._ontix_trainer import OntixTrainer

model_trainer_map: Dict[Type, Type] = {
    VarixArchitecture: GeneralTrainer,
    ImageVAEArchitecture: GeneralTrainer,
    VanillixArchitecture: GeneralTrainer,
    OntixArchitecture: OntixTrainer
}

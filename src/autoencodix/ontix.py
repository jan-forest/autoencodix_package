from typing import Dict, Optional, Type, Union, Tuple, List
import torch
import numpy as np

from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_evaluator import BaseEvaluator
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.evaluate._general_evaluator import GeneralEvaluator

# from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.modeling._ontix_architecture import OntixArchitecture
from autoencodix.trainers._ontix_trainer import OntixTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig

from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.visualize._general_visualizer import GeneralVisualizer


## Copy from Varix with ontology addition
class Ontix(BasePipeline):
    """Ontix specific version of the BasePipeline class.

    Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.

    This class extends BasePipeline. See the parent class for a full list
    of attributes and methods.

    Additional Attributes:
        _default_config: Is set to OntixConfig here.

    """

    def __init__(
        self,
        ontologies: Union[Tuple, List],  # Addition to Varix, mandotory for Ontix
        sep: Optional[str] = "\t",  # Addition to Varix, optional to read in ontologies
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = OntixTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        model_type: Type[BaseAutoencoder] = OntixArchitecture,
        loss_type: Type[BaseLoss] = VarixLoss,
        preprocessor_type: Type[BasePreprocessor] = GeneralPreprocessor,
        visualizer: Type[BaseVisualizer] = GeneralVisualizer,
        evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        """Initialize Ontix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        See parent class for full list of Arguments.

        Raises:
            TypeError: if ontologies are not a Tuple or List.

        """
        self._default_config = OntixConfig()
        if isinstance(ontologies, tuple):
            self.ontologies = ontologies
        elif isinstance(ontologies, list):
            if sep is None:
                raise ValueError(
                    "If ontologies are provided as a list, the seperator 'sep' cannot be None. "
                )
            ontologies_dict_list = [
                self._read_ont_file(ont_file, sep=sep) for ont_file in ontologies
            ]
            self.ontologies = tuple(ontologies_dict_list)
        else:
            raise TypeError(
                f"Expected ontologies to be of type tuple or list, got {type(ontologies)}."
            )

        super().__init__(
            data=data,
            dataset_type=dataset_type,
            trainer_type=trainer_type,
            model_type=model_type,
            loss_type=loss_type,
            preprocessor_type=preprocessor_type,
            visualizer=visualizer,
            evaluator=evaluator,
            result=result,
            datasplitter_type=datasplitter_type,
            config=config,
            custom_split=custom_splits,
            ontologies=self.ontologies,
        )
        if not isinstance(self.config, OntixConfig):
            raise TypeError(
                f"For Ontix Pipeline, we only allow OntixConfig as type for config, got {type(self.config)}"
            )

    def sample_latent_space(self, split: str = "test", epoch: int = -1) -> torch.Tensor:
        """Samples new latent space points from the learned distribution.

        Args:
            split: The split to sample from (train, valid, test), default is test
            epoch: The epoch to sample from, default is the last epoch (-1)
        Returns:
            z: torch.Tensor - The sampled latent space points
        Raises:
            ValueError: if model has not been trained.
            TypeError: if mu and or logvar are no numpy arrays

        """

        if not hasattr(self, "_trainer") or self._trainer is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        if self.result.mus is None or self.result.sigmas is None:
            raise ValueError("Model has not learned the latent space distribution yet.")
        mu = self.result.mus.get(split=split, epoch=epoch)
        logvar = self.result.sigmas.get(split=split, epoch=epoch)

        if not isinstance(mu, np.ndarray):
            raise TypeError(
                f"Expected value to be of type numpy.ndarray, got {type(mu)}."
            )
        if not isinstance(logvar, np.ndarray):
            raise TypeError(
                f"Expected value to be of type numpy.ndarray, got {type(logvar)}."
            )
        mu_t = torch.from_numpy(mu)
        logvar_t = torch.from_numpy(logvar)

        # Move to same device and dtype as model
        mu_t = mu_t.to(
            device=self._trainer._model.device, dtype=self._trainer._model.dtype
        )
        logvar_t = logvar_t.to(
            device=self._trainer._model.device, dtype=self._trainer._model.dtype
        )

        with self._trainer._fabric.autocast(), torch.no_grad():
            z = self._trainer._model.reparameterize(mu_t, logvar_t)
            return z

    def _read_ont_file(self, file_path: str, sep: str = "\t") -> dict:
        """Function to read-in text files of ontologies with format child - separator - parent into an dictionary.

        Args:
            file_path: Path to file with ontology
            sep: Separator used in file
        Returns:
            ont_dic: Dictionary containing the ontology as described in the text file.

        """
        ont_dic = dict()
        with open(file_path, "r") as ont_file:
            for line in ont_file:
                id_parent = line.strip().split(sep)[1]
                id_child = line.split(sep)[0]

                if id_parent in ont_dic:
                    ont_dic[id_parent].append(id_child)
                else:
                    ont_dic[id_parent] = list()
                    ont_dic[id_parent].append(id_child)

        return ont_dic

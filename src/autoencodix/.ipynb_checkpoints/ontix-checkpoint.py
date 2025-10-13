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
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data.general_preprocessor import GeneralPreprocessor
from autoencodix.evaluate.evaluate import Evaluator

# from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.modeling._ontix_architecture import OntixArchitecture
from autoencodix.trainers._ontix_trainer import OntixTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig

from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.visualize.visualize import Visualizer


## Copy from Varix with ontology addition
class Ontix(BasePipeline):
    """
    Ontix specific version of the BasePipeline class.
    Inherits preprocess, fit, predict, evaluate, and visualize methods from BasePipeline.

    Attributes
    ----------
    preprocessed_data : Optional[DatasetContainer]
        User data if no datafiles in the config are provided. We expect these to be split and processed.
    raw_user_data : Optional[DataPackage]
        We give users the option to populate a DataPacke with raw data i.e. pd.DataFrames, MuData.
        We will process this data as we would do wit raw files specified in the config.
    config : Optional[Union[None, DefaultConfig]]
        Configuration object containing customizations for the pipeline
    _preprocessor : Preprocessor
        Preprocessor object to preprocess the input data (maybe custom for Ontix)
    _visualizer : Visualizer
        Visualizer object to visualize the model output (maybe custom for Ontix)
    _trainer : GeneralTrainer
        Trainer object that trains the model (maybe custom for Ontix)
    _evaluator : Evaluator
        Evaluator object that evaluates the model performance or downstream tasks (maybe custom for Ontix)
    result : Result
        Result object to store the pipeline results
    _datasets : Optional[DatasetContainer]
        Container for train, validation, and test datasets (preprocessed)
    data_splitter : DataSplitter
        DataSplitter object to split the data into train, validation, and test sets

    Methods
    -------
    all methods from BasePipeline

    sample_latent_space(split: str = "test", epoch: int = -1) -> torch.Tensor
        Samples new latent space points from the learned distribution.

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
        visualizer: Optional[BaseVisualizer] = None,
        evaluator: Optional[Evaluator] = None,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        """Initialize Ontix pipeline with customizable components.

        Some components are passed as types rather than instances because they require
        data that is only available after preprocessing.

        Parameters
        ----------
        ontologies: Union[tuple,list]
                Tuple of dictionaries containing the ontologies to be used to construct sparse decoder layers.
            If a list is provided, it is assumed to be a list of file paths to ontology files.
            First item in list or tuple will be treated as first layer (after latent space) and so on.
            Last layer need to contain features identical to the input data.
        sep: Optional[str]
            Separator used in ontology files, default is tab ("\t")
        preprocessed_data : Union[np.ndarray, AnnData, pd.DataFrame, DatasetContainer]
            Input data to be processed
        trainer_type : Type[BaseTrainer]
            Type of trainer to be instantiated during fit step, default is GeneralTrainer
        dataset_type : Type[BaseDataset]
            Type of dataset to be instantiated post-preprocessing, default is NumericDataset
        loss_type : Type[BaseLoss], which loss to use for Varix, default is VarixAutoencoderLoss
        preprocessor : Optional[Preprocessor]
            For data preprocessing, default creates new Preprocessor
        visualizer : Optional[Visualizer]
            For result visualization, default creates new Visualizer
        evaluator : Optional[Evaluator]
            For model evaluation, default creates new Evaluator
        result : Optional[Result]
            Container for pipeline results, default creates new Result
        datasplitter_type : Type[DataSplitter], optional
            Type of splitter to be instantiated during preprocessing, default is DataSplitter
        custom_splits : Optional[Dict[str, np.ndarray]]
            Custom train/valid/test split indices
        config : Optional[DefaultConfig]
            Configuration for all pipeline components
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
        """
        Samples new latent space points from the learned distribution.
        Parameters:
            split: str - The split to sample from (train, valid, test), default is test
            epoch: int - The epoch to sample from, default is the last epoch (-1)
        Returns:
            z: torch.Tensor - The sampled latent space points

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
        ARGS:
            file_path - (str): Path to file with ontology
            sep - (str): Separator used in file
        RETURNS:
            ont_dic - (dict): Dictionary containing the ontology as described in the text file.

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

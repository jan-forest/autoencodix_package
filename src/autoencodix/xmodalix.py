from typing import Dict, Optional, Type, Union
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
from autoencodix.data._xmodal_preprocessor import XModalPreprocessor
from autoencodix.evaluate.evaluate import Evaluator
from autoencodix.trainers._general_trainer import GeneralTrainer
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig


class XModalix(BasePipeline):
    """

    Attributes
    ----------
    preprocessed_data : Optional[DatasetContainer]
        User data if no datafiles in the config are provided. We expect these to be split and processed.
    raw_user_data : Optional[DataPackage]
        We give users the option to populate a DataPackage with raw data i.e. pd.DataFrames, MuData.
        We will process this data as we would do wit raw files specified in the config.
    config : Optional[Union[None, DefaultConfig]]
        Configuration object containing customizations for the pipeline
    _preprocessor : Preprocessor
        Preprocessor object to preprocess the input data - custom for XModalix
    _visualizer : Visualizer
        Visualizer object to visualize the model output - custom for XModalix
    _trainer : XModalixTrainer
        Trainer object that trains the model - custom for XModalix
    _evaluator : Evaluator
        Evaluator object that evaluates the model performance or downstream tasks
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
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = GeneralTrainer,
        dataset_type: Type[BaseDataset] = NumericDataset,
        model_type: Type[
            BaseAutoencoder
        ] = BaseAutoencoder,  # TODO make custom for XModalix
        loss_type: Type[BaseLoss] = BaseLoss,  # TODO make custom for XModalix
        preprocessor_type: Type[BasePreprocessor] = XModalPreprocessor,
        visualizer: Optional[BaseVisualizer] = None,
        evaluator: Optional[Evaluator] = None,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
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
        )

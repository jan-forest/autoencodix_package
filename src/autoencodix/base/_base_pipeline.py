import abc
import copy
from typing import TYPE_CHECKING, Dict, Optional, Type, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
from mudata import MuData
from torch.utils.data import Dataset

from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.evaluate.evaluate import Evaluator
from autoencodix.utils._result import Result
from autoencodix.utils._utils import Loader, Saver, config_method
from autoencodix.utils.default_config import DataCase, DataInfo, DefaultConfig

from ._base_autoencoder import BaseAutoencoder
from ._base_dataset import BaseDataset
from ._base_loss import BaseLoss
from ._base_preprocessor import BasePreprocessor
from ._base_trainer import BaseTrainer
from ._base_visualizer import BaseVisualizer


# tests: done
# internal check done
class BasePipeline(abc.ABC):
    """
    Abstract base class defining the interface for all models.

    This class provides the methods for preprocessing, training, predicting,
    visualizing, and evaluating models. Subclasses should perform steps similar
    to the parent class but include custom attributes such as Processor, Trainer,
    Evaluator, and Visualizer.

    Attributes
    ----------
    config : DefaultConfig
        The configuration for the pipeline. Handles preprocessing, training, and
        other settings.
    preprocessed_data : Optional[DatasetContainer]
            User data if no datafiles in the config are provided. We expect these to be split and processed.
    raw_user_data : Optional[DataPackage]
        We give users the option to populate a DataPacke with raw data i.e. pd.DataFrames, MuData.
        We will process this data as we would do wit raw files specified in the config.
    data_splitter : Optional[DataSplitter]
        Returns train, validation, and test indices; users can provide a custom splitter.
    result : Result
        Dataclass to store all results from the pipeline.
    _features : Optional[torch.Tensor]
        The preprocessed data, output of the former `make_data`.
    _preprocessor_type : Type[Preprocessor]
        The preprocessor filters, scales, matches, and cleans the data.
        Specific implementations for this will be used in subclasses.
    _datasets : Optional[DatasetContainer]
        Data in the form of Dataset classes after splitting; essentially, self._features
        wrapped in a Dataset class and split into train, validation, and test datasets.
    _trainer : Optional[Trainer]
        Trainer object with a `train` method to train model weights.
        Each subclass will have its own Trainer class.
    _evaluator : Optional[Evaluator]
        Object responsible for evaluating model performance.
        Currently a placeholder, with detailed implementation pending.
    _visualizer : Optional[Visualizer]
        Object responsible for visualizing results.
        Currently a placeholder, with detailed implementation pending.
    _dataset_type : Type[BaseDataset]
        Used to ensure correct dataset class type when overriding methods in subclasses.
    _trainer_type : Type[BaseTrainer]
        Used to ensure correct trainer class type when overriding methods in subclasses.
    _model_type : Type[BaseAutoencoder]
        Specifies the model architecture used in the pipeline.
    _loss_type : Type[BaseLoss]
        Specifies the loss function used during training.
    _visualizer : BaseVisualizer
        Visualizer instance responsible for generating visual outputs.
    custom_split : Optional[Dict[str, np.ndarray]]
        Allows users to provide a custom data split for training, validation, and testing.
    _data_splitter : DataSplitter
        The instance of the DataSplitter used for dividing data into training, validation, and testing sets.


    Methods
    -------
    preprocess(*kwargs):
        Calls the Preprocessor instance to preprocess the data. Updates the self._features attribute.
        Populates the self.results attribute with self._features and self._datasets.
    fit(*kwargs):
        Calls the Trainer instance to train the model on the training and validation data of self._datasets.
        Populates the self.results attribute with the trained model and training dynamics and results.
    predict(*kwargs):
        Calls the `predict` method of the Trainer instance to run inference with the test data on the trained model.
        If user inputs data, it preprocesses the data and runs inference.
        Updates the result attribute.
    evaluate(*kwargs):
        Calls the Evaluator instance to evaluate the model performance in downstream tasks.
        Updates the result attribute.
    visualize(*kwargs):
        Calls the Visualizer instance to visualize all relevant data in the result attribute.
    show_result():
        Helper function to directly show the visualized results.
    run():
        Runs the entire pipeline in the following order: preprocess, fit, predict, evaluate, visualize.
        Updates the result attribute.
    """

    def __init__(
        self,
        dataset_type: Type[BaseDataset],
        trainer_type: Type[BaseTrainer],
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        datasplitter_type: Type[DataSplitter],
        preprocessor_type: Type[BasePreprocessor],
        visualizer: BaseVisualizer,
        user_data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData]],
        evaluator: Evaluator,
        result: Result,
        config: DefaultConfig = DefaultConfig(),
        custom_split: Optional[Dict[str, np.ndarray]] = None,
        **kwargs: dict,
    ) -> None:
        """
        Initialize the model interface.

        Parameters
        ----------
        config : DefaultConfig, optional
            The configuration dictionary for the model.
        """
        processed_data = user_data if isinstance(user_data, DatasetContainer) else None
        raw_user_data = (
            user_data
            if isinstance(user_data, (DataPackage, ad.AnnData, MuData))
            else None
        )
        if processed_data is not None and not isinstance(
            processed_data, DatasetContainer
        ):
            raise TypeError(
                f"Expected data type to be DatasetContainer, got {type(processed_data)}."
            )

        self.preprocessed_data: DatasetContainer = processed_data
        self.raw_user_data: Union[DataPackage, ad.AnnData, MuData] = raw_user_data
        self.config = config
        self._trainer_type = trainer_type
        self._model_type = model_type
        self._loss_type = loss_type
        self._preprocessor_type = preprocessor_type
        if self.raw_user_data is not None:
            self.raw_user_data, datacase = self._handle_direct_user_data(
                data=self.raw_user_data,
                data_case=self.config.data_case,  # if is None, data_case is inferred based on the data type
            )
            self.config.data_case = datacase
            self._fill_data_info()

        self._preprocessor = self._preprocessor_type(
            config=self.config,
        )
        self._visualizer = visualizer
        self._dataset_type = dataset_type
        self._evaluator = evaluator
        self.result = result
        self._data_splitter = datasplitter_type(
            config=self.config, custom_splits=custom_split
        )

        if config:
            if not isinstance(config, DefaultConfig):
                raise TypeError(
                    f"Expected config type to be DefaultConfig, got {type(config)}."
                )
            self.config = config

        self._datasets: Optional[DatasetContainer] = (
            processed_data  # None, or user input
        )

    def _handle_direct_user_data(self, data, data_case=None):
        if isinstance(data, ad.AnnData):
            mudata = MuData({"user-data": data})
            data_package = DataPackage(multi_sc={"multi_sc": mudata})
            data_case = data_case or DataCase.MULTI_SINGLE_CELL
        elif isinstance(data, MuData):
            data_package = DataPackage(multi_sc={"multi_sc": data})
            data_case = data_case or DataCase.MULTI_SINGLE_CELL
        elif isinstance(data, pd.DataFrame):
            data_package = DataPackage(multi_bulk={"user-data": data})
            data_case = data_case or DataCase.MULTI_BULK
        elif isinstance(data, dict):
            # Check if all values in the dictionary are pandas DataFrames
            if all(isinstance(value, pd.DataFrame) for value in data.values()):
                data_package = DataPackage(multi_bulk=data)
                data_case = data_case or DataCase.MULTI_BULK
            else:
                raise ValueError(
                    "All values in the dictionary must be pandas DataFrames."
                )
        elif isinstance(data, DataPackage):
            data_package = data
        else:
            raise TypeError(
                f"Type: {type(data)} is not supported as of now, please provide: Union[AnnData, MuData, pd.DataFrame, Dict[str, pd.DataFrame], DataPackage]"
            )

        if data_case is None:
            raise ValueError("data_case must be provided if it cannot be inferred.")

        return data_package, data_case

    def _validate_raw_user_data(self) -> None:
        """
        Validates the raw user data provided by the user.

        Ensures that:
        - The raw user data is a DataPackage.
        - Each attribute of the DataPackage is a dictionary.
        - At least one attribute of the DataPackage is not None.

        Raises:
            TypeError: If the raw user data is not a DataPackage.
            ValueError: If any attribute of the DataPackage is not a dictionary.
            ValueError: If all attributes of the DataPackage are None.
        """
        if not isinstance(self.raw_user_data, DataPackage):
            raise TypeError(
                f"Expected raw_user_data to be of type DataPackage, got {type(self.raw_user_data)}."
            )

        all_none = True
        for attr_name in self.raw_user_data.__annotations__:
            attr_value = getattr(self.raw_user_data, attr_name)
            if attr_value is not None:
                all_none = False
                if not isinstance(attr_value, dict):
                    raise ValueError(
                        f"Attribute '{attr_name}' of raw_user_data must be a dictionary, got {type(attr_value)}."
                    )

        if all_none:
            raise ValueError(
                "All attributes of raw_user_data are None. At least one must be non-None."
            )

    def _fill_data_info(self):
        all_keys = []
        for k in self.raw_user_data.__annotations__:
            attr_value = getattr(self.raw_user_data, k)
            all_keys.append(k)
            if isinstance(attr_value, dict):
                all_keys.extend(attr_value.keys())
                for k, v in attr_value.items():
                    if isinstance(v, MuData):
                        all_keys.extend(v.mod.keys())
        for k in all_keys:
            if self.config.data_config.data_info.get(k) is None:
                self.config.data_config.data_info[k] = DataInfo()

    def _validate_user_data(self) -> None:
        """
        Validates the data provided by the user.

        If the user passed preprocessed_data as dataset container, this function calls _validate_container().
        If the user only passed a config with data configuration, this function calls _validate_config_data().
        """
        if self.raw_user_data is None:
            if self._datasets is not None:  # case when user passes preprocessed data
                self._validate_container()
            else:  # user passes data via config
                self._validate_config_data()
        else:
            self._validate_raw_user_data()

    def _validate_container(self) -> None:
        """
        Validates a DatasetContainer object to ensure it contains valid datasets.

        Ensures that the container has at least a training dataset.

        Raises:
            ValueError: If the container validation fails.
        """
        if self.preprocessed_data is None:
            raise ValueError("DatasetContainer is None. Please provide valid datasets.")
        none_count = 0
        if not isinstance(self.preprocessed_data.train, Dataset):
            if self.preprocessed_data.train is not None:
                raise ValueError(
                    f"Train dataset has to be either None or Dataset, got {type(self.preprocessed_data.valid)}"
                )
            none_count += 1

        if not isinstance(self.preprocessed_data.test, Dataset):
            if self.preprocessed_data.test is not None:
                raise ValueError(
                    f"Test dataset has to be either None or Dataset, got {type(self.preprocessed_data.valid)}"
                )
            none_count += 1

        if not isinstance(self.preprocessed_data.valid, Dataset):
            if self.preprocessed_data.valid is not None:
                raise ValueError(
                    f"Valid dataset has to be either None or Dataset, got {type(self.preprocessed_data.valid)}"
                )
            none_count += 1
        if none_count == 3:
            raise ValueError("At lest one split needs to be provided")

    def _validate_config_data(self) -> None:
        """
        Validates the data configuration provided via config.data_config.data_info.

        Ensures that:
        1. There is at least one non-annotation file
        2. When working with non-single-cell data, there must be an annotation file

        Raises:
            ValueError: If the data configuration validation fails.
        """

        data_info_dict = self.config.data_config.data_info
        if not data_info_dict:
            raise ValueError("data_info dictionary is empty.")

        # Check if there's at least one non-annotation file
        non_annotation_files = {
            key: info
            for key, info in data_info_dict.items()
            if info.data_type != "ANNOTATION"
        }

        if not non_annotation_files:
            raise ValueError("At least one non-annotation file must be provided.")

        # Check if there's any non-single-cell data
        non_single_cell_data = {
            key: info
            for key, info in data_info_dict.items()
            if not info.is_single_cell and info.data_type != "ANNOTATION"
        }

        # If there's non-single-cell data, check for annotation file
        if non_single_cell_data:
            annotation_files = {
                key: info
                for key, info in data_info_dict.items()
                if info.data_type == "ANNOTATION"
            }

            if not annotation_files:
                raise ValueError(
                    "When working with non-single-cell data, an annotation file must be provided."
                )

    @config_method(valid_params={"config"})  # TODO allow more config params
    def preprocess(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs: dict
    ) -> None:
        """
        Takes the user input data and filters, norrmalizes and cleans the data.
        Populates the self._datasets attribute with the preprocessed data as DatasetContainer,
        containing the splits as a child of PyTorch Dataset, depending on which preprocessor was used
        for the pipeline.

        Parameters:
            config: DefaultConfig, optional (default: None)
                allows to pass a custom configuration for the preprocessing step.
            **kwargs:
                User can pass configuration parameters as single keyword arguments.
        Raises:
            NotImplementedError:
                If the preprocessor is not initialized.

        """
        if self._preprocessor_type is None:
            raise NotImplementedError("Preprocessor not initialized")
        self._validate_user_data()
        if self.preprocessed_data is None:
            self._datasets = self._preprocessor.preprocess(
                raw_user_data=self.raw_user_data
            )
            self.result.datasets = self._datasets
        else:
            self._datasets = self.preprocessed_data
            self.result.datasets = self.preprocessed_data

    @config_method(
        valid_params={
            "config",
            "batch_size",
            "epochs",
            "learning_rate",
            "n_workers",
            "device",
            "n_gpus",
            "gpu_strategy",
            "weight_decay",
            "reproducible",
            "global_seed",
            "reconstruction_loss",
            "checkpoint_interval",
        }
    )
    def fit(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Trains the model, based on the in the child class defined Trainer class.
        Populates the self.result attribute with the trained model and training dynamics and results.

        Parameters:
            config: DefaultConfig, optional (default: None)
                allows to pass a custom configuration for the training step.
            **kwargs:
                User can pass configuration parameters as single keyword arguments.
        Raises:
            ValueError:
                If no data is available for training.
            ValueError:
                If the datasets are not built. Please run the preprocess method first.

        """

        if self._datasets is None:
            raise ValueError(
                "Datasets not built. Please run the preprocess method first."
            )

        self._trainer = self._trainer_type(
            trainset=self._datasets.train,
            validset=self._datasets.valid,
            result=self.result,
            config=self.config,
            model_type=self._model_type,
            loss_type=self._loss_type,
        )
        trainer_result = self._trainer.train()
        self.result.update(trainer_result)

    @config_method(valid_params={"config"})
    def predict(
        self,
        data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData]] = None,
        config: Optional[Union[None, DefaultConfig]] = None,
        **kwargs,
    ) -> None:
        """
        Run inference with the test data on the trained model.
        If user inputs data, it preprocesses the data and runs inference.
        Populates the self.result attribute with the inference results.

        Parameters:
            data:
                User input data to run inference as DatasetContainer, we will use the test split here.
            config: DefaultConfig, optional (default: None)
                allows to pass a custom configuration for the prediction step.
            **kwargs:
                User can pass configuration parameters as single keyword arguments.
        Raises:
            NotImplementedError:
                If the preprocessor is not initialized.
            NotImplementedError:
                If the predictor is not initialized.
            NotImplementedError:
                If the datasets are not built. Please run the preprocess method first.
            NotImplementedError:
                If the model is not trained. Please run the fit method first.
        """
        if self._preprocessor is None:  # type ignore
            raise NotImplementedError("Preprocessor not initialized")
        if self.result.model is None:
            raise NotImplementedError(
                "Model not trained. Please run the fit method first"
            )

        if data is None:
            if self._datasets.test is None:
                raise ValueError("No test data available for prediction")
            predict_data: DatasetContainer = self._datasets
        elif isinstance(data, DatasetContainer):
            predict_data = data
            self.result.new_datasets = predict_data

        elif isinstance(data, (DataPackage, DatasetContainer, ad.AnnData, MuData)):
            data, _ = self._handle_direct_user_data(
                data=data, data_case=self.config.data_case
            )
            predict_data: DatasetContainer = self._preprocessor.preprocess(
                raw_user_data=data, predict_new_data=True
            )
            self.result.new_datasets = predict_data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        if predict_data.test is None:
            raise ValueError(
                f"The data for prediction need to be a DatasetContainer with a test attribute, got: {predict_data}"
            )

        predictor_results = self._trainer.predict(
            data=predict_data.test, model=self.result.model
        )
        latent = predictor_results.latentspaces.get(epoch=-1, split="test")
        self.result.adata_latent = ad.AnnData(latent)
        self.result.adata_latent.obs_names = predict_data.test.sample_ids
        self.result.adata_latent.uns["var_names"] = predict_data.test.feature_ids

        self.result.update(predictor_results)

        raw_recon = self.result.reconstructions.get(epoch=-1, split="test")
        if isinstance(data, DatasetContainer):
            temp = copy.deepcopy(data.test)
            temp.data = torch.from_numpy(raw_recon)
            self.result.final_reconstruction = temp
        # user inputs already processed data into pipeline and does not overwrite it in predict
        elif self.preprocessed_data is not None:
            temp = copy.deepcopy(self.preprocessed_data.test)
            temp.data = torch.from_numpy(raw_recon)
            self.result.final_reconstruction = temp

        # for SC-community UX
        elif self.config.data_case == DataCase.MULTI_SINGLE_CELL:
            pkg = self._preprocessor.format_reconstruction(reconstruction=raw_recon)
            self.result.final_reconstruction = pkg.multi_sc["multi_sc"]
        # Case C: other non‐DatasetContainer → full package
        elif not isinstance(data, DatasetContainer):
            pkg = self._preprocessor.format_reconstruction(reconstruction=raw_recon)
            self.result.final_reconstruction = pkg
        # Case D: fallback
        else:
            print("Reconstruction not available for this data type or case.")



    def decode(
        self, latent: Union[torch.Tensor, ad.AnnData, pd.DataFrame]
    ) -> Union[torch.Tensor, ad.AnnData, pd.DataFrame]:
        """
        Decodes a latent representation into the original data space.

        Handles both PyTorch tensors and AnnData objects as input.
        If an AnnData object is provided, it extracts the `.X` attribute,
        converts it to a tensor, and then decodes it.  It also checks
        that the size of the latent representation is compatible with
        the model's expected input size.

        Args:
            latent: The latent representation to decode.
                - If torch.Tensor: A tensor of shape (n_cells, n_latent).
                - If AnnData: An AnnData object. The .X attribute should
                have shape (n_cells, n_features), and the function
                will use a linear transformation to project this
                to (n_cells, n_latent) if necessary.

        Returns:
            recons: The reconstructed data.
                - If input is torch.Tensor: A tensor of shape (n_cells, n_features).
                - If input is AnnData: An AnnData object with the reconstructed data and obs_names + var_names
                - If input is pd.DataFrame: A DataFrame with the reconstructed data and index + columns.

        Raises:
            TypeError: If no model has been trained yet.
            ValueError: If the input `latent` has an incompatible size
                with the model's expected latent input size.
        """
        if self.result.model is None:
            raise TypeError("No model trained yet, use fit() or run() method first")

        if isinstance(latent, ad.AnnData):
            latent_data = torch.tensor(
                latent.X, dtype=torch.float32
            )  # Ensure float for compatibility

            expected_latent_dim = self.config.latent_dim
            if not latent_data.shape[1] == expected_latent_dim:
                raise ValueError(
                    f"Input AnnData's .X has shape {latent_data.shape}, but the model expects a latent vector of"
                    f" size {expected_latent_dim}.  Consider projecting the AnnData to the correct latent space first."
                )
            latent_tensor = latent_data

            recons: torch.Tensor = self._trainer.decode(x=latent_tensor)

            recons_adata = ad.AnnData(
                X=recons.to("cpu").detach().numpy(),
                obs=pd.DataFrame(index=latent.obs_names),
                var=pd.DataFrame(index=self._datasets.train.feature_ids),
            )

            return recons_adata
        elif isinstance(latent, pd.DataFrame):
            latent_tensor = torch.tensor(latent.values, dtype=torch.float32)
            recons: torch.Tensor = self._trainer.decode(x=latent_data)
            return pd.DataFrame(
                recons.to("cpu").detach().numpy(),
                index=latent.index,
                columns=latent.columns,
            )
        elif isinstance(latent, torch.Tensor):
            # Check size compatibility
            expected_latent_dim = self.config.latent_dim
            if not latent.shape[1] == expected_latent_dim:
                raise ValueError(
                    f"Input tensor has shape {latent.shape}, but the model expects a latent vector of"
                    f" size {expected_latent_dim}."
                )
            latent_tensor = latent
        else:
            raise TypeError(
                f"Input 'latent' must be either a torch.Tensor or an AnnData object, not {type(latent)}."
            )

        recons: torch.Tensor = self._trainer.decode(x=latent_tensor)
        return recons

    @config_method(valid_params={"config"})
    def evaluate(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        if config is None:
            config = self.config
        # Add your evaluation logic here

    @config_method(valid_params={"config"})
    def visualize(
        self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs
    ) -> None:
        """
        Visualize model results.

        Raises
        ------
        NotImplementedError
            If the visualizer is not initialized.
        ValueError
            If no data is available for visualization.
        """
        if self._visualizer is None:
            raise NotImplementedError("Visualizer not initialized")

        self._visualizer.visualize(result=self.result, config=self.config)

    def show_result(self) -> None:
        print("Make plots")

        self._visualizer.show_loss(plot_type="absolute")

        self._visualizer.show_latent_space(result=self.result, plot_type="Ridgeline")

        self._visualizer.show_latent_space(result=self.result, plot_type="2D-scatter")

    def run(
        self, data: Optional[Union[DatasetContainer, DataPackage]] = None
    ) -> Result:
        """
        Run the entire model pipeline (preprocess, fit, predict, evaluate, visualize).
        When predict step should be run on user input data, the data parameter should be provided.
        (Overrides test data from the pipeline object)
        Populates the result attribute and returns it.

        Parameters:
            data: Data

        Returns:
            Result:
                The result object containing all relevant data from the pipeline.

        """
        self.preprocess()
        self.fit()
        self.predict(data=data)
        self.evaluate()
        self.visualize()
        return self.result

    def save(self, file_path):
        """Saves the pipeline using the Saver class."""
        saver = Saver(file_path)
        saver.save(self)

    @classmethod
    def load(cls, file_path):
        """Loads the pipeline using the Loader class."""
        loader = Loader(file_path)
        return loader.load()

import abc
import copy
from typing import Dict, Optional, Tuple, Type, Union, Any

import warnings
import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import torch
from mudata import MuData  # type: ignore
from torch.utils.data import Dataset

# ML evaluation
from sklearn import linear_model  # type: ignore
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor  # type: ignore


from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage

# from autoencodix.evaluate.evaluate import Evaluator
from autoencodix.base._base_evaluator import BaseEvaluator
from autoencodix.utils._result import Result
from autoencodix.utils._utils import Loader, Saver
from autoencodix.configs.default_config import DataCase, DataInfo, DefaultConfig

from ._base_autoencoder import BaseAutoencoder
from ._base_dataset import BaseDataset
from ._base_loss import BaseLoss
from ._base_preprocessor import BasePreprocessor
from ._base_trainer import BaseTrainer
from ._base_visualizer import BaseVisualizer


class BasePipeline(abc.ABC):
    """Provides a standardized interface for building model pipelines.

    Implements methods for preprocessing data, training models, making predictions,
    evaluating performance, and visualizing results. Subclasses customize behavior
    by providing specific implementations for processing, training, evaluation,
    and visualization. For example when using the Stackix Model, we would use
    the StackixPreprocessor Type for preprocessing.

    Attributes:
        config: Configuration for the pipeline's components and behavior.
        preprocessed_data: Pre-split and processed data that can be provided by user.
        raw_user_data: Raw input data for processing (DataFrames, MuData, etc.).
        result: Storage container for all pipeline outputs.
        _preprocessor: Component that filters, scales, and cleans data.
        _visualizer: Component that generates visual representations of results.
        _dataset_type: Base class for dataset implementations.
        _trainer_type: Base class for trainer implementations.
        _model_type: Base class for model architecture implementations.
        _loss_type: Base class for loss function implementations.
        _datasets: Split datasets after preprocessing.
        _evaluator: Component that assesses model performance. Not implemented yet
        _data_splitter: Component that divides data into train/validation/test sets.
        _ontologies: Tuple of dictionaries containing the ontologies to be used to construct sparse decoder layers.
            If a list is provided, it is assumed to be a list of file paths to ontology files.
            First item in list or tuple will be treated as first layer (after latent space) and so on.

    """

    def __init__(
        self,
        dataset_type: Type[BaseDataset],
        trainer_type: Type[BaseTrainer],
        model_type: Type[BaseAutoencoder],
        loss_type: Type[BaseLoss],
        datasplitter_type: Type[DataSplitter],
        preprocessor_type: Type[BasePreprocessor],
        data: Optional[
            Union[DataPackage, DatasetContainer, ad.AnnData, MuData, pd.DataFrame, dict]
        ],
        visualizer: Optional[BaseVisualizer] = None,
        evaluator: Optional[BaseEvaluator] = None,
        result: Optional[Result] = None,
        config: Optional[DefaultConfig] = None,
        custom_split: Optional[Dict[str, np.ndarray]] = None,
        ontologies: Optional[Union[Tuple, Dict[Any, Any]]] = None,
        **kwargs: dict,
    ) -> None:
        """Initializes the pipeline with components and configuration.

        Args:
            dataset_type: Class for dataset implementations.
            trainer_type: Class for model training implementations.
            model_type: Class for model architecture implementations.
            loss_type: Class for loss function implementations.
            datasplitter_type: Class for data splitting implementation.
            preprocessor_type: Class for data preprocessing implementation.
            visualizer: Component for generating visualizations.
            data: Input data to be processed or already processed data.
            evaluator: Component for assessing model performance.
            result: Storage container for pipeline outputs.
            config: Configuration parameters for all pipeline components.
            custom_split: User-provided data splits (train/validation/test).
            **kwargs: Additional keyword arguments.

        Raises:
            TypeError: If inputs have incorrect types.
        """
        if not hasattr(self, "_default_config"):
            raise ValueError(
                """
                            The _default_config attribute has not been specified in your pipeline class.

                            Example:
                            self._default_config = XModalixConfig()

                            This error typically occurs when a new architecture is added without setting the
                            _default_config in its corresponding pipeline class.

                            For more details, please refer to the 'how to add a new architecture' section in our documentation.
                            """
            )

        self._validate_config(config=config)
        self._validate_user_input(data=data)
        processed_data = data if isinstance(data, DatasetContainer) else None
        raw_user_data = (
            data
            if isinstance(data, (DataPackage, ad.AnnData, MuData, pd.DataFrame, dict))
            else None
        )
        if processed_data is not None and not isinstance(
            processed_data, DatasetContainer
        ):
            raise TypeError(
                f"Expected data type to be DatasetContainer, got {type(processed_data)}."
            )

        self.preprocessed_data: Optional[DatasetContainer] = processed_data
        self.raw_user_data: Union[
            DataPackage, ad.AnnData, MuData, pd.DataFrame, dict
        ] = raw_user_data
        self._trainer_type = trainer_type
        self._model_type = model_type
        self._loss_type = loss_type
        self._preprocessor_type = preprocessor_type
        if self.raw_user_data is not None:
            self.raw_user_data, datacase = self._handle_direct_user_data(
                data=self.raw_user_data,
            )
            self.config.data_case = datacase
            self._fill_data_info()

        self._ontologies = ontologies
        self._preprocessor = self._preprocessor_type(
            config=self.config, ontologies=self._ontologies
        )

        self._visualizer = visualizer() if visualizer is not None else BaseVisualizer()
        self._evaluator = evaluator() if evaluator is not None else BaseEvaluator()
        self.result = result if result is not None else Result()
        self._dataset_type = dataset_type
        self._data_splitter = datasplitter_type(
            config=self.config, custom_splits=custom_split
        )

        self._datasets: Optional[DatasetContainer] = (
            processed_data  # None, or user input
        )

    def _validate_config(self, config: Any) -> None:
        """Sets config to default if None, or validates its type.
        Args:
            config: Configuration object to validate or set to default.
        Raises:
            TypeError: If config is not of type DefaultConfig
        """
        if config is None:
            self.config = self._default_config  # type: ignore
        else:
            if not isinstance(config, DefaultConfig):
                raise TypeError(
                    f"Expected config type to be DefaultConfig, got {type(config)}."
                )
            if not isinstance(config, type(self._default_config)):  # type: ignore
                warnings.warn(
                    f"Your config is of type: {type(config)}, for this pipeline the default params of: {type(self._default_config)} work best"
                )
            self.config = config

    def _validate_user_input(self, data: Any) -> None:
        """Ensures that user-provided data is of a valid type.
        Args:
            data: User-provided data to validate.
        Raises:
            TypeError: If data is not of a supported type.
        """
        if not isinstance(
            data,
            (
                DataPackage,
                ad.AnnData,
                MuData,
                pd.DataFrame,
                dict,
                type(None),
                DatasetContainer,
            ),
        ):
            raise TypeError(
                f"Expected data type to be one of [DataPackage, AnnData, MuData, "
                f"pd.DataFrame, dict, DatasetContainer], got {type(data)}."
            )

    def _handle_direct_user_data(
        self,
        data,
    ) -> Tuple[DataPackage, DataCase]:
        """Converts raw user data into a standardized DataPackage format.

        Args:
            data: Raw input data in various formats.
            data_case: Specifies how to interpret the data structure.

        Returns:
            A tuple containing:
                - DataPackage containing the standardized data
                - DataCase, muliti_single_cell or multi_bulk, etc.

        Raises:
            TypeError: If data format is not supported.
            ValueError: If data doesn't meet format requirements or data_case
                cannot be inferred.
        """
        print(f"in handle_direct_user_data with data: {type(data)}")
        data_case = self.config.data_case
        if isinstance(data, DataPackage):
            data_package = data
            data_case = self.config.data_case
        elif isinstance(data, ad.AnnData):
            mudata = MuData({"user-data": data})
            data_package = DataPackage(multi_sc={"multi_sc": mudata})
            if self.config.data_case is None:
                data_case = DataCase.MULTI_SINGLE_CELL
        elif isinstance(data, MuData):
            data_package = DataPackage(multi_sc={"multi_sc": data})
            if self.config.data_case is None:
                data_case = DataCase.MULTI_SINGLE_CELL
        elif isinstance(data, pd.DataFrame):
            data_package = DataPackage(multi_bulk={"user-data": data})
            if self.config.data_case is None:
                data_case = DataCase.MULTI_BULK
        elif isinstance(data, dict):
            # Check if all values in the dictionary are pandas DataFrames
            if all(isinstance(value, pd.DataFrame) for value in data.values()):
                data_package = DataPackage(multi_bulk=data)
                if self.config.data_case is None:
                    data_case = DataCase.MULTI_BULK
            else:
                raise ValueError(
                    "All values in the dictionary must be pandas DataFrames."
                )
        if data_case is None:
            raise ValueError("data_case must be provided if it cannot be inferred.")

        return data_package, data_case

    def _validate_raw_user_data(self) -> None:
        """Validates the format and content of user-provided raw data.

        Ensures that raw_user_data is a valid DataPackage with properly formatted
        attributes.

        Raises:
            TypeError: If raw_user_data is not a DataPackage.
            ValueError: If DataPackage attributes aren't dictionaries or all are None.
        """
        if not isinstance(self.raw_user_data, DataPackage):
            raise TypeError(
                f"Expected raw_user_data to be of type DataPackage, got "
                f"{type(self.raw_user_data)}."
            )

        all_none = True
        for attr_name in self.raw_user_data.__annotations__:
            attr_value = getattr(self.raw_user_data, attr_name)
            if attr_value is not None:
                all_none = False
                if not isinstance(attr_value, dict):
                    raise ValueError(
                        f"Attribute '{attr_name}' of raw_user_data must be a dictionary, "
                        f"got {type(attr_value)}."
                    )

        if all_none:
            raise ValueError(
                "All attributes of raw_user_data are None. At least one must be non-None."
            )

    def _fill_data_info(self) -> None:
        """Populates the config's data_info with entries for all data keys.

        Creates DataInfo objects for each data key found in raw_user_data
        if they don't already exist in the configuration.
        This method is needed, when the user provides data via the Pipeline and
        not via the config.
        """
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

    def _validate_user_data(self):
        """Validates user-provided data based on its source and format.

        Performs different validation based on whether the user provided
        preprocessed data, raw data, or a data configuration.

        Raises:
            Various exceptions depending on validation results.
        """
        if self.raw_user_data is None:
            if self._datasets is not None:  # case when user passes preprocessed data
                self._validate_container()
            else:  # user passes data via config
                self._validate_config_data()
        else:
            self._validate_raw_user_data()

    def _validate_container(self):
        """Validates that a DatasetContainer has at least one valid dataset.

        Ensures the container has properly formatted datasets and at least
        one split is present.

        Raises:
            ValueError: If container validation fails.
        """
        if self.preprocessed_data is None:
            raise ValueError("DatasetContainer is None. Please provide valid datasets.")
        none_count = 0
        if not isinstance(self.preprocessed_data.train, Dataset):
            if self.preprocessed_data.train is not None:
                raise ValueError(
                    f"Train dataset has to be either None or Dataset, got "
                    f"{type(self.preprocessed_data.train)}"
                )
            none_count += 1
        if not isinstance(self.preprocessed_data.test, Dataset):
            if self.preprocessed_data.test is not None:
                raise ValueError(
                    f"Test dataset has to be either None or Dataset, got "
                    f"{type(self.preprocessed_data.test)}"
                )
            none_count += 1

        if not isinstance(self.preprocessed_data.valid, Dataset):
            if self.preprocessed_data.valid is not None:
                raise ValueError(
                    f"Valid dataset has to be either None or Dataset, got "
                    f"{type(self.preprocessed_data.valid)}"
                )
            none_count += 1
        if none_count == 3:
            raise ValueError("At least one split needs to be provided")

    def _validate_config_data(self):
        """Validates the data configuration provided via config.

        Ensures the data configuration has the necessary components based
        on the data types being processed.

        Raises:
            ValueError: If data configuration validation fails.
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
                    "When working with non-single-cell data, an annotation file must be "
                    "provided."
                )

    def preprocess(self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs):
        """Filters, normalizes and prepares data for model training.

        Processes raw input data into the format required by the model and creates
        train/validation/test splits as needed.

        Args:
            config: Optional custom configuration for preprocessing.
            **kwargs: Additional configuration parameters as keyword arguments.

        Raises:
            NotImplementedError: If preprocessor is not initialized.
        """
        print("preprocessing")
        if self._preprocessor_type is None:
            raise NotImplementedError("Preprocessor not initialized")
        self._validate_user_data()
        if self.preprocessed_data is None:
            self._datasets = self._preprocessor.preprocess(
                raw_user_data=self.raw_user_data,  # type: ignore
            )
            self.result.datasets = self._datasets
        else:
            self._datasets = self.preprocessed_data
            self.result.datasets = self.preprocessed_data

    def fit(self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs):
        """Trains the model on preprocessed data.

        Creates and configures a trainer instance, then executes the training
        process using the preprocessed datasets.

        Args:
            config: Optional custom configuration for training.
            **kwargs: Additional configuration parameters as keyword arguments.

        Raises:
            ValueError: If datasets aren't available for training.
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
            ontologies=self._ontologies,  # Ontix
        )
        trainer_result = self._trainer.train()
        self.result.update(other=trainer_result)

    def predict(
        self,
        data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData]] = None,
        config: Optional[Union[None, DefaultConfig]] = None,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
        **kwargs,
    ):
        """Generates predictions using the trained model.

        Uses the trained model to make predictions on test data or new data
        provided by the user. Processes the results and stores them in the
        result container.

        Args:
            data: Optional new data for predictions.
            config: Optional custom configuration for prediction.
            **kwargs: Additional configuration parameters as keyword arguments.

        Raises:
            NotImplementedError: If required components aren't initialized.
            ValueError: If no test data is available or data format is invalid.
        """
        self._validate_prediction_requirements()

        original_input = data
        predict_data = self._prepare_prediction_data(data=data)

        predictor_results = self._generate_predictions(
            predict_data=predict_data,
        )

        self._process_latent_results(
            predictor_results=predictor_results, predict_data=predict_data
        )
        self._postprocess_reconstruction(
            predictor_results=predictor_results,
            original_input=original_input,
            predict_data=predict_data,
        )

        return self.result

    def _validate_prediction_requirements(self):
        """Validate that required components are initialized."""
        if self._preprocessor is None:
            raise NotImplementedError("Preprocessor not initialized")
        if self.result.model is None:
            raise NotImplementedError(
                "Model not trained. Please run the fit method first"
            )

    def _prepare_prediction_data(
        self,
        data: Optional[Union[DataPackage, DatasetContainer, ad.AnnData, MuData]] = None,
    ) -> DatasetContainer:
        """Prepare and validate input data for prediction.
        Args:
            data: Optional new data for predictions. If None, uses existing datasets.
        Returns:
            DatasetContainer: The prepared dataset container for predictions.
        Raises:
            ValueError: If data type is unsupported or no test data is available.
        """
        if data is None:
            return self._get_existing_datasets()
        elif isinstance(data, DatasetContainer):
            return self._handle_dataset_container(data=data)
        elif isinstance(data, (DataPackage, ad.AnnData, MuData, dict, pd.DataFrame)):
            return self._handle_user_data(data=data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _get_existing_datasets(self) -> DatasetContainer:
        """Get existing preprocessed datasets and validate them for prediction.
        Returns:
            DatasetContainer: The preprocessed datasets available for prediction.
        Raises:
            ValueError: If no datasets are available or no test data is present.
        """
        if self._datasets is None:
            raise ValueError(
                "No data provided for prediction and no preprocessed datasets "
                "available. Please run the preprocess method first or provide "
                "data for prediction."
            )
        if self._datasets.test is None:
            raise ValueError("No test data available for prediction")
        return self._datasets

    def _handle_dataset_container(self, data: DatasetContainer) -> DatasetContainer:
        """Handle DatasetContainer input for prediction.
        Args:
            data: DatasetContainer containing preprocessed datasets.
        Returns:
            DatasetContainer: The processed dataset container for predictions.
        """
        self.result.new_datasets = data

        if hasattr(self._preprocessor, "_dataset_container"):
            self._preprocessor._dataset_container = data

        return data

    def _handle_user_data(self, data: Any) -> DatasetContainer:
        """Handle user-provided data (DataPackage, AnnData, etc.).
        Args:
            data: Raw user data in various formats (DataPackage, AnnData, etc.).
        Returns:
            DatasetContainer: The processed dataset container for predictions.
        Raises:
            ValueError: If data type is unsupported or no test data is available.
        """
        processed_data, _ = self._handle_direct_user_data(data=data)
        predict_data = self._preprocessor.preprocess(
            raw_user_data=processed_data, predict_new_data=True
        )
        self.result.new_datasets = predict_data
        return predict_data

    def _validate_prediction_data(self, predict_data: DatasetContainer):
        """Validate that prediction data has required test split."""
        if predict_data.test is None:
            raise ValueError(
                f"The data for prediction need to be a DatasetContainer with a test "
                f"attribute, got: {predict_data}"
            )

    def _generate_predictions(
        self,
        predict_data: DatasetContainer,
    ):
        """Generate predictions using the trained model."""
        self._validate_prediction_data(predict_data=predict_data)
        return self._trainer.predict(
            data=predict_data.test,
            model=self.result.model,
        )  # type: ignore

    def _process_latent_results(
        self, predictor_results, predict_data: DatasetContainer
    ):
        """Process and store latent space results."""
        latent = predictor_results.latentspaces.get(epoch=-1, split="test")
        if isinstance(latent, dict):
            print("Detected dictionary in latent results, extracting array...")
            latent = next(iter(latent.values()))  # TODO better adjust for xmodal
        self.result.adata_latent = ad.AnnData(latent)
        self.result.adata_latent.obs_names = predict_data.test.sample_ids  # type: ignore
        self.result.adata_latent.uns["var_names"] = predict_data.test.feature_ids  # type: ignore
        self.result.update(predictor_results)

    def _postprocess_reconstruction(
        self, predictor_results, original_input, predict_data: DatasetContainer
    ):
        """Postprocess reconstruction results based on input type.

        This outpus the reconstruction in the same format as the original input data,
        whether it is a DatasetContainer, DataPackage, AnnData, MuData, or other formats.

        Args:
            predictor_results: Results from the prediction step containing reconstructions.
            original_input: Original input data format (if provided).
            predict_data: DatasetContainer with preprocessed datasets for prediction.
        Raises:
            ValueError: If reconstruction fails or data types are incompatible.
        """
        raw_recon: Union[Dict, np.ndarray, torch.Tensor] = (
            self.result.reconstructions.get(epoch=-1, split="test")
        )
        if isinstance(raw_recon, np.ndarray):
            raw_recon = torch.from_numpy(raw_recon)  # type: ignore
        elif isinstance(raw_recon, dict):
            raw_recon = raw_recon.get("translation")  # type: ignore
            if raw_recon is None:
                raise ValueError(
                    f"Raw recon is dict, but has no translation key, this should not happen: {raw_recon}"
                )
            raw_recon = torch.from_numpy(raw_recon)  # type: ignore
        else:
            raise ValueError(
                f"type of raw_recon has to be 'dict' or 'np.ndarray', got: {type(raw_recon)}"
            )

        if original_input is None:
            # Using existing datasets
            self._handle_dataset_container_reconstruction(
                raw_recon=raw_recon,  # type: ignore
                dataset_container=predict_data,
                context="existing datasets",
            )
        elif isinstance(original_input, DatasetContainer):
            self._handle_dataset_container_reconstruction(
                raw_recon=raw_recon,  # type: ignore
                dataset_container=original_input,
                context="provided DatasetContainer",
            )
        elif self.config.data_case == DataCase.MULTI_SINGLE_CELL:
            self._handle_multi_single_cell_reconstruction(
                raw_recon=raw_recon,
                predictor_results=predictor_results,  # type: ignore
            )
        elif isinstance(
            original_input, (DataPackage, ad.AnnData, MuData, dict, pd.DataFrame)
        ):
            self._handle_user_data_reconstruction(
                raw_recon=raw_recon, predictor_results=predictor_results
            )
        else:
            self._handle_unsupported_reconstruction()

    def _handle_dataset_container_reconstruction(
        self,
        raw_recon: torch.Tensor,
        dataset_container: DatasetContainer,
        context: str = "DatasetContainer",
    ):
        """Handle reconstruction for DatasetContainer input."""
        if dataset_container.test is None:
            raise ValueError(f"No test data available in {context} for reconstruction.")
        temp = copy.deepcopy(dataset_container.test)
        temp.data = raw_recon
        self.result.final_reconstruction = temp

    def _handle_multi_single_cell_reconstruction(
        self, raw_recon: torch.Tensor, predictor_results
    ):
        """Handle reconstruction for multi-single-cell data."""
        pkg = self._preprocessor.format_reconstruction(
            reconstruction=raw_recon, result=predictor_results
        )
        if not isinstance(pkg.multi_sc, dict):
            raise ValueError(
                "Expected pkg.multi_sc to be a dictionary, got "
                f"{type(pkg.multi_sc)} instead."
            )
        self.result.final_reconstruction = pkg.multi_sc["multi_sc"]

    def _handle_user_data_reconstruction(
        self, raw_recon: torch.Tensor, predictor_results
    ):
        """Handle reconstruction for user-provided data formats."""
        pkg = self._preprocessor.format_reconstruction(
            reconstruction=raw_recon, result=predictor_results
        )
        self.result.final_reconstruction = pkg

    def _handle_unsupported_reconstruction(self):
        """Handle cases where reconstruction formatting is not available."""
        print(
            "Reconstruction Formatting (the process of using the reconstruction "
            "output of the autoencoder models and combine it with metadata to get "
            "the exact same data structure as the raw input data i.e, a DataPackage, "
            "DatasetContainer, or AnnData) not available for this data type or case."
        )

    def decode(
        self, latent: Union[torch.Tensor, ad.AnnData, pd.DataFrame]
    ) -> Union[torch.Tensor, ad.AnnData, pd.DataFrame]:
        """Transforms latent space representations back to input space.

        Handles various input formats for the latent representation and
        returns the decoded data in a matching format.

        Args:
            latent: Latent space representation to decode.

        Returns:
            Decoded data in a format matching the input.

        Raises:
            TypeError: If no model has been trained or input type is invalid.
            ValueError: If latent dimensions are incompatible with the model.
        """
        if self.result.model is None:
            raise TypeError("No model trained yet, use fit() or run() method first")
        recons: torch.Tensor
        if isinstance(latent, ad.AnnData):
            latent_data = torch.tensor(
                latent.X, dtype=torch.float32
            )  # Ensure float for compatibility

            expected_latent_dim = self.config.latent_dim
            if not latent_data.shape[1] == expected_latent_dim:
                raise ValueError(
                    f"Input AnnData's .X has shape {latent_data.shape}, but the model "
                    f"expects a latent vector of size {expected_latent_dim}. Consider "
                    f"projecting the AnnData to the correct latent space first."
                )
            latent_tensor = latent_data

            recons = self._trainer.decode(x=latent_tensor)
            if self._datasets is None:
                raise ValueError(
                    "No datasets available in the DatasetContainer to reconstruct "
                    "AnnData objects. Please provide a valid DatasetContainer."
                )
            if self._datasets.train is None:
                raise ValueError(
                    "The train dataset in the DatasetContainer is None. "
                    "Please provide a valid train dataset to reconstruct AnnData objects."
                )
            if not isinstance(self._datasets.train, BaseDataset):
                raise TypeError(
                    "The train dataset in the DatasetContainer must be a BaseDataset "
                    "to reconstruct AnnData objects."
                )
            recons_adata = ad.AnnData(
                X=recons.to("cpu").detach().numpy(),
                obs=pd.DataFrame(index=latent.obs_names),
                var=pd.DataFrame(index=self._datasets.train.feature_ids),
            )

            return recons_adata
        elif isinstance(latent, pd.DataFrame):
            latent_tensor = torch.tensor(latent.values, dtype=torch.float32)
            recons = self._trainer.decode(x=latent_tensor)
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
                    f"Input tensor has shape {latent.shape}, but the model expects a "
                    f"latent vector of size {expected_latent_dim}."
                )
            latent_tensor = latent
        else:
            raise TypeError(
                f"Input 'latent' must be either a torch.Tensor or an AnnData object, "
                f"not {type(latent)}."
            )

        return self._trainer.decode(x=latent_tensor)

    def evaluate(
        self,
        ml_model_class: ClassifierMixin = linear_model.LogisticRegression(),  # Default is sklearn LogisticRegression
        ml_model_regression: RegressorMixin = linear_model.LinearRegression(),  # Default is sklearn LinearRegression
        params: Union[
            list, str
        ] = [],  # Default empty list, to use all parameters use string "all"
        metric_class: str = "roc_auc_ovr",  # Default is 'roc_auc_ovr' via https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        metric_regression: str = "r2",  # Default is 'r2'
        reference_methods: list = [],  # Default [], Options are "PCA", "UMAP", "TSNE", "RandomFeature"
        split_type: str = "use-split",  # Default is "use-split", other options: "CV-5", ... "LOOCV"?
    ) -> Result:
        """TODO"""
        if self._evaluator is None:
            raise NotImplementedError("Evaluator not initialized")
        if self.result.model is None:
            raise NotImplementedError(
                "Model not trained. Please run the fit method first"
            )
        if not is_classifier(ml_model_class):
            warnings.warn(
                "The provided model is not a sklearn-type classifier. "
                "Evaluation continues but may produce incorrect results or errors."
            )
        if not is_regressor(ml_model_regression):
            warnings.warn(
                "The provided model is not a sklearn-type regressor. "
                "Evaluation continues but may produce incorrect results or errors."
            )

        self.result = self._evaluator.evaluate(
            datasets=self._datasets,
            result=self.result,
            ml_model_class=ml_model_class,
            ml_model_regression=ml_model_regression,
            params=params,
            metric_class=metric_class,
            metric_regression=metric_regression,
            reference_methods=reference_methods,
            split_type=split_type,
        )

        ml_plots = self._visualizer._plot_evaluation(result=self.result)

        return self.result

    def visualize(self, config: Optional[Union[None, DefaultConfig]] = None, **kwargs):
        """Creates visualizations of model results and performance.

        Args:
            config: Optional custom configuration for visualization.
            **kwargs: Additional configuration parameters.

        Raises:
            NotImplementedError: If visualizer is not initialized.
        """
        if self._visualizer is None:
            raise NotImplementedError("Visualizer not initialized")

        self._visualizer.visualize(result=self.result, config=self.config)

    def show_result(self):
        """Displays key visualizations of model results.

        This method generates the following visualizations:
        1. Loss Curves: Displays the absolute loss curves to provide insights into
           the model's training and validation performance over epochs.
        2. Latent Space Ridgeline Plot: Visualizes the distribution of the latent
           space representations across different dimensions, offering a high-level
           overview of the learned embeddings.
        3. Latent Space 2D Scatter Plot: Projects the latent space into two dimensions
           for a detailed view of the clustering or separation of data points.

        These visualizations help in understanding the model's performance and
        the structure of the latent space representations.
        """
        print("Creating plots ...")

        self._visualizer.show_loss(plot_type="absolute")

        self._visualizer.show_latent_space(result=self.result, plot_type="Ridgeline")

        self._visualizer.show_latent_space(result=self.result, plot_type="2D-scatter")

    def run(
        self, data: Optional[Union[DatasetContainer, DataPackage]] = None
    ) -> Result:
        """Executes the complete pipeline from preprocessing to visualization.

        Runs all pipeline steps in sequence and returns the result.

        Args:
            data: Optional data for prediction (overrides test data).

        Returns:
            Complete pipeline results.
        """
        self.preprocess()
        self.fit()
        self.predict(data=data)
        self.visualize()
        return self.result

    def save(self, file_path):
        """Saves the pipeline to a file.

        Args:
            file_path: Path where the pipeline should be saved.
        """
        saver = Saver(file_path)
        saver.save(self)

    @classmethod
    def load(cls, file_path) -> Any:
        """Loads a pipeline from a file.

        Args:
            file_path: Path to the saved pipeline.

        Returns:
            The loaded pipeline instance.
        """
        loader = Loader(file_path)
        return loader.load()

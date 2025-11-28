from typing import Dict, Optional, Type, Union, Literal
import numpy as np

import anndata as ad  # type: ignore
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_pipeline import BasePipeline
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._multimodal_dataset import MultiModalDataset
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.datapackage import DataPackage
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.data._xmodal_preprocessor import XModalPreprocessor
from autoencodix.evaluate._xmodalix_evaluator import XModalixEvaluator
from autoencodix.trainers._xmodal_trainer import XModalTrainer
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.xmodalix_config import XModalixConfig
from autoencodix.utils._losses import XModalLoss
from autoencodix.utils._utils import find_translation_keys
from autoencodix.visualize._xmodal_visualizer import XModalVisualizer

from mudata import MuData
import torch  # type: ignore


class XModalix(BasePipeline):
    """XModalix specific version of the BasePipeline class.

    Inherits preprocess, fit, evaluate, and visualize methods from BasePipeline.
    Overrides predict

    This class extends BasePipeline. See the parent class for a full list
    of attributes and methods.

    Additional Attributes:
        _default_config: Is set to XModalixConfig here.

    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = XModalTrainer,
        dataset_type: Type[BaseDataset] = MultiModalDataset,
        model_type: Type[
            BaseAutoencoder
        ] = VarixArchitecture,  # TODO make custom for XModalix
        loss_type: Type[BaseLoss] = XModalLoss,  # TODO make custom for XModalix
        preprocessor_type: Type[BasePreprocessor] = XModalPreprocessor,
        visualizer: Optional[Type[BaseVisualizer]] = XModalVisualizer,
        evaluator: Optional[Type[XModalixEvaluator]] = XModalixEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
    ) -> None:
        """See base class for full list of Args."""
        self._default_config = XModalixConfig()
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
        if not isinstance(self.config, XModalixConfig):
            raise TypeError(
                f"For XModalix Pipeline, we only allow XModalixConfig as type for config, got {type(self.config)}"
            )

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

        self.visualizer.show_loss(plot_type="absolute")

        self.visualizer.show_latent_space(result=self.result, plot_type="Ridgeline")

        self.visualizer.show_latent_space(result=self.result, plot_type="2D-scatter")

        dm_keys = find_translation_keys(
            config=self.config,
            trained_modalities=self._trainer._modality_dynamics.keys(),
        )
        if "IMG" in dm_keys["to"]:
            self.visualizer.show_image_translation(
                result=self.result,
                from_key=dm_keys["from"],
                to_key=dm_keys["to"],
                split="test",
            )

    def _process_latent_results(
        self, predictor_results: Result, predict_data: DatasetContainer
    ):
        """Processes latent space results into a single AnnData object for the source modality.

        This method identifies the source ('from') modality used for translation,
        extracts its latent space and sample IDs, and creates a single, informative
        AnnData object. The original feature names from the source dataset are
        also stored for reference.

        The final AnnData object is stored in `self.result.adata_latent`.

        Args:
            predictor_results: The Result object returned by the `predict` method.
            predict_data: The MultiModalDataset used for prediction, to access metadata.

        Raises:
            TypeError: if predicitonsresults arr no Dicts.
            ValueError: if no translate direction can be found or no latentspace is stored for the specified modality.
        """
        print("Processing latent space results into a single AnnData object...")
        all_latents = predictor_results.latentspaces.get(epoch=-1, split="test")
        all_sample_ids = predictor_results.sample_ids.get(epoch=-1, split="test")
        all_recons = predictor_results.reconstructions.get(epoch=-1, split="test")

        if not all(
            [isinstance(res, dict) for res in [all_latents, all_sample_ids, all_recons]]
        ):
            raise TypeError("Expected prediction results to be dictionaries.")

        try:
            predict_keys = find_translation_keys(
                config=self.config,
                trained_modalities=all_latents.keys(),  # type: ignore
            )
            from_key = predict_keys["from"]
        except Exception as e:
            # Provide a helpful error if the key can't be determined.
            raise ValueError(
                "Could not determine the 'from_key' for translation. "
                "Ensure the translation direction is specified in the config."
            ) from e

        print(f"Identified source modality for latent space: '{from_key}'")

        latent = all_latents.get(from_key)  # type: ignore
        sample_ids = all_sample_ids.get(from_key)  # type: ignore

        if latent is None:
            raise ValueError(
                f"Latent space for source modality '{from_key}' not found."
            )

        self.result.adata_latent = ad.AnnData(latent)
        self.result.adata_latent.var_names = [
            f"latent_{i}" for i in range(latent.shape[1])
        ]

        if sample_ids is not None:
            self.result.adata_latent.obs_names = sample_ids

        source_dataset = predict_data.test.datasets.get(from_key)  # type: ignore
        if (
            source_dataset
            and hasattr(source_dataset, "feature_ids")
            and source_dataset.feature_ids is not None
        ):
            self.result.adata_latent.uns["feature_ids"] = source_dataset.feature_ids
            print(
                f"  - Added {len(source_dataset.feature_ids)} source feature IDs to .uns"
            )

        self.result.update(predictor_results)
        print("Finished processing latent results.")

    def _validate_prediction_data(self, predict_data: BaseDataset):
        """Validate that prediction data has required test split.

        Args:
            predict_data: Dataset for prediciton
        Raises:
            ValueError: if prediciton data is empty.

        """
        if predict_data is None:
            raise ValueError(
                f"The data for prediction need to be a DatasetContainer with a test attribute containing a `BaseDataset` child"
                f"attribute, got: {predict_data}"
            )

    def _predict(
        self,
        predict_data: BaseDataset,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
        split: Literal["train", "valid", "test"] = "test",
    ):
        """Utility warpper of predict method of the trainer instance"""
        self._validate_prediction_data(predict_data=predict_data)
        return self._trainer.predict(
            data=predict_data,
            model=self.result.model,
            from_key=from_key,
            to_key=to_key,
            split=split,
        )  # type: ignore

    def predict(
        self,
        data: Optional[
            Union[
                DataPackage,
                DatasetContainer,
                ad.AnnData,
                MuData,  # ty: ignore[invalid-type-form]
            ]
        ] = None,
        config: Optional[Union[None, DefaultConfig]] = None,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
        predict_all: bool = False,
        **kwargs,
    ):
        """Generates predictions using the trained model.

        Uses the trained model to make predictions on test data or new data
        provided by the user. Processes the results and stores them in the
        result container.

        Args:
            data: Optional new data for predictions.
            config: Optional custom configuration for prediction.
            from_key: string indicator of 'from' translation direction.
            to_key: string indicator of 'to' translation direction.
            **kwargs: Additional configuration parameters as keyword arguments.

        Raises:
            NotImplementedError: If required components aren't initialized.
            ValueError: If no test data is available or data format is invalid.
        """
        self._validate_prediction_requirements()

        original_input = data
        predict_data = self._prepare_prediction_data(data=data)
        if predict_data.test is None:
            raise ValueError("No test data available for predictions.")
        predictor_results = self._predict(
            predict_data=predict_data.test,
            from_key=from_key,
            to_key=to_key,
            split="test",
        )

        self._process_latent_results(
            predictor_results=predictor_results, predict_data=predict_data
        )
        self._postprocess_reconstruction(
            predictor_results=predictor_results,
            original_input=original_input,
            predict_data=predict_data,
        )

        if predict_all:
            if self._datasets is None:
                raise ValueError(
                    "No training/validation data available for predictions."
                )
            train_pred_results = self._predict(
                predict_data=self._datasets.train,
                from_key=from_key,
                to_key=to_key,
                split="train",
            )
            self.result.update(other=train_pred_results)
            if self._datasets.valid is None:
                raise ValueError("No validation data available for predictions.")
            valid_pred_results = self._predict(
                predict_data=self._datasets.valid,
                from_key=from_key,
                to_key=to_key,
                split="valid",
            )
            self.result.update(other=valid_pred_results)

        return self.result

    def sample_latent_space(
        self,
        n_samples: int,
        split: str = "test",
        epoch: int = -1,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Sampling latent space is not implemented for XModalix."
        )

from typing import Dict, Optional, Type, Union
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
from autoencodix.evaluate.evaluate import Evaluator
from autoencodix.trainers._xmodal_trainer import XModalTrainer
from autoencodix.utils._result import Result
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.utils._losses import XModalLoss
from autoencodix.utils._utils import find_translation_keys
from autoencodix.visualize._xmodal_visualizer import XModalVisualizer


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
        trainer_type: Type[BaseTrainer] = XModalTrainer,
        dataset_type: Type[BaseDataset] = MultiModalDataset,
        model_type: Type[
            BaseAutoencoder
        ] = VarixArchitecture,  # TODO make custom for XModalix
        loss_type: Type[BaseLoss] = XModalLoss,  # TODO make custom for XModalix
        preprocessor_type: Type[BasePreprocessor] = XModalPreprocessor,
        visualizer: Optional[BaseVisualizer] = XModalVisualizer,
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

    def fit(self):  # TODO use from base
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

        dm_keys = find_translation_keys(config=self.config, trained_modalities=self._trainer._modality_dynamics.keys())
        if "IMG" in dm_keys["to"]:
            self._visualizer.show_image_translation(result=self.result, from_key=dm_keys["from"], to_key=dm_keys["to"], split="test")

    # def _process_latent_results(
    #     self, predictor_results, predict_data: DatasetContainer
    # ):
    #     """Process and store latent space results."""
    #     latent = predictor_results.latentspaces.get(epoch=-1, split="test")
    #     if isinstance(latent, dict):
    #         print("Detected dictionary in latent results, extracting array...")
    #         latent = next(iter(latent.values()))  # TODO better adjust for xmodal
    #     self.result.adata_latent = ad.AnnData(latent)
    #     # self.result.adata_latent.obs_names = predict_data.test.sample_ids  # type: ignore
    #     # self.result.adata_latent.uns["var_names"] = predict_data.test.feature_ids  # type: ignore
    #     self.result.update(predictor_results)

    def _process_latent_results(
        self, predictor_results: Result, predict_data: DatasetContainer
    ):
        """
        Processes latent space results into a single AnnData object for the source modality.

        This method identifies the source ('from') modality used for translation,
        extracts its latent space and sample IDs, and creates a single, informative
        AnnData object. The original feature names from the source dataset are
        also stored for reference.

        The final AnnData object is stored in `self.result.adata_latent`.

        Args:
            predictor_results: The Result object returned by the `predict` method.
            predict_data: The MultiModalDataset used for prediction, to access metadata.
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

from typing import Dict, Optional, Type, Union, List
import torch
import numpy as np

import anndata as ad  # type: ignore
from autoencodix.base._base_dataset import BaseDataset
from autoencodix.utils._utils import config_method
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
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.utils._result import Result
from autoencodix.configs.default_config import DefaultConfig

from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.data._stackix_preprocessor import StackixPreprocessor
from autoencodix.data._stackix_dataset import StackixDataset
from autoencodix.trainers._stackix_trainer import StackixTrainer
from autoencodix.visualize._general_visualizer import GeneralVisualizer


class Stackix(BasePipeline):
    """Stackix pipeline for training multiple VAEs on different modalities and stacking their latent spaces.

    This pipeline uses:
    1. StackixPreprocessor to prepare data for multi-modality training
    2. StackixTrainer to train individual VAEs, extract latent spaces, and train the final stacked model

    Like other pipelines, it follows the standard BasePipeline interface and workflow.

    Additional Attributes:
        _default_config: Is set to StackixConfig here.

    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = StackixTrainer,
        dataset_type: Type[BaseDataset] = StackixDataset,
        model_type: Type[BaseAutoencoder] = VarixArchitecture,
        loss_type: Type[BaseLoss] = VarixLoss,
        preprocessor_type: Type[BasePreprocessor] = StackixPreprocessor,
        visualizer: Type[BaseVisualizer] = GeneralVisualizer,
        evaluator: Optional[Type[BaseEvaluator]] = GeneralEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[DefaultConfig] = None,
        ontologies: Optional[Union[List, Dict]] = None,
    ) -> None:
        """Initialize the Stackix pipeline.

        See parent class for full list of Args.
        """
        self._default_config = StackixConfig()
        super().__init__(
            data=data,
            dataset_type=dataset_type
            or NumericDataset,  # Fallback, but not directly used
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
            ontologies=ontologies,
        )
        if not isinstance(self.config, StackixConfig):
            raise TypeError(
                f"For Stackix Pipeline, we only allow StackixConfig as type for config, got {type(self.config)}"
            )

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
    def sample_latent_space(
        self, config, split: str = "test", epoch: int = -1
    ) -> torch.Tensor:
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
        stacked_model = self._trainer.get_model()
        mu_t = mu_t.to(device=stacked_model.device, dtype=stacked_model.dtype)
        logvar_t = logvar_t.to(device=stacked_model.device, dtype=stacked_model.dtype)

        with self._trainer._trainer._fabric.autocast(), torch.no_grad():
            z = stacked_model.reparameterize(mu_t, logvar_t)
            return z

    def _process_latent_results(
        self, predictor_results: Result, predict_data: DatasetContainer
    ):
        """Processes the latent spaces from the StackixTrainer prediction results.

        Creates a correctly annotated AnnData object.
        This method overrides the BasePipeline implementation to specifically handle
        the aligned latent space from the unpaired/stacked workflow.


        Args:
            predictor_results: Result object after predict step
            predict_data: not used here, only to keep interface structure

        """
        latent = predictor_results.latentspaces.get(epoch=-1, split="test")
        sample_ids = predictor_results.sample_ids.get(epoch=-1, split="test")
        if latent is None:
            import warnings

            warnings.warn(
                "No latent space found in predictor results. Cannot create AnnData object."
            )
            return

        self.result.adata_latent = ad.AnnData(X=latent)
        self.result.adata_latent.obs_names = sample_ids
        self.result.adata_latent.var_names = [
            f"Latent_{i}" for i in range(latent.shape[1])  # ty: ignore
        ]

        # 4. Update the main result object with the rest of the prediction results.
        self.result.update(predictor_results)

        print("Successfully created annotated latent space object (adata_latent).")

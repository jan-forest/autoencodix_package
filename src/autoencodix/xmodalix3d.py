from typing import Dict, Optional, Type, Union

import numpy as np

from autoencodix.base._base_autoencoder import BaseAutoencoder
from autoencodix.base._base_dataset import BaseDataset, DataSetTypes
from autoencodix.base._base_loss import BaseLoss
from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.base._base_trainer import BaseTrainer
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.configs.xmodalix3d_config import XModalix3DConfig
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._multimodal_dataset import MultiModalDataset
from autoencodix.data._xmodal3d_preprocessor import XModal3DPreprocessor
from autoencodix.data.datapackage import DataPackage
from autoencodix.evaluate._xmodalix_evaluator import XModalixEvaluator
from autoencodix.modeling._varix_architecture import VarixArchitecture
from autoencodix.modeling._volumevae_architecture import VolumeVAEArchitecture
from autoencodix.trainers._xmodal_trainer import XModalTrainer
from autoencodix.utils._losses import XModalLoss
from autoencodix.utils._result import Result
from autoencodix.visualize._xmodal_visualizer import XModalVisualizer
from autoencodix.xmodalix import XModalix


class XModalix3D(XModalix):
    """3D image variant of XModalix.

    XModalix3D keeps the existing XModalix trainer, loss, multimodal dataset,
    evaluator, and prediction logic. The two 3D-specific substitutions are:

    1. XModal3DPreprocessor for volumetric NIfTI preprocessing.
    2. VolumeVAEArchitecture for DataSetTypes.IMG modalities.
    """

    def __init__(
        self,
        data: Optional[Union[DataPackage, DatasetContainer]] = None,
        trainer_type: Type[BaseTrainer] = XModalTrainer,
        dataset_type: Type[BaseDataset] = MultiModalDataset,
        model_type: Type[BaseAutoencoder] = VarixArchitecture,
        loss_type: Type[BaseLoss] = XModalLoss,
        preprocessor_type: Type[BasePreprocessor] = XModal3DPreprocessor,
        visualizer: Optional[Type[BaseVisualizer]] = XModalVisualizer,
        evaluator: Optional[Type[XModalixEvaluator]] = XModalixEvaluator,
        result: Optional[Result] = None,
        datasplitter_type: Type[DataSplitter] = DataSplitter,
        custom_splits: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[XModalix3DConfig] = None,
        model_map: Optional[
            Dict[DataSetTypes, Type[BaseAutoencoder]]
        ] = None,
    ) -> None:
        if config is None:
            config = XModalix3DConfig()

        if model_map is None:
            model_map = {
                DataSetTypes.NUM: VarixArchitecture,
                DataSetTypes.IMG: VolumeVAEArchitecture,
            }

        super().__init__(
            data=data,
            trainer_type=trainer_type,
            dataset_type=dataset_type,
            model_type=model_type,
            loss_type=loss_type,
            preprocessor_type=preprocessor_type,
            visualizer=visualizer,
            evaluator=evaluator,
            result=result,
            datasplitter_type=datasplitter_type,
            custom_splits=custom_splits,
            config=config,
            model_map=model_map,
        )

        # Keep the pipeline's declared default aligned with its actual config.
        self._default_config = XModalix3DConfig()

        if not isinstance(self.config, XModalix3DConfig):
            raise TypeError(
                "For XModalix3D, config must be XModalix3DConfig, "
                f"got {type(self.config)}"
            )

    def show_result(self):
        """Show XModalix plots that are dimension-agnostic.

        The existing XModalVisualizer.show_image_translation() is written for
        2D images and is therefore intentionally not called here.
        """

        print("Creating plots ...")
        self.visualizer.show_loss(plot_type="absolute")
        self.visualizer.show_latent_space(
            result=self.result,
            plot_type="Ridgeline",
        )
        self.visualizer.show_latent_space(
            result=self.result,
            plot_type="2D-scatter",
        )

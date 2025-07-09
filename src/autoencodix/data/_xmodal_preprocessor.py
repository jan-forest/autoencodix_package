from autoencodix.base._base_preprocessor import BasePreprocessor
from autoencodix.utils.default_config import DefaultConfig
from typing import Optional, Union, Tuple, Dict
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data.datapackage import DataPackage


class XModalPreprocessor(BasePreprocessor):
    """
    Preprocessor for cross-modal data, handling multiple data types and their transformations.
    Inherits from BasePreprocessor.
    """

    def __init__(
        self, config: DefaultConfig, ontologies: Optional[Union[Tuple, Dict]] = None
    ):
        super().__init__(config=config, ontologies=ontologies)
        self.data_config = config.data_config

    def preprocess(
        self, raw_user_data: Optional[DataPackage] = None
    ) -> DatasetContainer:
        """
        Preprocess the data according to the configuration.
        """
        self.dataset_dicts = self._general_preprocess()
        for k, v in self.dataset_dicts.items():
            print(f"key: {k}, type: {type(v)}")

        return DatasetContainer(train=None, test=None, valid=None)

    def format_reconstruction(self, reconstruction, result=None):
        pass

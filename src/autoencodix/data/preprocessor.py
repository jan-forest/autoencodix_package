from autoencodix.base._base_preprocessor import BasePreprocessor


# internal check done
# write tests: TODO
class GeneralPreprocessor(BasePreprocessor):
    """
    General Preprocessor classes that uses the general_preprocessing steps from BasePreprocessor.
    It then takes the split, cleaned, scaled and filtered datapackes and transforms it into a PyTorch
    Dataset, that can used in training. This class will mostly be used for the Vanillix and Varix 
    pipelines for numeric data.

    Attributes
    ----------
    - TODO
    

    Methods
    --------
    Preprocess
    
    """
    def __init__(self):
        super().__init__()

    def preprocess(self):
        self._datapackge = self._general_preprocess()
        return super().preprocess()


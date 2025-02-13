from enum import Enum
from typing import Optional, Tuple
from autoencodix.utils.default_config import DefaultConfig, DataInfo

"""
I've structured the possible data combinations that we support as of now (25/02/11)
To prevent the user from given unsupported data combinations, I've created a validator that checks the configuration
I've also determined 6 cased that we support, and I've created a function that determines the case based on the configuration
These casses help us later in the prerocessor to know which reader to call
My inital notes on what is allwoed:
*****No translation******
"Multi Single Cell"
# Description: if all numeric is single cell and no translation, we use this case
# RULES:
# ANNO  skipped, since ANNO comes from h5ad file and we don't allow mixed and sc data
# print warning when skipping
# All numeric data has to be single cell, raise error if not
# If translate_direction is not set, raise when other data types are present
# IMG, and other skipped
# -------------------------


"Mulit Bulk"
# Description: if all numeric is bulk and no translation, we use this case
# RULES:
# All numeric data has to be bulk, raise error if not
# ANNO is required and saved in obs
# each modality as an entry in obsm with the key being the name of the modality as defined in the config
# -------------------------



**** Translation paired****
# General description:
# if we have two different datasets and annotation, we can translate between them
# we can translate between bulk(1) <-> bulk(2)bulk <-> img, sc(1) <-> sc(2), sc <-> img
"Bulk<->Bulk"
# Description: if all numeric is bulk and translation is set, we use this case
# RULES:
# if we have only 1 datase where translate_direction is set, we raise an error
# if we have more than 2 datasets, we raise an error (except ANNOTATION)
# we require ANNOTATION
# we read like standard bulk data
----------------------------------



"IMG<->Bulk"
# Here we allow translation between bulk and img
# RULES:
# if we have only 1 datase where translate_direction is set, we raise an error
# if we have more than 2 datasets, we raise an error (except ANNOTATION)
# we require ANNOTATION (that maps img to bulk and including bulk and img metadata)
# only one numeric dataset allowed
# -------------------------


"Single Cell <-> Single Cell"
# Description: if all numeric is single cell and translation is set, we use this case
# RULES:
# if we have only 1 datase where translate_direction is set, we raise an error
# if we have more than 2 datasets, we raise an error (except ANNOTATION)
# we dont' require ANNOTATION, becuase we can use the h5ad file
# -------------------------


"Single Cell <-> IMG"
# Description: we need one Numiric that is single cell and one that is img
# RULES:
# The single cell is like standard case with a h5ad file
# adata.X is the single cell data from the h5ad file
# adata.obs is the annotation from the h5ad file
# adata.obsm["IMG"] is the img data

"""
class DataCase(str, Enum):
    MULTI_SINGLE_CELL = "Multi Single Cell"
    MULTI_BULK = "Multi Bulk"
    BULK_TO_BULK = "Bulk<->Bulk"
    IMG_TO_BULK = "IMG<->Bulk"
    SINGLE_CELL_TO_SINGLE_CELL = "Single Cell<->Single Cell"
    SINGLE_CELL_TO_IMG = "Single Cell<->IMG"


class ConfigValidationError(Exception):
    pass


class ConfigAnalyzer:
    """Class to analyze configuration and extract key information."""

    def __init__(self, config: DefaultConfig):
        self.config = config
        self.data_info = config.data_config.data_info

    def get_dataset_counts(self) -> Tuple[int, int, int]:
        """Get counts of different dataset types."""
        numeric_count = sum(
            1 for info in self.data_info.values() if info.data_type == "NUMERIC"
        )
        img_count = sum(
            1 for info in self.data_info.values() if info.data_type == "IMG"
        )
        anno_count = sum(
            1 for info in self.data_info.values() if info.data_type == "ANNOTATION"
        )
        return numeric_count, img_count, anno_count

    def get_translation_datasets(
        self,
    ) -> Tuple[Optional[Tuple[str, DataInfo]], Optional[Tuple[str, DataInfo]]]:
        """Get the 'from' and 'to' datasets for translation if they exist."""
        from_dataset = next(
            (
                (name, info)
                for name, info in self.data_info.items()
                if info.translate_direction == "from"
            ),
            None,
        )
        to_dataset = next(
            (
                (name, info)
                for name, info in self.data_info.items()
                if info.translate_direction == "to"
            ),
            None,
        )
        return from_dataset, to_dataset

    def is_translation_config(self) -> bool:
        """Check if this is a translation configuration."""
        return any(
            info.translate_direction is not None for info in self.data_info.values()
        )

    def check_numeric_consistency(self) -> bool:
        """Check if all numeric datasets are of the same type (single cell or bulk)."""
        numeric_datasets = [
            info for info in self.data_info.values() if info.data_type == "NUMERIC"
        ]
        if not numeric_datasets:
            return True
        is_single_cell = numeric_datasets[0].is_single_cell
        return all(info.is_single_cell == is_single_cell for info in numeric_datasets)


class ConfigValidator:
    """Class to validate configuration against specific rules."""

    def __init__(self, config: DefaultConfig):
        self.analyzer = ConfigAnalyzer(config)

    def validate_basic_requirements(self):
        """Validate basic configuration requirements."""
        numeric_count, _, _ = self.analyzer.get_dataset_counts()
        if numeric_count == 0:
            raise ConfigValidationError("At least one NUMERIC dataset is required")

        if not self.analyzer.check_numeric_consistency():
            raise ConfigValidationError(
                "All numeric datasets must be of the same type (either all single cell or all bulk)"
            )

    def validate_translation_config(self):
        """Validate translation-specific requirements."""
        from_dataset, to_dataset = self.analyzer.get_translation_datasets()

        if bool(from_dataset) != bool(to_dataset):
            raise ConfigValidationError(
                "Translation requires exactly one 'from' and one 'to' direction"
            )

        if (
            from_dataset
            and from_dataset[1].data_type == "NUMERIC"
            and to_dataset[1].data_type == "NUMERIC"
            and from_dataset[1].is_single_cell != to_dataset[1].is_single_cell
        ):
            raise ConfigValidationError(
                "Cannot translate between single cell and bulk data"
            )


def determine_case(config: DefaultConfig) -> DataCase:
    """
    Determine the appropriate case for the configuration without validation.

    Args:
        config: DefaultConfig object containing the data configuration

    Returns:
        DataCase: The determined case
    """
    analyzer = ConfigAnalyzer(config)
    numeric_count, img_count, anno_count = analyzer.get_dataset_counts()

    if analyzer.is_translation_config():
        from_dataset, to_dataset = analyzer.get_translation_datasets()

        if (
            from_dataset[1].data_type == "NUMERIC"
            and to_dataset[1].data_type == "NUMERIC"
        ):
            if from_dataset[1].is_single_cell:
                return DataCase.SINGLE_CELL_TO_SINGLE_CELL
            return DataCase.BULK_TO_BULK

        if "IMG" in {from_dataset[1].data_type, to_dataset[1].data_type}:
            numeric_dataset = (
                from_dataset if from_dataset[1].data_type == "NUMERIC" else to_dataset
            )
            if numeric_dataset[1].is_single_cell:
                return DataCase.SINGLE_CELL_TO_IMG
            return DataCase.IMG_TO_BULK
    else:
        # Non-translation cases
        numeric_dataset = next(
            info
            for info in config.data_config.data_info.values()
            if info.data_type == "NUMERIC"
        )
        if numeric_dataset.is_single_cell:
            return DataCase.MULTI_SINGLE_CELL
        return DataCase.MULTI_BULK


def validate_and_assign_case(config: DefaultConfig) -> DataCase:
    """
    Validate configuration and assign appropriate case.

    Args:
        config: DefaultConfig object

    Returns:
        DataCase: The validated case

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    validator = ConfigValidator(config)
    validator.validate_basic_requirements()
    analyzer = ConfigAnalyzer(config)

    if analyzer.is_translation_config():
        validator.validate_translation_config()

    return determine_case(config)

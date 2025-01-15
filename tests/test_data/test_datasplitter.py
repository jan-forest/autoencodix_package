import pytest

from autoencodix.data._datasplitter import DataSplitter

@pytest.fixture
def data_splitter():
    return DataSplitter()
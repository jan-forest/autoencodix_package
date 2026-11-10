import numpy as np
import pandas as pd
import pytest
import torch

import autoencodix as acx
from autoencodix.configs.default_config import DataCase
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.data.datapackage import DataPackage


@pytest.fixture
def trained_vanillix():
    rng = np.random.default_rng(0)
    samples = [f"s{i}" for i in range(120)]
    X = pd.DataFrame(
        rng.normal(size=(120, 30)),
        index=samples,
        columns=[f"g{j}" for j in range(30)],
    )
    anno = pd.DataFrame({"grp": rng.choice(["a", "b"], 120)}, index=samples)
    pipeline = acx.Vanillix(
        data=DataPackage(multi_bulk={"rna": X}, annotation={"paired": anno}),
        config=VanillixConfig(data_case=DataCase.MULTI_BULK, epochs=2, device="cpu"),
    )
    pipeline.preprocess()
    pipeline.fit()
    pipeline.predict()
    return pipeline


def _first_weight(model: torch.nn.Module) -> torch.Tensor:
    return next(model.parameters()).detach().clone()


@pytest.mark.parametrize("save_all", [False, True])
def test_save_does_not_modify_pipeline(trained_vanillix, tmp_path, save_all):
    # regression: save(save_all=False) used to re-initialize the in-memory model
    # and drop results and datasets of the caller's pipeline
    weights = _first_weight(trained_vanillix.result.model)
    n_latent = len(trained_vanillix.result.latentspaces._data)

    trained_vanillix.save(str(tmp_path / "pipe.pkl"), save_all=save_all)

    assert torch.equal(_first_weight(trained_vanillix.result.model), weights)
    assert len(trained_vanillix.result.latentspaces._data) == n_latent
    assert trained_vanillix._datasets is not None
    trained_vanillix.predict()  # still usable after saving


@pytest.mark.parametrize("save_all", [False, True])
def test_load_restores_trained_weights(trained_vanillix, tmp_path, save_all):
    weights = _first_weight(trained_vanillix.result.model)
    path = str(tmp_path / "pipe.pkl")
    trained_vanillix.save(path, save_all=save_all)

    loaded = acx.Vanillix.load(path)

    assert torch.equal(_first_weight(loaded.result.model), weights)

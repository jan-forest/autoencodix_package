import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoencodix.tuning import _conditional, _propose

FIXTURE_ARTIFACT = Path(__file__).parent / "fixtures" / "initial_configs.json"

SYNTHETIC_BUDGETS = [10, 50, 300]


def _best_log_lr(latent_dim):
    """Optimum of the synthetic varix benchmark: the best learning rate grows
    with latent_dim, so a conditional proposal must move it accordingly."""
    return -4.5 + 0.5 * np.log2(latent_dim)


@pytest.fixture
def best_log_lr():
    return _best_log_lr


def _synthetic_varix_tables(n_configs=400, seed=0):
    rng = np.random.default_rng(seed)
    space = {
        "k_filter": {"choices": [128, 512, 2048], "low": 128, "high": 2048},
        "n_layers": {"choices": [2, 3, 4], "low": 2, "high": 4},
        "enc_factor": {"choices": [1, 2, 4], "low": 1, "high": 4},
        "batch_size": {"choices": [32, 64, 128, 256], "low": 32, "high": 256},
        "learning_rate": {"scale": "log", "low": 1e-5, "high": 1e-1},
        "drop_p": {"scale": "linear", "low": 0.0, "high": 0.9},
        "weight_decay": {"scale": "log", "low": 1e-5, "high": 1e-1},
        "beta": {"scale": "log", "low": 1e-4, "high": 1.0},
        "latent_dim": {"choices": [2, 4, 8, 16, 32, 64], "low": 2, "high": 64},
    }
    frames = []
    for dataset in ("tcga", "schc"):
        hps = pd.DataFrame(
            {
                name: (
                    rng.choice(spec["choices"], n_configs)
                    if "choices" in spec
                    else (
                        10
                        ** rng.uniform(
                            np.log10(spec["low"]), np.log10(spec["high"]), n_configs
                        )
                        if spec["scale"] == "log"
                        else rng.uniform(spec["low"], spec["high"], n_configs)
                    )
                )
                for name, spec in space.items()
            }
        )
        lr_gap = (np.log10(hps["learning_rate"]) - _best_log_lr(hps["latent_dim"])) ** 2
        for task in (f"{dataset}_A", f"{dataset}_B"):
            df = hps.copy()
            df.insert(0, "config_id", np.arange(n_configs))
            df.insert(0, "task", task)
            df.insert(0, "dataset", dataset)
            df["downstream"] = (
                -lr_gap - 0.1 * hps["drop_p"] + rng.normal(0, 0.01, n_configs)
            )
            for budget in SYNTHETIC_BUDGETS:
                df[f"reconstruction_{budget}"] = (
                    lr_gap
                    + 0.05 * hps["n_layers"]
                    + 10 / budget
                    + rng.normal(0, 0.01, n_configs)
                )
            frames.append(df)
    search_space = {
        "constants": {
            "epochs": 300,
            "checkpoint_interval": 300,
            "loss_reduction": "sum",
        },
        "budgets": SYNTHETIC_BUDGETS,
        "architectures": {"varix": space},
    }
    return search_space, pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_repo(tmp_path_factory):
    """A local copy of the Hub repository layout: the trimmed portfolio JSON
    plus a synthetic varix evaluations table and search space."""
    root = tmp_path_factory.mktemp("initial-configs")
    search_space, table = _synthetic_varix_tables()
    (root / "evaluations").mkdir()
    table.to_parquet(root / "evaluations" / "varix.parquet", index=False)
    (root / "search_space.json").write_text(json.dumps(search_space))
    (root / "initial_configs.json").write_text(FIXTURE_ARTIFACT.read_text())
    return root


@pytest.fixture(autouse=True)
def mock_hf_download(monkeypatch, request, synthetic_repo):
    """Serve the local test repository instead of hitting the Hub, so tests run
    offline. Records each call's kwargs in the returned list. Tests marked
    `network` use the real Hub instead."""
    _propose._artifact_cache.clear()
    _conditional._surrogate_cache.clear()
    calls = []
    if request.node.get_closest_marker("network") is None:

        def fake_download(**kwargs):
            calls.append(kwargs)
            return str(synthetic_repo / kwargs["filename"])

        monkeypatch.setattr(_propose, "hf_hub_download", fake_download)
    yield calls
    _propose._artifact_cache.clear()
    _conditional._surrogate_cache.clear()

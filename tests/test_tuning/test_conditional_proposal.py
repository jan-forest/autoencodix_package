import math
import warnings
from pathlib import Path

import pytest
from huggingface_hub.utils import EntryNotFoundError

from autoencodix.configs.varix_config import VarixConfig
from autoencodix.tuning import _conditional, _propose, propose_initial_config

FIXTURE_ARTIFACT = Path(__file__).parent / "fixtures" / "initial_configs.json"


class TestFixedValues:
    def test_fixed_values_are_kept(self):
        config = propose_initial_config(
            "varix", fixed={"latent_dim": 16, "n_layers": 2}
        )
        assert isinstance(config, VarixConfig)
        assert config.latent_dim == 16
        assert config.n_layers == 2

    @pytest.mark.parametrize("latent_dim", [2, 64])
    def test_free_values_are_conditioned_on_fixed(self, latent_dim, best_log_lr):
        config = propose_initial_config("varix", fixed={"latent_dim": latent_dim})
        assert math.log10(config.learning_rate) == pytest.approx(
            best_log_lr(latent_dim), abs=0.75
        )

    def test_free_values_differ_from_static_override(self):
        conditional = propose_initial_config("varix", fixed={"latent_dim": 2})
        overridden = propose_initial_config("varix", latent_dim=2)
        assert conditional.learning_rate != overridden.learning_rate

    def test_proposal_is_deterministic(self):
        first = propose_initial_config("varix", fixed={"latent_dim": 8}, top_k=3)
        _conditional._surrogate_cache.clear()
        _propose._artifact_cache.clear()
        second = propose_initial_config("varix", fixed={"latent_dim": 8}, top_k=3)
        assert [c.model_dump_json() for c in first] == [
            c.model_dump_json() for c in second
        ]

    def test_integer_hyperparameters_are_ints(self):
        config = propose_initial_config("varix", fixed={"learning_rate": 1e-3})
        for name in ("k_filter", "n_layers", "batch_size", "latent_dim"):
            assert isinstance(getattr(config, name), int)

    def test_constants_are_applied(self):
        config = propose_initial_config("varix", fixed={"latent_dim": 8})
        assert config.epochs == 300
        assert config.checkpoint_interval == 300
        assert config.loss_reduction == "sum"

    def test_overrides_apply_after_fixed(self):
        config = propose_initial_config("varix", fixed={"latent_dim": 8}, drop_p=0.5)
        assert config.latent_dim == 8
        assert config.drop_p == 0.5

    def test_empty_fixed_uses_static_portfolio(self, mock_hf_download):
        assert (
            propose_initial_config("varix", fixed={}).model_dump()
            == propose_initial_config("varix").model_dump()
        )
        assert {call["filename"] for call in mock_hf_download} == {
            "initial_configs.json"
        }


class TestTopK:
    def test_top_k_returns_distinct_configs_with_fixed_values(self):
        configs = propose_initial_config("varix", fixed={"n_layers": 3}, top_k=3)
        assert len(configs) == 3
        assert all(c.n_layers == 3 for c in configs)
        assert len({c.model_dump_json() for c in configs}) == 3


class TestScopes:
    def test_single_dataset(self):
        config = propose_initial_config(
            "varix", dataset="tcga", fixed={"latent_dim": 4}
        )
        assert config.latent_dim == 4

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            propose_initial_config("varix", dataset="nope", fixed={"latent_dim": 4})

    def test_reconstruction_budget_snaps(self):
        with pytest.warns(UserWarning, match="snapped"):
            config = propose_initial_config(
                "varix",
                objective="reconstruction",
                budget_epochs=40,
                fixed={"latent_dim": 4},
            )
        assert config.epochs == 50
        assert config.checkpoint_interval == 50


class TestValidation:
    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="overrides"):
            propose_initial_config("varix", fixed={"epochs": 50})

    def test_out_of_range_value_warns(self):
        with pytest.warns(UserWarning, match="outside the range"):
            config = propose_initial_config("varix", fixed={"latent_dim": 128})
        assert config.latent_dim == 128

    def test_in_range_value_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            propose_initial_config("varix", fixed={"latent_dim": 20})

    def test_all_fixed_warns_and_returns_fixed(self):
        space = _propose._load("search_space.json", "r", "main", None)
        fixed = {
            name: spec["choices"][0] if "choices" in spec else spec["low"]
            for name, spec in space["architectures"]["varix"].items()
        }
        with pytest.warns(UserWarning, match="nothing left to propose"):
            config = propose_initial_config("varix", fixed=fixed)
        assert config.learning_rate == fixed["learning_rate"]


class TestSources:
    def test_local_artifact_path_reads_sibling_files(
        self, synthetic_repo, mock_hf_download
    ):
        config = propose_initial_config(
            "varix",
            fixed={"latent_dim": 8},
            artifact_path=synthetic_repo / "initial_configs.json",
        )
        assert config.latent_dim == 8
        assert mock_hf_download == []

    def test_local_artifact_path_without_evaluations_raises(self):
        with pytest.raises(FileNotFoundError, match="layout"):
            propose_initial_config(
                "varix", fixed={"latent_dim": 8}, artifact_path=FIXTURE_ARTIFACT
            )

    def test_revision_without_evaluations_raises_helpful_error(
        self, monkeypatch, synthetic_repo
    ):
        def old_revision_download(**kwargs):
            if kwargs["filename"] != "initial_configs.json":
                raise EntryNotFoundError("404 entry not found")
            return str(synthetic_repo / kwargs["filename"])

        monkeypatch.setattr(_propose, "hf_hub_download", old_revision_download)
        with pytest.raises(RuntimeError, match="before v2"):
            propose_initial_config("varix", revision="v1", fixed={"latent_dim": 8})

    def test_evaluations_are_fetched_for_requested_revision(self, mock_hf_download):
        propose_initial_config("varix", revision="v2", fixed={"latent_dim": 8})
        assert {call["filename"] for call in mock_hf_download} == {
            "initial_configs.json",
            "search_space.json",
            "evaluations/varix.parquet",
        }
        assert {call["revision"] for call in mock_hf_download} == {"v2"}


@pytest.mark.network
class TestRealHubConditional:
    def test_conditional_proposal_from_hub(self):
        config = propose_initial_config("varix", fixed={"latent_dim": 16})
        assert isinstance(config, VarixConfig)
        assert config.latent_dim == 16

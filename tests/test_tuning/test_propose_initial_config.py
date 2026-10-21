import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoencodix.configs.disentanglix_config import DisentanglixConfig
from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.tuning import list_portfolio_versions, propose_initial_config
from autoencodix.tuning._propose import (
    DEFAULT_FILENAME,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    _load_artifact,
)

FIXTURE_ARTIFACT = Path(__file__).parent / "fixtures" / "initial_configs.json"

COVERED = {
    "vanillix": VanillixConfig,
    "varix": VarixConfig,
    "ontix": OntixConfig,
    "disentanglix": DisentanglixConfig,
}


def _bounds(cls, field_name):
    """Read numeric ge/le constraints straight off the Pydantic field, rather
    than hardcoding them in the test."""
    field = cls.model_fields[field_name]
    ge = le = None
    for meta in field.metadata:
        if hasattr(meta, "ge"):
            ge = meta.ge
        if hasattr(meta, "le"):
            le = meta.le
    return ge, le


class TestProposeInitialConfigDefault:
    @pytest.mark.parametrize("architecture,config_cls", COVERED.items())
    def test_default_downstream_combined(self, architecture, config_cls):
        config = propose_initial_config(architecture)
        assert isinstance(config, config_cls)
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["epochs"] == 300

    @pytest.mark.parametrize("architecture", COVERED)
    def test_case_insensitive(self, architecture):
        config = propose_initial_config(architecture.upper())
        assert isinstance(config, COVERED[architecture])


class TestDatasetScoping:
    def test_known_dataset_still_works(self):
        artifact = _load_artifact()
        datasets = [k for k in artifact["varix"] if k != "combined"]
        assert datasets, "expected at least one non-combined dataset in the artifact"
        config = propose_initial_config("varix", dataset=datasets[0])
        assert isinstance(config, VarixConfig)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            propose_initial_config("varix", dataset="not_a_real_dataset")


class TestReconstructionObjective:
    def test_default_budget_is_full_length(self):
        config = propose_initial_config("varix", objective="reconstruction")
        assert config.epochs == 300
        assert config.checkpoint_interval == 300

    def test_low_budget_snaps_and_warns(self):
        with pytest.warns(UserWarning):
            config = propose_initial_config(
                "varix", objective="reconstruction", budget_epochs=40
            )
        assert config.epochs == 50
        assert config.checkpoint_interval == 50

    def test_exact_budget_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = propose_initial_config(
                "varix", objective="reconstruction", budget_epochs=300
            )
        assert config.epochs == 300

    def test_reduced_budget_reduces_epochs(self):
        config = propose_initial_config(
            "varix", objective="reconstruction", budget_epochs=10
        )
        assert config.epochs == 10
        assert config.checkpoint_interval == 10


class TestObjectiveValidation:
    def test_downstream_with_budget_epochs_raises(self):
        with pytest.raises(ValueError):
            propose_initial_config("varix", objective="downstream", budget_epochs=25)

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError):
            propose_initial_config("varix", objective="not_a_real_objective")


class TestBoundsAndOverrides:
    @pytest.mark.parametrize("architecture,config_cls", COVERED.items())
    def test_numeric_fields_within_schema_bounds(self, architecture, config_cls):
        config = propose_initial_config(architecture)
        for field_name in ("latent_dim", "n_layers", "batch_size", "learning_rate"):
            if field_name not in config_cls.model_fields:
                continue
            ge, le = _bounds(config_cls, field_name)
            value = getattr(config, field_name)
            if ge is not None:
                assert value >= ge
            if le is not None:
                assert value <= le

    def test_top_k_returns_multiple_distinct_configs(self):
        configs = propose_initial_config("varix", top_k=3)
        assert isinstance(configs, list)
        assert len(configs) == 3
        dumps = [c.model_dump_json() for c in configs]
        assert len(set(dumps)) > 1

    def test_overrides_apply(self):
        config = propose_initial_config("varix", latent_dim=7)
        assert config.latent_dim == 7

    def test_invalid_override_raises_validation_error(self):
        with pytest.raises(ValidationError):
            propose_initial_config("varix", latent_dim=-1)


class TestUnsupportedArchitecture:
    def test_unsupported_architecture_raises_by_default(self):
        with pytest.raises(ValueError):
            propose_initial_config("stackix")

    def test_unsupported_architecture_falls_back_with_warning(self):
        with pytest.warns(UserWarning):
            config = propose_initial_config("stackix", allow_fallback_to_defaults=True)
        from autoencodix.configs.stackix_config import StackixConfig

        assert isinstance(config, StackixConfig)

    def test_completely_unknown_architecture_raises_even_with_fallback(self):
        with pytest.raises(ValueError):
            propose_initial_config("not_a_real_architecture", allow_fallback_to_defaults=True)


class TestHubLoading:
    def test_default_downloads_main_from_default_repo(self, mock_hf_download):
        propose_initial_config("varix")
        assert mock_hf_download == [
            {
                "repo_id": DEFAULT_REPO_ID,
                "filename": DEFAULT_FILENAME,
                "repo_type": "dataset",
                "revision": DEFAULT_REVISION,
            }
        ]
        assert DEFAULT_REVISION == "main"

    def test_revision_and_repo_id_are_forwarded(self, mock_hf_download):
        propose_initial_config("varix", revision="v1", repo_id="someone/portfolio")
        assert mock_hf_download[0]["revision"] == "v1"
        assert mock_hf_download[0]["repo_id"] == "someone/portfolio"

    def test_artifact_is_cached_per_revision(self, mock_hf_download):
        propose_initial_config("varix")
        propose_initial_config("ontix")
        assert len(mock_hf_download) == 1
        propose_initial_config("varix", revision="v1")
        assert len(mock_hf_download) == 2

    def test_local_artifact_path_skips_hub(self, mock_hf_download):
        config = propose_initial_config("varix", artifact_path=FIXTURE_ARTIFACT)
        assert isinstance(config, VarixConfig)
        assert mock_hf_download == []

    def test_hub_kwargs_do_not_leak_into_config(self):
        config = propose_initial_config("varix", revision="v1")
        assert "revision" not in config.model_dump()

    def test_download_failure_raises_helpful_error(self, monkeypatch):
        from autoencodix.tuning import _propose

        def failing_download(**kwargs):
            raise ConnectionError("no network")

        monkeypatch.setattr(_propose, "hf_hub_download", failing_download)
        with pytest.raises(RuntimeError, match="artifact_path"):
            propose_initial_config("varix")


@pytest.mark.network
class TestRealHub:
    def test_default_portfolio_downloads(self):
        config = propose_initial_config("varix")
        assert isinstance(config, VarixConfig)

    def test_v1_tag_is_listed_and_loadable(self):
        assert "v1" in list_portfolio_versions()
        config = propose_initial_config("varix", revision="v1")
        assert config.epochs == 300


@pytest.mark.slow
class TestPackagingSmokeTest:
    def test_wheel_excludes_data_and_is_importable(self, tmp_path):
        repo_root = Path(__file__).resolve().parents[2]
        dist_dir = tmp_path / "dist"
        build = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            pytest.skip(f"could not build wheel: {build.stderr[-2000:]}")

        wheels = list(dist_dir.glob("*.whl"))
        assert wheels, "no wheel produced"
        wheel_path = wheels[0]

        import zipfile

        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
        assert not any(
            name.endswith("initial_configs.json") for name in names
        ), "initial_configs.json should be fetched from the Hub, not bundled"

        venv_dir = tmp_path / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)], check=True
        )
        venv_python = venv_dir / "bin" / "python"
        install = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--quiet", str(wheel_path)],
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            pytest.skip(f"could not install wheel: {install.stderr[-2000:]}")

        check = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import autoencodix as acx; "
                f"print(acx.propose_initial_config('varix', artifact_path={str(FIXTURE_ARTIFACT)!r}).epochs)",
            ],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr
        assert check.stdout.strip() == "300"

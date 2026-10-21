"""Static, transfer-learning-derived initial-config proposals for autoencodix models.

The portfolios returned here were computed *offline* by fitting Syne Tune's
``ZeroShotTransfer`` scheduler against the BBOmix benchmark (105k training runs
across vanillix/varix/ontix/disentanglix). Since a caller of
:func:`propose_initial_config` has zero evaluations of their own yet, there is no
live optimizer state to maintain at call time. This module is a plain JSON
lookup plus Pydantic validation and never imports ``syne-tune``.

The JSON lookup table is not bundled with the package. It is hosted as a
Hugging Face dataset (``autoencodix/initial-configs``) and downloaded on first
use via ``huggingface_hub``, which caches it locally (so later calls, and calls
with ``HF_HUB_OFFLINE=1``, need no network). New portfolio versions are
published there as git tags (``v1``, ``v2``, ...); callers select one via
``revision``, point at another repo via ``repo_id``, or bypass the Hub entirely
with ``artifact_path``.

Passing ``fixed={...}`` switches from the static lookup to a conditional
proposal: the free hyperparameters are chosen *given* the fixed ones by a
surrogate fitted on the raw BBOmix evaluations, which are published in the same
Hugging Face repository (see ``_conditional.py``).
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_refs
from huggingface_hub.utils import EntryNotFoundError

from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.disentanglix_config import DisentanglixConfig
from autoencodix.configs.maskix_config import MaskixConfig
from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.configs.xmodalix_config import XModalixConfig
from autoencodix.tuning._conditional import propose_conditional, validate_fixed

Objective = Literal["downstream", "reconstruction"]

# Every architecture that has a dedicated Config class, used only as the
# fallback target for `allow_fallback_to_defaults=True`. Whether an architecture
# additionally has a BBOmix-derived portfolio is determined dynamically from the
# loaded artifact (see `_covered_architectures`), not hardcoded here.
_CONFIG_CLASSES: Dict[str, type] = {
    "vanillix": VanillixConfig,
    "varix": VarixConfig,
    "ontix": OntixConfig,
    "disentanglix": DisentanglixConfig,
    "stackix": StackixConfig,
    "xmodalix": XModalixConfig,
    "maskix": MaskixConfig,
}

DEFAULT_REPO_ID = "autoencodix/initial-configs"
DEFAULT_FILENAME = "initial_configs.json"
# "main" always points at the newest published portfolio. Pin a tag (e.g. "v1")
# or a commit hash for reproducible proposals.
DEFAULT_REVISION = "main"
# Only needed for conditional proposals (`fixed=...`); published from v2 on.
SEARCH_SPACE_FILENAME = "search_space.json"
EVALUATIONS_DIR = "evaluations"

# Keyed by (repo_id, revision, artifact_path, filename) so different portfolio
# versions can be used side by side within one session.
_artifact_cache: Dict[Tuple[str, str, Optional[str], str], Any] = {}


def _fetch(
    filename: str,
    repo_id: str,
    revision: str,
    artifact_path: Optional[Union[str, Path]],
) -> Path:
    """Local path of `filename` from the Hub, or from the same layout next to
    `artifact_path` when one is given."""
    if artifact_path is not None:
        path = Path(artifact_path)
        if filename != DEFAULT_FILENAME:
            path = path.parent / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. With artifact_path, conditional proposals "
                f"expect the Hugging Face repository layout next to the JSON "
                f"({SEARCH_SPACE_FILENAME}, {EVALUATIONS_DIR}/<architecture>.parquet)."
            )
        return path
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
            )
        )
    except EntryNotFoundError as e:
        raise RuntimeError(
            f"{filename!r} is not in the Hugging Face dataset {repo_id!r} at "
            f"revision {revision!r}. Conditional proposals (`fixed=...`) need "
            "the BBOmix evaluation tables, which portfolio versions before v2 "
            "don't include; pass a revision that has them (see "
            "list_portfolio_versions())."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Could not download {filename!r} from the Hugging Face "
            f"dataset {repo_id!r} at revision {revision!r}: {e}\n"
            "If you are offline, either run once with network access (the "
            "file is then cached and HF_HUB_OFFLINE=1 works), or download "
            "the repository files manually and pass artifact_path=<path to json>."
        ) from e


def _load(
    filename: str,
    repo_id: str,
    revision: str,
    artifact_path: Optional[Union[str, Path]],
) -> Any:
    local_key = None if artifact_path is None else str(Path(artifact_path).resolve())
    cache_key = (repo_id, revision, local_key, filename)
    if cache_key not in _artifact_cache:
        path = _fetch(filename, repo_id, revision, artifact_path)
        if filename.endswith(".parquet"):
            _artifact_cache[cache_key] = pd.read_parquet(path)
        else:
            _artifact_cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
    return _artifact_cache[cache_key]


def _load_artifact(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    artifact_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    return _load(DEFAULT_FILENAME, repo_id, revision, artifact_path)


def list_portfolio_versions(repo_id: str = DEFAULT_REPO_ID) -> List[str]:
    """List the tagged portfolio versions available on the Hugging Face Hub.

    Any returned tag can be passed as ``revision`` to
    :func:`propose_initial_config`. ``"main"`` (the default) is always
    available as well and points at the newest portfolio.

    Args:
        repo_id: Hugging Face dataset repository holding the portfolio.

    Returns:
        Sorted list of tag names, e.g. ``["v1", "v2"]``.
    """
    refs = list_repo_refs(repo_id, repo_type="dataset")
    return sorted(tag.name for tag in refs.tags)


def _covered_architectures(artifact: Dict[str, Any]) -> List[str]:
    return sorted(set(artifact) & set(_CONFIG_CLASSES))


def _resolve_scope(
    artifact: Dict[str, Any], architecture: str, arch_key: str, dataset: Optional[str]
) -> Dict[str, Any]:
    arch_entry = artifact[arch_key]
    scope_key = "combined" if dataset is None else dataset
    if scope_key not in arch_entry:
        available = sorted(k for k in arch_entry if k != "combined")
        raise ValueError(
            f"Unknown dataset {dataset!r} for architecture {architecture!r}. "
            f"Available datasets: {available}. Omit `dataset` (or pass None) to "
            "use the combined, cross-dataset portfolio -- the recommended default "
            "for a new dataset that wasn't part of the BBOmix benchmark."
        )
    return arch_entry[scope_key]


def _choose_budget(available_budgets: List[int], budget_epochs: Optional[int]) -> int:
    available_budgets = sorted(available_budgets)
    if budget_epochs is None:
        return max(available_budgets)
    chosen_budget = min(available_budgets, key=lambda b: abs(b - budget_epochs))
    if chosen_budget != budget_epochs:
        warnings.warn(
            f"budget_epochs={budget_epochs} is not one of the available "
            f"budgets {available_budgets}; snapped to {chosen_budget}.",
            UserWarning,
            stacklevel=3,
        )
    return chosen_budget


def _propose_given_fixed(
    architecture: str,
    arch_key: str,
    fixed: Dict[str, Any],
    dataset: Optional[str],
    objective: Objective,
    budget_epochs: Optional[int],
    top_k: int,
    repo_id: str,
    revision: str,
    artifact_path: Optional[Union[str, Path]],
) -> List[Dict[str, Any]]:
    search_space = _load(SEARCH_SPACE_FILENAME, repo_id, revision, artifact_path)
    space = search_space["architectures"][arch_key]
    validate_fixed(architecture, fixed, space)
    table = _load(
        f"{EVALUATIONS_DIR}/{arch_key}.parquet", repo_id, revision, artifact_path
    )

    constants = dict(search_space["constants"])
    if objective == "downstream":
        target, maximize = "downstream", True
    else:
        budget = _choose_budget(search_space["budgets"], budget_epochs)
        target, maximize = f"reconstruction_{budget}", False
        constants.update(epochs=budget, checkpoint_interval=budget)

    datasets = [dataset] if dataset is not None else sorted(table["dataset"].unique())
    local_key = None if artifact_path is None else str(Path(artifact_path).resolve())
    proposals = propose_conditional(
        arch_key,
        fixed,
        space,
        table,
        datasets,
        target,
        maximize,
        top_k,
        cache_prefix=(repo_id, revision, local_key),
    )
    return [{**constants, **hp} for hp in proposals]


def propose_initial_config(
    architecture: str,
    dataset: Optional[str] = None,
    objective: Objective = "downstream",
    budget_epochs: Optional[int] = None,
    top_k: int = 1,
    allow_fallback_to_defaults: bool = False,
    revision: str = DEFAULT_REVISION,
    repo_id: str = DEFAULT_REPO_ID,
    artifact_path: Optional[Union[str, Path]] = None,
    fixed: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Union[DefaultConfig, List[DefaultConfig]]:
    """Propose a strong initial config for an autoencodix architecture.

    The proposal is a ranked portfolio entry derived offline from the BBOmix
    benchmark via zero-shot transfer learning. It is not a config tuned on any
    evaluations of your own.

    To keep some hyperparameters at values of your choice, pass them as
    ``fixed``: the remaining hyperparameters are then proposed *given* those
    values, using a surrogate fitted on the raw BBOmix evaluations::

        propose_initial_config("varix", fixed={"latent_dim": 16, "n_layers": 2})

    Keyword ``**overrides`` differ: they only overwrite fields of the proposal
    afterwards, so the other proposed values stay the ones chosen without them.

    Args:
        architecture: Model architecture, e.g. ``"varix"`` (case-insensitive).
        dataset: Restrict the source portfolio to one BBOmix dataset (whatever
            datasets happen to be in the loaded artifact, e.g. ``"tcga"`` or
            ``"schc"``). Defaults to ``None``, which uses the ``"combined"``
            portfolio pooling all available source tasks -- this is the
            recommended default, since it is the one expected to generalize to
            a dataset that wasn't part of the benchmark.
        objective: ``"downstream"`` (maximize aggregate downstream task
            performance, final-epoch only) or ``"reconstruction"`` (minimize
            reconstruction loss, with a `budget_epochs` fidelity axis).
        budget_epochs: Only valid for ``objective="reconstruction"``. Snapped to
            the nearest budget actually present in the artifact (a `UserWarning`
            is emitted on snap); defaults to the largest available budget
            (full training length) when omitted.
        top_k: Number of ranked configs to return. Returns a single config when
            ``top_k=1`` (the default), otherwise a list.
        allow_fallback_to_defaults: If the architecture has no BBOmix-derived
            portfolio but does have a Config class, emit a `UserWarning` and
            return that class's plain schema defaults instead of raising.
        revision: Version of the portfolio on the Hugging Face Hub: a tag
            (e.g. ``"v1"``, see :func:`list_portfolio_versions`), a branch, or a
            commit hash. Defaults to ``"main"``, the newest published
            portfolio -- note that proposals may change when a new version is
            published, so pin a tag for reproducibility.
        repo_id: Hugging Face dataset repository to load the portfolio from.
            Defaults to ``"autoencodix/initial-configs"``; set this to use your
            own portfolio with the same JSON layout.
        artifact_path: Path to a local portfolio JSON. When given, the Hub is
            not contacted and `revision`/`repo_id` are ignored. With `fixed`,
            the evaluation files are read from the same layout as the Hub
            repository next to it.
        fixed: Hyperparameters to hold at the given values, e.g.
            ``{"latent_dim": 16}``. The other tuned hyperparameters are proposed
            conditionally on them (deterministic; the first call per
            architecture fits a surrogate and takes a few seconds). Only
            hyperparameters tuned in the BBOmix benchmark can be fixed; values
            outside the benchmarked range emit a `UserWarning`.
        **overrides: Applied on top of the proposed hyperparameters before
            constructing the Config object, without changing the other proposed
            values; invalid overrides raise `pydantic.ValidationError`.

    Returns:
        A single Config instance, or a list of `top_k` Config instances.

    Raises:
        ValueError: Unknown/unsupported architecture, unknown dataset, an
            invalid combination of `objective`/`budget_epochs`, or a `fixed`
            key that isn't a tuned hyperparameter of the architecture.
        RuntimeError: The portfolio could not be downloaded from the Hub.
    """
    artifact = _load_artifact(
        repo_id=repo_id, revision=revision, artifact_path=artifact_path
    )
    arch_key = architecture.strip().lower()
    covered = _covered_architectures(artifact)

    if arch_key not in covered:
        if allow_fallback_to_defaults and arch_key in _CONFIG_CLASSES:
            warnings.warn(
                f"No BBOmix-derived portfolio for architecture {architecture!r}; "
                f"falling back to {_CONFIG_CLASSES[arch_key].__name__} schema "
                "defaults. Supported architectures with a portfolio: "
                f"{covered}.",
                UserWarning,
                stacklevel=2,
            )
            return _CONFIG_CLASSES[arch_key](**overrides)
        raise ValueError(
            f"Unsupported architecture {architecture!r}. propose_initial_config "
            f"has a BBOmix-derived portfolio for: {covered}. Pass "
            "allow_fallback_to_defaults=True to fall back to plain schema "
            "defaults for any other architecture with a Config class."
        )

    if objective not in ("downstream", "reconstruction"):
        raise ValueError(
            f"objective must be 'downstream' or 'reconstruction', got {objective!r}."
        )

    if objective == "downstream" and budget_epochs is not None:
        raise ValueError(
            "budget_epochs is only meaningful for objective='reconstruction' -- "
            "there is no per-epoch signal for downstream performance in the "
            "benchmark data, so it has nothing to act on here."
        )

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    scope = _resolve_scope(artifact, architecture, arch_key, dataset)

    if fixed:
        selected = _propose_given_fixed(
            architecture,
            arch_key,
            fixed,
            dataset,
            objective,
            budget_epochs,
            top_k,
            repo_id,
            revision,
            artifact_path,
        )
    elif objective == "downstream":
        selected = scope["downstream"][:top_k]
    else:
        recon = scope["reconstruction"]
        chosen_budget = _choose_budget([int(b) for b in recon], budget_epochs)
        selected = recon[str(chosen_budget)][:top_k]

    config_cls = _CONFIG_CLASSES[arch_key]
    configs = [config_cls(**{**hp, **overrides}) for hp in selected]

    return configs[0] if top_k == 1 else configs

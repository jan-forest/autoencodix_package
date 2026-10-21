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
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from huggingface_hub import hf_hub_download, list_repo_refs

from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.disentanglix_config import DisentanglixConfig
from autoencodix.configs.maskix_config import MaskixConfig
from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.configs.xmodalix_config import XModalixConfig

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

# Keyed by (repo_id, revision, artifact_path) so different portfolio versions
# can be used side by side within one session.
_artifact_cache: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}


def _load_artifact(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    artifact_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    local_key = None if artifact_path is None else str(Path(artifact_path).resolve())
    cache_key = (repo_id, revision, local_key)
    if cache_key in _artifact_cache:
        return _artifact_cache[cache_key]

    if artifact_path is not None:
        path = Path(artifact_path)
    else:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=DEFAULT_FILENAME,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not download {DEFAULT_FILENAME!r} from the Hugging Face "
                f"dataset {repo_id!r} at revision {revision!r}: {e}\n"
                "If you are offline, either run once with network access (the "
                "file is then cached and HF_HUB_OFFLINE=1 works), or download "
                "the JSON manually and pass artifact_path=<path to json>."
            ) from e

    _artifact_cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
    return _artifact_cache[cache_key]


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
    **overrides: Any,
) -> Union[DefaultConfig, List[DefaultConfig]]:
    """Propose a strong initial config for an autoencodix architecture.

    The proposal is a ranked portfolio entry derived offline from the BBOmix
    benchmark via zero-shot transfer learning. It is not a config tuned on any
    evaluations of your own.

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
            not contacted and `revision`/`repo_id` are ignored.
        **overrides: Applied on top of the proposed hyperparameters before
            constructing the Config object; invalid overrides raise
            `pydantic.ValidationError`.

    Returns:
        A single Config instance, or a list of `top_k` Config instances.

    Raises:
        ValueError: Unknown/unsupported architecture, unknown dataset, or an
            invalid combination of `objective`/`budget_epochs`.
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

    if objective == "downstream":
        candidates = scope["downstream"]
    else:
        recon = scope["reconstruction"]
        available_budgets = sorted(int(b) for b in recon)
        if budget_epochs is None:
            chosen_budget = max(available_budgets)
        else:
            chosen_budget = min(
                available_budgets, key=lambda b: abs(b - budget_epochs)
            )
            if chosen_budget != budget_epochs:
                warnings.warn(
                    f"budget_epochs={budget_epochs} is not one of the available "
                    f"budgets {available_budgets}; snapped to {chosen_budget}.",
                    UserWarning,
                    stacklevel=2,
                )
        candidates = recon[str(chosen_budget)]

    selected = candidates[: min(top_k, len(candidates))]
    config_cls = _CONFIG_CLASSES[arch_key]
    configs = [config_cls(**{**hp, **overrides}) for hp in selected]

    return configs[0] if top_k == 1 else configs

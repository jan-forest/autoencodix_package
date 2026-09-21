"""Static, transfer-learning-derived initial-config proposals for autoencodix models.

The portfolios returned here were computed *offline* (see
``benchmarking/tuning/generate_initial_configs.py``) by fitting Syne Tune's
``ZeroShotTransfer`` scheduler against the BBOmix benchmark (105k training runs
across vanillix/varix/ontix/disentanglix). Since a caller of
:func:`propose_initial_config` has zero evaluations of their own yet, there is no
live optimizer state to maintain at call time -- this module is a plain JSON
lookup plus Pydantic validation, and never imports ``syne-tune``.

Optional future extension (not implemented here): if a caller wants the proposal
refined using a handful of their *own* live training runs, that is a distinct
"suggest next trial" mode requiring a real warm-started online scheduler
(e.g. Syne Tune's ``BoundingBox`` / ``QuantileBasedSurrogateSearcher``) and
``syne-tune`` as a genuine runtime dependency (a future ``autoencodix[tuning]``
extra).
"""

import json
import warnings
from importlib import resources
from typing import Any, Dict, List, Literal, Optional, Union

from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.disentanglix_config import DisentanglixConfig
from autoencodix.configs.maskix_config import MaskixConfig
from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.configs.xmodalix_config import XModalixConfig

Objective = Literal["downstream", "reconstruction"]

# Every architecture that *has* a dedicated Config class, used only as the
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

_ARTIFACT_PACKAGE = "autoencodix.tuning.data"
_ARTIFACT_FILENAME = "initial_configs.json"

_artifact_cache: Optional[Dict[str, Any]] = None


def _load_artifact() -> Dict[str, Any]:
    global _artifact_cache
    if _artifact_cache is None:
        data_text = (
            resources.files(_ARTIFACT_PACKAGE)
            .joinpath(_ARTIFACT_FILENAME)
            .read_text(encoding="utf-8")
        )
        _artifact_cache = json.loads(data_text)
    return _artifact_cache


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
    **overrides: Any,
) -> Union[DefaultConfig, List[DefaultConfig]]:
    """Propose a strong initial config for an autoencodix architecture.

    The proposal is a ranked portfolio entry derived offline from the BBOmix
    benchmark via zero-shot transfer learning -- not a config tuned on any
    evaluations of your own.

    Args:
        architecture: Model architecture, e.g. ``"varix"`` (case-insensitive).
        dataset: Restrict the source portfolio to one BBOmix dataset (whatever
            datasets happen to be in the shipped artifact, e.g. ``"tcga"`` or
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
        **overrides: Applied on top of the proposed hyperparameters before
            constructing the Config object; invalid overrides raise
            `pydantic.ValidationError`.

    Returns:
        A single Config instance, or a list of `top_k` Config instances.

    Raises:
        ValueError: Unknown/unsupported architecture, unknown dataset, or an
            invalid combination of `objective`/`budget_epochs`.
    """
    artifact = _load_artifact()
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

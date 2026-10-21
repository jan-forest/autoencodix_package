"""Conditional initial-config proposals: fix some hyperparameters, propose the rest.

The static portfolios in ``initial_configs.json`` are ranked unconditionally, so
overriding one hyperparameter leaves the others tuned for a different value. To
propose the free hyperparameters *given* the fixed ones, this module fits a
random-forest surrogate on the raw BBOmix evaluations (published next to the
portfolio as ``evaluations/<architecture>.parquet``, see
``benchmarking/tuning/export_evaluations.py``) and searches the free
hyperparameters with the fixed ones held at the caller's values.

Scores are converted to per-task ranks in [0, 1] (higher is better) so tasks with
different metric scales are comparable, and ``top_k`` configs are picked greedily
the way Syne Tune's ``ZeroShotTransfer`` builds its portfolios: each next config
maximizes the mean over tasks of the best rank achieved by any config picked so
far, which keeps the returned configs complementary rather than near-duplicates.
"""

import warnings
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

N_RANDOM_CANDIDATES = 4096
_SEED = 0

_surrogate_cache: Dict[Hashable, Tuple[RandomForestRegressor, List[str]]] = {}


def validate_fixed(
    architecture: str, fixed: Dict[str, Any], space: Dict[str, Dict[str, Any]]
) -> None:
    unknown = sorted(set(fixed) - set(space))
    if unknown:
        raise ValueError(
            f"Cannot fix {unknown} for architecture {architecture!r}: only the "
            f"hyperparameters tuned in the BBOmix benchmark can be fixed: "
            f"{sorted(space)}. Use keyword overrides to set any other Config field, "
            "and `budget_epochs` (with objective='reconstruction') for the "
            "training length."
        )
    for name, value in fixed.items():
        low, high = space[name]["low"], space[name]["high"]
        if not low <= value <= high:
            warnings.warn(
                f"fixed {name}={value!r} is outside the range covered by the "
                f"BBOmix benchmark [{low}, {high}]; the proposal for the other "
                f"hyperparameters is extrapolated as if {name} were at the "
                "nearest edge of that range.",
                UserWarning,
                stacklevel=4,
            )


def _task_ranks(table: pd.DataFrame, target: str, maximize: bool) -> pd.DataFrame:
    """config_id x task matrix of per-task ranks in [0, 1], higher is better."""
    ranks = table.groupby("task")[target].rank(pct=True, ascending=maximize)
    return (
        table.assign(rank=ranks)
        .pivot(index="config_id", columns="task", values="rank")
        .sort_index()
    )


def _fit_surrogate(
    cache_key: Hashable,
    table: pd.DataFrame,
    hp_names: List[str],
    target: str,
    maximize: bool,
) -> Tuple[RandomForestRegressor, List[str]]:
    if cache_key not in _surrogate_cache:
        ranks = _task_ranks(table, target, maximize)
        hps = table.drop_duplicates("config_id").set_index("config_id").sort_index()
        model = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=_SEED
        )
        model.fit(hps.loc[ranks.index, hp_names].to_numpy(), ranks.to_numpy())
        _surrogate_cache[cache_key] = (model, list(ranks.columns))
    return _surrogate_cache[cache_key]


def _candidates(
    table: pd.DataFrame,
    space: Dict[str, Dict[str, Any]],
    fixed: Dict[str, Any],
) -> pd.DataFrame:
    """Random samples of the free hyperparameters plus every evaluated
    combination of them, with the fixed hyperparameters held at their values."""
    rng = np.random.default_rng(_SEED)
    free = [name for name in space if name not in fixed]
    sampled: Dict[str, Any] = {}
    for name in free:
        spec = space[name]
        if "choices" in spec:
            sampled[name] = rng.choice(spec["choices"], size=N_RANDOM_CANDIDATES)
        elif spec["scale"] == "log":
            sampled[name] = 10 ** rng.uniform(
                np.log10(spec["low"]), np.log10(spec["high"]), N_RANDOM_CANDIDATES
            )
        else:
            sampled[name] = rng.uniform(spec["low"], spec["high"], N_RANDOM_CANDIDATES)
    candidates = pd.concat(
        [pd.DataFrame(sampled), table[free]], ignore_index=True
    ).drop_duplicates(ignore_index=True)
    for name, value in fixed.items():
        candidates[name] = value
    return candidates[list(space)]


def _greedy_portfolio(scores: np.ndarray, top_k: int) -> List[int]:
    picked: List[int] = []
    best: Optional[np.ndarray] = None
    # Ties in the coverage gain (common once a few configs are picked) are broken
    # by the candidate's own mean rank.
    tiebreak = 1e-6 * scores.mean(axis=1)
    for _ in range(min(top_k, len(scores))):
        if best is None:
            gain = scores.mean(axis=1)
        else:
            gain = np.maximum(scores, best).mean(axis=1) + tiebreak
        gain[picked] = -np.inf
        idx = int(np.argmax(gain))
        picked.append(idx)
        best = scores[idx] if best is None else np.maximum(best, scores[idx])
    return picked


def propose_conditional(
    architecture: str,
    fixed: Dict[str, Any],
    space: Dict[str, Dict[str, Any]],
    table: pd.DataFrame,
    datasets: List[str],
    target: str,
    maximize: bool,
    top_k: int,
    cache_prefix: Hashable,
) -> List[Dict[str, Any]]:
    """Return up to `top_k` ranked hyperparameter dicts for the free
    hyperparameters, each merged with `fixed`.

    Args:
        architecture: Lower-case architecture name (for messages and caching).
        fixed: Hyperparameters to hold at the given values; validated by the caller.
        space: The architecture's entry in ``search_space.json``.
        table: The architecture's evaluations table.
        datasets: BBOmix datasets whose tasks the proposal should do well on.
        target: Score column, ``"downstream"`` or ``"reconstruction_<budget>"``.
        maximize: Whether higher `target` values are better.
        top_k: Number of configs to return.
        cache_prefix: Identifies the source of `table` (repo, revision, path) so
            fitted surrogates are reused only for the same data.
    """
    hp_names = list(space)
    if len(fixed) == len(hp_names):
        warnings.warn(
            f"All tunable hyperparameters of {architecture!r} are fixed; there is "
            "nothing left to propose.",
            UserWarning,
            stacklevel=4,
        )
        return [dict(fixed)]

    candidates = _candidates(table[table["dataset"].isin(datasets)], space, fixed)
    X = candidates.to_numpy()
    scores = []
    for dataset in datasets:
        model, _ = _fit_surrogate(
            (cache_prefix, architecture, dataset, target),
            table[table["dataset"] == dataset],
            hp_names,
            target,
            maximize,
        )
        pred = model.predict(X)
        scores.append(pred.reshape(len(X), -1))
    picked = _greedy_portfolio(np.hstack(scores), top_k)

    proposals = []
    for idx in picked:
        row = candidates.iloc[idx]
        hp = {
            name: int(row[name]) if "choices" in space[name] else float(row[name])
            for name in hp_names
            if name not in fixed
        }
        proposals.append({**hp, **fixed})
    return proposals

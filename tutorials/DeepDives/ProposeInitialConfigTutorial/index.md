# Propose an Initial Config[¶](#propose-an-initial-config)

`autoencodix.propose_initial_config()` gives you a strong starting hyperparameter config for `Vanillix`, `Varix`, `Ontix`, and `Disentanglix` instead of hand-picking one or accepting the generic schema defaults.

Under the hood, it's a lookup into a small ranked portfolio computed **offline** via Syne Tune's zero-shot transfer-learning tooling against the BBOmix benchmark (105,000 autoencodix training runs across 5 architectures, 2 datasets, and many downstream tasks). No optimizer runs at call time, and `syne-tune` is **not** a runtime dependency of this package. This function is a plain JSON lookup + Pydantic validation.

The portfolio is not bundled with the package: it is hosted as the Hugging Face dataset [`autoencodix/initial-configs`](https://huggingface.co/datasets/autoencodix/initial-configs), downloaded on first use and cached locally afterwards. See section 7 for selecting a portfolio version.

## 1) Quick start[¶](#1-quick-start)

In \[1\]:

Copied!

```
import autoencodix as acx

config = acx.propose_initial_config("varix")
print(type(config).__name__)
print("epochs:", config.epochs, "| checkpoint_interval:", config.checkpoint_interval)
print("latent_dim:", config.latent_dim, "| beta:", config.beta)
```

import autoencodix as acx config = acx.propose_initial_config("varix") print(type(config).__name__) print("epochs:", config.epochs, "| checkpoint_interval:", config.checkpoint_interval) print("latent_dim:", config.latent_dim, "| beta:", config.beta)

```
VarixConfig
epochs: 300 | checkpoint_interval: 300
latent_dim: 58 | beta: 0.1497123009757529
```

With no other arguments, this returns the top-ranked config from the **combined** portfolio (pooling all BBOmix source tasks across datasets) for the `"downstream"` objective (maximizing aggregate downstream classification performance, at the full training budget). That combination is the recommended default: it's the portfolio expected to generalize best to a dataset that wasn't part of the original benchmark, which is the situation most callers of this function will actually be in.

## 2) Objectives: `"downstream"` vs. `"reconstruction"`[¶](#2-objectives-downstream-vs-reconstruction)

There are exactly two objective *kinds*:

- `"downstream"` (the default) — maximizes an aggregate downstream classification score. This is only ever evaluated at the final epoch (there's no per-epoch downstream signal in the benchmark data), so it always recommends the full training budget.
- `"reconstruction"` — minimizes reconstruction loss, which *was* recorded every epoch during the sweep. This objective has a genuine fidelity axis: pass `budget_epochs` to get a config tuned for training that long, not for 300 epochs.

In \[2\]:

Copied!

```
config_downstream = acx.propose_initial_config("varix", objective="downstream")
config_recon_full = acx.propose_initial_config("varix", objective="reconstruction")
config_recon_short = acx.propose_initial_config(
    "varix", objective="reconstruction", budget_epochs=25
)

print("downstream epochs:                ", config_downstream.epochs)
print("reconstruction (full) epochs:      ", config_recon_full.epochs)
print("reconstruction (budget=25) epochs: ", config_recon_short.epochs,
      "| checkpoint_interval:", config_recon_short.checkpoint_interval)
```

config_downstream = acx.propose_initial_config("varix", objective="downstream") config_recon_full = acx.propose_initial_config("varix", objective="reconstruction") config_recon_short = acx.propose_initial_config( "varix", objective="reconstruction", budget_epochs=25 ) print("downstream epochs: ", config_downstream.epochs) print("reconstruction (full) epochs: ", config_recon_full.epochs) print("reconstruction (budget=25) epochs: ", config_recon_short.epochs, "| checkpoint_interval:", config_recon_short.checkpoint_interval)

```
downstream epochs:                 300
reconstruction (full) epochs:       300
reconstruction (budget=25) epochs:  25 | checkpoint_interval: 25
```

`budget_epochs` snaps to the nearest of a fixed grid (`10, 25, 50, 100, 150, 200, 300`) and warns when it does:

In \[3\]:

Copied!

```
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    config = acx.propose_initial_config("varix", objective="reconstruction", budget_epochs=40)
    print("snapped epochs:", config.epochs)
    print("warning:", caught[0].message)
```

import warnings with warnings.catch_warnings(record=True) as caught: warnings.simplefilter("always") config = acx.propose_initial_config("varix", objective="reconstruction", budget_epochs=40) print("snapped epochs:", config.epochs) print("warning:", caught[0].message)

```
snapped epochs: 50
warning: budget_epochs=40 is not one of the available budgets [10, 25, 50, 100, 150, 200, 300]; snapped to 50.
```

Combining `objective="downstream"` with `budget_epochs` raises a `ValueError` — there's nothing for it to act on.

## 3) Scoping to a benchmark dataset (secondary, optional)[¶](#3-scoping-to-a-benchmark-dataset-secondary-optional)

`dataset=None` (the default, used above) is the recommended path for a new dataset. If you specifically want the portfolio derived from just one of the BBOmix benchmark's own datasets, pass its name:

In \[4\]:

Copied!

```
config_tcga = acx.propose_initial_config("varix", dataset="tcga")
config_schc = acx.propose_initial_config("varix", dataset="schc")
print(config_tcga.epochs, config_schc.epochs)
```

config_tcga = acx.propose_initial_config("varix", dataset="tcga") config_schc = acx.propose_initial_config("varix", dataset="schc") print(config_tcga.epochs, config_schc.epochs)

```
300 300
```

`dataset` is validated dynamically against whatever's actually in the loaded portfolio — not a fixed set baked into the function signature. An unknown name raises a `ValueError` that tells you what is available:

In \[5\]:

Copied!

```
try:
    acx.propose_initial_config("varix", dataset="my_new_cohort")
except ValueError as e:
    print(e)
```

try: acx.propose_initial_config("varix", dataset="my_new_cohort") except ValueError as e: print(e)

```
Unknown dataset 'my_new_cohort' for architecture 'varix'. Available datasets: ['schc', 'tcga']. Omit `dataset` (or pass None) to use the combined, cross-dataset portfolio -- the recommended default for a new dataset that wasn't part of the BBOmix benchmark.
```

## 4) Getting several ranked candidates[¶](#4-getting-several-ranked-candidates)

In \[6\]:

Copied!

```
candidates = acx.propose_initial_config("varix", top_k=3)
for i, c in enumerate(candidates, 1):
    print(i, "latent_dim:", c.latent_dim, "| n_layers:", c.n_layers, "| lr:", c.learning_rate)
```

candidates = acx.propose_initial_config("varix", top_k=3) for i, c in enumerate(candidates, 1): print(i, "latent_dim:", c.latent_dim, "| n_layers:", c.n_layers, "| lr:", c.learning_rate)

```
1 latent_dim: 58 | n_layers: 3 | lr: 0.0033023196273975707
2 latent_dim: 15 | n_layers: 4 | lr: 8.629522298260548e-05
3 latent_dim: 57 | n_layers: 3 | lr: 0.0003654758769476195
```

## 5) Overriding fields[¶](#5-overriding-fields)

Any keyword you pass through `**overrides` is applied on top of the proposed hyperparameters before the Config object is constructed — so you can take the proposal as a base and tweak just what you care about. Pydantic still validates the result, so an out-of-range override raises the normal `ValidationError`.

In \[7\]:

Copied!

```
from pydantic import ValidationError

config = acx.propose_initial_config("varix", latent_dim=8, learning_rate=1e-3)
print(config.latent_dim, config.learning_rate)

try:
    acx.propose_initial_config("varix", latent_dim=-1)
except ValidationError as e:
    print("rejected:", e.errors()[0]["msg"])
```

from pydantic import ValidationError config = acx.propose_initial_config("varix", latent_dim=8, learning_rate=1e-3) print(config.latent_dim, config.learning_rate) try: acx.propose_initial_config("varix", latent_dim=-1) except ValidationError as e: print("rejected:", e.errors()[0]["msg"])

```
8 0.001
rejected: Input should be greater than or equal to 1
```

## 6) Architectures without a BBOmix portfolio[¶](#6-architectures-without-a-bbomix-portfolio)

Only `vanillix`, `varix`, `ontix`, and `disentanglix` have a BBOmix-derived portfolio. Other architectures with a Config class (e.g. `stackix`, `xmodalix`, `maskix`) raise a `ValueError` by default; pass `allow_fallback_to_defaults=True` to instead get that class's plain schema defaults with a warning.

In \[8\]:

Copied!

```
try:
    acx.propose_initial_config("stackix")
except ValueError as e:
    print(e)

config = acx.propose_initial_config("stackix", allow_fallback_to_defaults=True)
print(type(config).__name__, "epochs:", config.epochs)
```

try: acx.propose_initial_config("stackix") except ValueError as e: print(e) config = acx.propose_initial_config("stackix", allow_fallback_to_defaults=True) print(type(config).__name__, "epochs:", config.epochs)

```
Unsupported architecture 'stackix'. propose_initial_config has a BBOmix-derived portfolio for: ['disentanglix', 'ontix', 'vanillix', 'varix']. Pass allow_fallback_to_defaults=True to fall back to plain schema defaults for any other architecture with a Config class.
StackixConfig epochs: 3
```

```
/var/folders/fz/3nckfp3n60s2f4p0_rb0vbph0000gn/T/ipykernel_20732/2371638015.py:6: UserWarning: No BBOmix-derived portfolio for architecture 'stackix'; falling back to StackixConfig schema defaults. Supported architectures with a portfolio: ['disentanglix', 'ontix', 'vanillix', 'varix'].
  config = acx.propose_initial_config("stackix", allow_fallback_to_defaults=True)
```

## 7) Fixing some hyperparameters[¶](#7-fixing-some-hyperparameters)

Sometimes a few hyperparameters are already decided, e.g. a `latent_dim` that downstream analysis needs, or a `batch_size` that fits your GPU. Pass them as `fixed` and the **remaining** hyperparameters are proposed *given* those values.

This is different from overrides (section 5): `latent_dim=16` as an override only overwrites that one field of the static proposal. The other values stay the ones that were tuned together with a different `latent_dim`. With `fixed`, a random-forest surrogate fitted on the raw BBOmix evaluations picks the free hyperparameters that work best with your fixed values.

In \[9\]:

Copied!

```
fixed_small = acx.propose_initial_config("varix", fixed={"latent_dim": 2})
fixed_large = acx.propose_initial_config("varix", fixed={"latent_dim": 64})
overridden = acx.propose_initial_config("varix", latent_dim=2)

for name, c in [("fixed latent_dim=2", fixed_small), ("fixed latent_dim=64", fixed_large), ("override latent_dim=2", overridden)]:
    print(f"{name:22s} lr={c.learning_rate:.2e} | n_layers={c.n_layers} | batch_size={c.batch_size} | beta={c.beta:.2e}")
```

fixed_small = acx.propose_initial_config("varix", fixed={"latent_dim": 2}) fixed_large = acx.propose_initial_config("varix", fixed={"latent_dim": 64}) overridden = acx.propose_initial_config("varix", latent_dim=2) for name, c in \[("fixed latent_dim=2", fixed_small), ("fixed latent_dim=64", fixed_large), ("override latent_dim=2", overridden)\]: print(f"{name:22s} lr={c.learning_rate:.2e} | n_layers={c.n_layers} | batch_size={c.batch_size} | beta={c.beta:.2e}")

```
fixed latent_dim=2     lr=3.70e-03 | n_layers=2 | batch_size=32 | beta=7.62e-04
fixed latent_dim=64    lr=2.51e-04 | n_layers=2 | batch_size=64 | beta=6.93e-04
override latent_dim=2  lr=3.30e-03 | n_layers=3 | batch_size=159 | beta=1.50e-01
```

`fixed` works together with `objective`, `budget_epochs`, `dataset`, `top_k` and overrides. Only hyperparameters that were tuned in the benchmark can be fixed (for anything else use overrides), and values outside the benchmarked range trigger a warning, because the proposal is then extrapolated. The first call per architecture fits the surrogate (well under a second); later calls reuse it.

In \[10\]:

Copied!

```
configs = acx.propose_initial_config(
    "varix",
    fixed={"latent_dim": 16, "batch_size": 64},
    objective="reconstruction",
    budget_epochs=50,
    top_k=3,
)
for c in configs:
    print(c.latent_dim, c.batch_size, c.epochs, "| lr:", f"{c.learning_rate:.2e}", "| n_layers:", c.n_layers)
```

configs = acx.propose_initial_config( "varix", fixed={"latent_dim": 16, "batch_size": 64}, objective="reconstruction", budget_epochs=50, top_k=3, ) for c in configs: print(c.latent_dim, c.batch_size, c.epochs, "| lr:", f"{c.learning_rate:.2e}", "| n_layers:", c.n_layers)

```
16 64 50 | lr: 1.75e-03 | n_layers: 2
16 64 50 | lr: 2.43e-03 | n_layers: 2
16 64 50 | lr: 1.34e-03 | n_layers: 3
```

## 8) Portfolio versions and offline use[¶](#8-portfolio-versions-and-offline-use)

By default the newest portfolio (the `main` revision on the Hub) is used, so proposals can change when we publish an improved portfolio. Every published portfolio is also tagged (`v1`, `v2`, ...); pin one with `revision` for reproducible results. `revision` also accepts a branch or commit hash.

In \[11\]:

Copied!

```
print(acx.tuning.list_portfolio_versions())

config_v1 = acx.propose_initial_config("varix", revision="v1")
print(config_v1.latent_dim)
```

print(acx.tuning.list_portfolio_versions()) config_v1 = acx.propose_initial_config("varix", revision="v1") print(config_v1.latent_dim)

```
['v1', 'v2']
```

```
58
```

To use your own portfolio (same JSON layout), either host it in your own Hugging Face dataset and pass `repo_id="<user>/<repo>"`, or skip the Hub entirely with `artifact_path="path/to/initial_configs.json"`.

After the first download the file is cached by `huggingface_hub`, so later calls also work offline, e.g. with `HF_HUB_OFFLINE=1`. If the download fails, a `RuntimeError` explains these options.

## What's next[¶](#whats-next)

This function only ever proposes an *initial* config, so it has no way to learn from your own training runs. To optimize your model further you can use actual HPO methods. These steps are explained for Syne Tune in `HyperparameterOptimizationTutorial.ipynb` and Optuna in `HyperparameterOptimizationOptunaTutorial.ipynb`.

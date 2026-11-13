# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. For what the package does, see [README.md](README.md). Rendered docs: https://jan-forest.github.io/autoencodix_package/ (agent-friendly text: [`llms.txt`](https://jan-forest.github.io/autoencodix_package/llms.txt), [`llms-full.txt`](https://jan-forest.github.io/autoencodix_package/llms-full.txt)).

## Setup and commands

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv sync                                   # installs package (editable) + dev group

uv run pytest -m "not slow and not network" tests/   # fast offline tests: run before every change
uv run pytest tests/test_varix.py                    # single module
uv run pytest tests/                                 # full suite (needs Hugging Face access)
black src/ tests/                                    # formatting is enforced in CI
ruff check src/                                      # linting
```

- Python `>=3.9,<3.13`. CI tests 3.9–3.12 on Linux, macOS and Windows: no 3.10+ only syntax (`match`, `X | Y` type unions at runtime).
- Test markers: `slow` (builds a wheel), `network` (downloads from the Hugging Face Hub).
- Docs build: see [Documentation](#documentation) below.

## Using the package: discover parameters, don't guess them

All behaviour is controlled by one validated Pydantic config per pipeline (`VanillixConfig`, `VarixConfig`, `OntixConfig`, ... in `autoencodix.configs`, all subclasses of `DefaultConfig`). Before writing or changing config code, list the real parameter names, types, defaults and descriptions:

```python
from autoencodix.configs.varix_config import VarixConfig

VarixConfig.print_schema()                                  # all parameters
VarixConfig.print_schema(filter_params=["beta", "epochs"])  # selected ones
VarixConfig.get_params()                                    # same as a dict
```

- For a tuned starting point instead of defaults: `acx.propose_initial_config("varix")`.
- Every pipeline has the same API: `acx.<Name>(data=DataPackage(...), config=<Name>Config(...))`, then `.run()` or the individual steps `.preprocess() .fit() .predict() .visualize() .generate() .evaluate() .explain() .save()/.load()`.
- Results: `pipeline.result` (`Result` object), e.g. `result.get_latent_df(split="test", epoch=-1)`.
- Runnable, current examples are the notebooks in `Tutorials/` (start with `Tutorials/PipelineTutorials/Vanillix.ipynb`). Notebooks contain large base64 image outputs: search them for the code cells you need instead of reading them whole.

## Code map (`src/autoencodix/`)

| Path | Contents |
|---|---|
| `<name>.py` (e.g. `varix.py`) | Public pipeline classes, subclassing `base/_base_pipeline.py::BasePipeline` |
| `base/` | Abstract base classes: `BasePipeline`, `BaseAutoencoder`, `BaseTrainer`, `BaseLoss`, `BaseDataset`, `BasePreprocessor`, `BaseEvaluator`, `BaseVisualizer` |
| `configs/` | Pydantic configs; `default_config.py` holds `DefaultConfig`, `DataCase`, `DataConfig` |
| `modeling/` | Network architectures (`_<name>_architecture.py`) |
| `losses/` | Loss classes, re-exported via `utils/_losses.py` |
| `trainers/` | `GeneralTrainer` plus architecture-specific trainers |
| `data/` | `DataPackage`, datasets, preprocessing, filtering, splitting |
| `evaluate/`, `visualize/` | `.evaluate()` and `.visualize()` implementations |
| `utils/` | Readers, `Result`, feature attribution (`_explainer.py`), LLM explanation (`_llm_explainer.py`) |
| `tuning/` | `propose_initial_config` (portfolio downloaded from HF dataset `autoencodix/initial-configs`) |

A pipeline wires together types, not instances: `trainer_type`, `model_type`, `loss_type`, ... are passed as classes and instantiated inside `BasePipeline`.

## Adding a new architecture

Follow `Tutorials/DevGuide.ipynb`. Checklist:

1. Architecture in `modeling/`, subclassing `BaseAutoencoder`; loss in `losses/` (subclassing `BaseLoss`, export it in `utils/_losses.py`); custom trainer in `trainers/` only if needed.
2. Config `configs/<name>_config.py` subclassing `DefaultConfig`; every new field gets a `description=` (it feeds `print_schema()`).
3. Pipeline `<name>.py` subclassing `BasePipeline`; export it in `src/autoencodix/__init__.py` (`import` + `__all__`).
4. Tests in `tests/`.
5. Docs: `docs/api/<name>.md` stub (copy an existing one), nav entries for API page and tutorial in `mkdocs.yml`, tutorial notebook in `Tutorials/PipelineTutorials/`.
6. README: add a row to the architecture table and a short section with its reference.

## Conventions

- Google-style docstrings with type hints on all public functions and classes (rendered by mkdocstrings).
- Keep the pipeline API identical across architectures; put architecture specifics in configs, trainers and losses.
- Private modules use a leading underscore; public names go through `__init__.py`.
- Tutorial notebooks are committed with outputs (the docs show them as saved and do not re-execute them).

## Documentation

- Built by `.github/workflows/docs.yml` on push to `main` and deployed to the `gh-pages` branch. Never edit `gh-pages` by hand.
- `docs/index.md` is overwritten in CI with a copy of `README.md` (GitHub tutorial links are rewritten to docs-site links). Edit `README.md`, not `docs/index.md`.
- README links must be absolute (`https://github.com/jan-forest/autoencodix_package/blob/main/...`), because the README is also rendered on PyPI and Hugging Face.
- Tutorial notebooks are copied into `docs/tutorials/` during CI; they only need a nav entry in `mkdocs.yml`.

## Do not

- Commit secrets: API keys for the `.explain()` LLM clients live in `.env` / `.scadsai_api_key` and must never be staged. Stage files by name rather than `git add -A`.
- Commit data, models or results (`data/`, `results/`, `*.pkl`, `*.pth`, `*.h5ad` are gitignored).
- Change public signatures or config field names without updating tutorials, README and tests.

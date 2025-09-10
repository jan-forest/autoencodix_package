# DEVELOPMENT
## Requirements
- Python>=3.8 <3.13
- uv (or another package manager, we recommend uv)
- git or gh
## Installation
- `gh repo clone MaxJoas/autoencodix_package`
- `uv venv --python 3.10`
- `source .venv/bin/activate`
- `uv pip install -e '.[dev]'`

## Style guidlines
- We use black formatting style
  - `uv run black .`
- We use typed python
  - we recommend mypy to enforce/lint typing
  - you can run `uv run mypy --install-types --non-interactive --package autoencodix`
- We want to pass arguments named, whenever possilbe
  - example bad: `result = trainer.train(10, arr)`
  - example good: `result = trainer.train(epochs=10, data=array)`
- Style is enforced with our GitHub Actions workflow for pull requests and pushes to the main branch
- We try to provide a docstring for every class and method/function. 
## Coding principles
- We try to adhere to SOLID, for more info see [here](https://realpython.com/solid-principles-python/). Howver if it makes our code too complicated, we violate the principles when necessary.
- We use pytest to write tests, a good guide is [here](https://pytest-with-eric.com)
- We aim for a 90% test coverage, but this is no strictly enforced
  - get testcoverage with: `uv run pytest --cov=src --cov-report=term-missing --cov-report=html`
- We run automatic tests with our GitHub Actions workflow after pushes to main or pull requests

## Notebooks
- We provide a `AUTOENCODIX PACKAGE HANDBOOK` in `notebooks/0.0-mj-sandbox.ipynb`. This should help greatly with development and understanding the code. Please create a notebook in a similar fashion for your contributions

## General contribution workflow.
- we mainly work with pull requests, but feature branches are also ok in some cases.
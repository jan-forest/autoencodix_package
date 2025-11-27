# Contributing to Autoencodix Package
We're happy about all kinds of contributions and feedback to Autoencodix. For feedback, you can use the [issue board](https://github.com/jan-forest/autoencodix/issues) or send us an [mail](mailto:jan.ewald@uni-leipzig.de). If you want to contribute with code, please open a pull request to either work on your own idea or tackle an exisiting [issue](https://github.com/jan-forest/autoencodix/issues) is a good starting point. If you are unsure or confused about something, please just open an [issue](https://github.com/jan-forest/autoencodix/issues/new/choose), and we will try to address it.
We also value non-code contributions in the form of a feature request or bug report.

## Feature Request

Have a great idea for a new feature in Autoencodix? Let us know by submitting a feature request!
We are in particular interested in expanding the number of architectures and related options. Hence, we value your input on what might be missing in order to enable your application and research on autoencoders with our framework.

If you'd like to take the initiative, feel free to [fork Autoencodix](https://github.com/jan-forest/autoencodix_package/fork), work on implementing the feature, and submit a patch. Before starting, it is best to [open an issue](https://github.com/jan-forest/autoencodix_package/issues/new/choose) first to describe your enhancement. This helps you to get early feedback and ensures that your work can be effectively incorporated into the Autoencodix codebase, saving time and aligning with the project’s direction.

## Bug Report
Even the smartest Gaul can make mistakes, so if you encounter a bug when using Autoencodix, we highly appreciate a bug report. 
When submitting a [bug report](https://github.com/jan-forest/autoencodix_package/issues/new/choose), please provide as much context as possible. In particular, always attach the related YAML config causing the issue. Further, depending on the issue, try to include details like the Python version, Autoencodix version, any error messages or stack traces, and clear steps to reproduce the issue (if possible, include a small sample dataset that triggers the error). The more information provided, the easier it will be to address the issue effectively

## Pull Request
Interested in contributing to Autoencodix? Fantastic!

If there are [open issues](https://github.com/jan-forest/autoencodix_package/issues), feel free to start with those (especially ones labeled "good first issue"). Have your own ideas? That’s welcome too! Before diving into significant changes, it’s recommended to open an issue describing what you plan to work on to get feedback early.

To contribute, fork the repository, push your changes to your fork, and submit a [Pull Request](https://github.com/jan-forest/autoencodix_package/compare).


Detailed guidelines, including instructions on running tests, will be provided soon. These will likely be outlined in a DEVELOPMENT.md file.





## DEVELOPMENT
If you are a developer and you want to contribute code, please refer to the following guide:
## Requirements
- Python>=3.8 <3.13
- uv (or another package manager, we recommend uv)
- git or gh
## Installation
- `gh repo clone jan-forest/autoencodix_package`
- `uv venv --python 3.10`
- `source .venv/bin/activate`
- `uv pip install -e '.[dev]'`

## Style guidlines
- We use black formatting style
  - `uv run black .`
- We use typed Python
  - We recommend `ty` to enforce/lint typing
- We want to pass arguments named, whenever possible
  - example bad: `result = trainer.train(10, arr)`
  - example good: `result = trainer.train(epochs=10, data=array)`
- Style is enforced with our GitHub Actions workflow for pull requests and pushes to the main branch
- We try to provide a docstring for every class and method/function. 
## Coding principles
- We try to adhere to SOLID, for more info see [here](https://realpython.com/solid-principles-python/). However, if it makes our code too complicated, we violate the principles when necessary.
- We use pytest to write tests, a good guide is [here](https://pytest-with-eric.com)
- We aim for a 90% test coverage, but this is not strictly enforced
  - get test coverage with: `uv run pytest --cov=src --cov-report=term-missing --cov-report=html`
- We run automatic tests with our GitHub Actions workflow after pushes to main or pull requests

## Notebooks
- We provide a detailed [guide](https://github.com/jan-forest/autoencodix_package/main/Tutorials/DevGuide.ipynb) on how to add a new architecture to our framework. Additionally, we have tutorial notebooks for every pipeline (Varix, Ontix, etc), see [here](https://github.com/jan-forest/autoencodix_package/tree/main/Tutorials). If you implement a new pipeline, it would be great, if you provided a similar notebook, which will greatly foster adoption of your pipeline.
## General contribution workflow.
- We mainly work with pull requests, but feature branches are also ok in some cases.

  
---
Still questions? Reach out to [us](mailto:jan.ewald@uni-leipzig.de).

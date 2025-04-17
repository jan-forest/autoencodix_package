# autoencodix

[![PyPI - Version](https://img.shields.io/pypi/v/autoencodix.svg)](https://pypi.org/project/autoencodix)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autoencodix.svg)](https://pypi.org/project/autoencodix)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Requirements
- Python>=3.8 <3.13
- uv (or another package manager, we recommend uv)
- git or gh
## Installation
- `gh repo clone MaxJoas/autoencodix_package`
- `uv venv --python 3.10`
- `source .venv/bin/activate`
- `uv pip install -e .'` 

## Sample Usage
```python
import numpy as np
import autoencodix as acx
from autoencodix.utils.default_config import DefaultConfig
sample_data = np.random.rand(100, 10)
sample_data.shape
van = acx.Vanillix(data=sample_data)
result_object = van.run()
```
## Contributing
- for more infos please read our `DEVELOPMENT.md`
- for more details on usage and how to contribute notebook `notebooks/0.0-mj-sandbox.ipynb` might be helpful.

## Structure
```
src
|-- autoencodix
|   |-- base
|   |   |-- __init__.py
|   |   |-- _base_autoencoder.py
|   |   |-- _base_dataset.py
|   |   |-- _base_evaluator.py
|   |   |-- _base_pipeline.py
|   |   |-- _base_predictor.py
|   |   |-- _base_preprocessor.py
|   |   |-- _base_trainer.py
|   |   |-- _base_visualizer.py
|   |-- data
|   |   |-- __init__.py
|   |   |-- _datasetcontainer.py
|   |   |-- _datasplitter.py
|   |   |-- _numeric_dataset.py
|   |   `-- preprocessor.py
|   |-- evaluate
|   |   |-- __init__.py
|   |   `-- evaluate.py
|   |-- modeling
|   |   |-- __init__.py
|   |   |-- _layer_factory.py
|   |   `-- _vanillix_architecture.py
|   |-- trainers
|   |   |-- __init__.py
|   |   |-- _vanillix_trainer.py
|   |   `-- predictor.py
|   |-- utils
|   |   |-- __init__.py
|   |   |-- _model_output.py
|   |   |-- _result.py
|   |   |-- _traindynamics.py
|   |   |-- _utils.py
|   |   `-- default_config.py
|   |-- visualize
|   |   |-- __init__.py
|   |   `-- visualize.py
|   |-- __init__.py
|   |-- py.typed
|   `-- vanillix.py
`-- __init__.py
tests
|-- test_base
|   |-- test_base_autoencoder.py
|   |-- test_base_dataset.py
|   |-- test_base_pipeline.py
|   `-- test_base_trainer.py
|-- test_data
|   |-- test_datasetcontainer.py
|   |-- test_datasplitter.py
|   `-- test_numericdataset.py
|-- test_evaluate
|-- test_modelling
|   |-- test_layer_factory.py
|   `-- test_vanillix_architecture.py
|-- test_trainers
|   `-- test_vanillix_trainer.py
`-- test_utils
    |-- test_default_config.py
    |-- test_modeloutput.py
    |-- test_result.py
    |-- test_traindynamics.py
    `-- test_utils.py


```
## License

Copyright [2024] [Maximilian Josef Joas & Jan Ewald, ScaDS.AI, Leipzig University]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

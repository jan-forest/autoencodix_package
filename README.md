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
- uv venv --python 3.10
- source .venv/bin/activate
- `uv pip install -e '.[dev]'`

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

## License

Copyright [2024] [Maximilian Josef Joas & Jan Ewald, ScaDS.AI, Leipzig University]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
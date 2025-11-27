# AUTOENCODIX
Autoencoders are deep-learning-based networks for dimension reduction and embedding by a combination of a compressing encoder and decoder structure for non-linear and multi-modal data integration, with promising applications to complex biological data from large-scale omics measurements. Current ongoing research and publications provide many exciting architectures and implementations of autoencoders. However, there is a lack of easy-to-use and unified implementations covering the whole pipeline of autoencoder applications.
Consequently, we present `AUTOENCODIX` with the following features:
- Multi-modal data integration for any numerical or categorical data
- Different autoencoder architectures:
  - vanilla `vanillix`
  - variational `varix`
  - hierarchical/stacked `stackix`
  - ontology-based `ontix`
  - masking `maskix`
  - cross-modal autoencoder (translation between different data modalities) `x-modalix` (works for multiple modalities paired and unpaired)
- A Python package with a scikit-learn-like interface

## Requirements
- Python>=3.8 <3.13
- uv or another package manager (we recommend uv)
- git or gh
## Installation
- `gh repo clone jan-forest/autoencodix_package`
- `cd autoencodix_package`
- `uv venv --python 3.10`
- `source .venv/bin/activate`
- `uv sync`

## Sample Usage
```python
import autoencodix as acx
from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.configs.default_config import DataCase

# If your data is stored in pandas DataFrames, you can easily pass them to our custom DataPackage.
# For any tabular data that is not single-cell, provide it as a dictionary to the "multi_bulk" attribute of DataPackage.
# Note: "multi" might be misleading — it's valid to provide just one modality (1–n data modalities).
# Here, we assume paired metadata. If you have separate metadata for each modality, use the same dict keys as in multi_bulk, e.g.:
# annotation = {"rna": rna_annotation, "protein": protein_annotation}
my_datapackage: DataPackage = DataPackage(
    multi_bulk={"rna": raw_rna, "protein": raw_protein},
    annotation={"paired": annotation},
)

myconfig: VanillixConfig = VanillixConfig(data_case=DataCase.MULTI_BULK, epochs=30, device="cpu")
vanillix = acx.Vanillix(data=my_datapackage, config=myconfig)
result = vanillix.run()
```

## Getting Started
We provide extensive tutorials for all of our use cases. The best place to start is the `Vanillix Tutorial`. Here, we explain the design and features of our pipeline, which applies to other pipelines. From there, you can explore the tutorials for the more specialized architectures (`Varix`, `Ontix`, etc). We also provide tutorials for each pipeline, but the `Vanillix Tutorial` explains the general concepts, while the other tutorials go into the specifics of the corresponding pipeline. For even more details on extra functionality, such as visualizing or customizing we provide deep dive tutorials for these topics. You can find the tutorials here:

- Best to get started: [Vanillix Tutorial](https://github.com/jan-forest/autoencodix_package/blob/main/Tutorials/PipelineTutorials/Vanillix.ipynb)
- Tutorials for each specific pipeline (more advanced, do Vanillix first): [Pipeline Tutorials](https://github.com/jan-forest/autoencodix_package/tree/main/Tutorials/PipelineTutorials)
- Tutorials for specific extra functionality, see [Deep Dives](https://github.com/jan-forest/autoencodix_package/tree/main/Tutorials/DeepDives)
## Contributing
Whether you have a feature request, found a bug, or have any other idea, we're always happy. For more details, refer to our [Guide](https://github.com/jan-forest/autoencodix_package/blob/main/DEVELOPER.md)

## Read The Docs
You can find our documentation [here](https://jan-forest.github.io/autoencodix_package/).

## Cite
TODO

## License

Copyright [2024] [Maximilian Josef Joas & Jan Ewald, ScaDS.AI, Leipzig University]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

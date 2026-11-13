> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[1\]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

```
env: CUBLAS_WORKSPACE_CONFIG=:16:8
```

# How To Use Imagix[¶](#how-to-use-imagix)

Imagix is our implementation of a variational autoencoder for image data.\
This tutorial follows the structure of [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Imagix/Vanillix.ipynb), but is much less extensive because our pipeline works similarly across different architectures. Here we focus only on Imagix specifics.

**AUTOENCODIX** supports far more functionality than shown here, so we’ll also point to advanced tutorials where relevant.

**IMPORTANT**

> This tutorial only shows the specifics of the Imagix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Imagix/Vanillix.ipynb) Tutorial first.

## What You'll Learn[¶](#what-youll-learn)

You’ll learn how to:

1. **Initialize** the pipeline and run it.
1. Understand the Imagix specific **pipeline steps**.
1. Access the Imagix specific **results** (mus, sigma, kl/mmd losses).
1. **Visualize** outputs effectively.
1. Apply **custom parameters**. or Architecture
1. **Save, load, and reuse** a trained pipeline.

**Setting the Correct Path**

In \[2\]:

Copied!

```
import os

p = os.getcwd()
d = "autoencodix_package"
if d not in p:
    raise FileNotFoundError(f"'{d}' not found in path: {p}")
os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1]))
print(f"Changed to: {os.getcwd()}")
```

import os p = os.getcwd() d = "autoencodix_package" if d not in p: raise FileNotFoundError(f"'{d}' not found in path: {p}") os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1])) print(f"Changed to: {os.getcwd()}")

```
Changed to: /home/alicia/dev/biomarker_autoencoder/autoencodix_package
```

## 1) Initialize the Imagix Pipeline[¶](#1-initialize-the-imagix-pipeline)

`Imagix` is a standard VAE implementation for image data. We don't allow different data modalities for `Imagix`.\
To run the pipeline we need to prepare two things:

1. A directory with image files. We allow the following extensions: `".jpg", ".jpeg", ".png", ".tif", ".tiff"` (NOT case sensitive).
1. An annotation file with metadata, where we map `sample_ids` to image paths. For this, we need to provide the name of the column where the image path information is stored. This is done via the `img_paths_col` config parameter.

### ❗❗Requirements ❗❗[¶](#requirements)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/mnist](https://huggingface.co/datasets/autoencodix/mnist)) and is downloaded (and extracted) automatically in the cell below on first run.

#### 1.1 The Dataset[¶](#11-the-dataset)

Here we use a balanced subsample of the classic MNIST handwritten digits dataset: 2000 images, 200 per digit (0-9), 28x28 grayscale. The annotation column `label` gives the digit class for each image.

**A Look Inside the Annotation File**:

In \[3\]:

Copied!

```
import os
import zipfile
import pandas as pd
import autoencodix as acx
from huggingface_hub import hf_hub_download
from autoencodix.configs.default_config import (
    DefaultConfig,
    DataConfig,
    DataCase,
    DataInfo,
)

IMGMAPPING = hf_hub_download(repo_id="autoencodix/mnist", repo_type="dataset", filename="mnist_mappings.txt")
images_zip = hf_hub_download(repo_id="autoencodix/mnist", repo_type="dataset", filename="mnist_images.zip")

IMGROOT = os.path.join("data/images/mnist_images/")
if not os.path.isdir(IMGROOT):
    with zipfile.ZipFile(images_zip) as zf:
        zf.extractall("data/images/")

anno_df = pd.read_csv(IMGMAPPING, sep="\t", index_col=0)
anno_df.head()
```

import os import zipfile import pandas as pd import autoencodix as acx from huggingface_hub import hf_hub_download from autoencodix.configs.default_config import ( DefaultConfig, DataConfig, DataCase, DataInfo, ) IMGMAPPING = hf_hub_download(repo_id="autoencodix/mnist", repo_type="dataset", filename="mnist_mappings.txt") images_zip = hf_hub_download(repo_id="autoencodix/mnist", repo_type="dataset", filename="mnist_images.zip") IMGROOT = os.path.join("data/images/mnist_images/") if not os.path.isdir(IMGROOT): with zipfile.ZipFile(images_zip) as zf: zf.extractall("data/images/") anno_df = pd.read_csv(IMGMAPPING, sep="\\t", index_col=0) anno_df.head()

Out\[3\]:

|              | img_paths        | label |
| ------------ | ---------------- | ----- |
| sample_ids   |                  |       |
| mnist_0_0000 | mnist_0_0000.png | '0'   |
| mnist_0_0001 | mnist_0_0001.png | '0'   |
| mnist_0_0002 | mnist_0_0002.png | '0'   |
| mnist_0_0003 | mnist_0_0003.png | '0'   |
| mnist_0_0004 | mnist_0_0004.png | '0'   |

**Define Config and Run Pipeline**

In \[4\]:

Copied!

```
imgconfig = DefaultConfig(
    data_case=DataCase.IMG_TO_IMG,
    checkpoint_interval=25,
    epochs=250,
    reconstruction_loss="bce",
    beta=0.005,
    scaling="MINMAX",
    anneal_function="logistic-late",
    data_config=DataConfig(
        data_info={
            "IMG": DataInfo(
                file_path=IMGROOT,
                scaling="MINMAX",
                data_type="IMG",
            ),
            "ANNO": DataInfo(
                file_path=IMGMAPPING,
                data_type="ANNOTATION",
            ),
        },
    ),
)

imagix = acx.Imagix(config=imgconfig)
result = imagix.run()
backup_datset = result.datasets
```

imgconfig = DefaultConfig( data_case=DataCase.IMG_TO_IMG, checkpoint_interval=25, epochs=250, reconstruction_loss="bce", beta=0.005, scaling="MINMAX", anneal_function="logistic-late", data_config=DataConfig( data_info={ "IMG": DataInfo( file_path=IMGROOT, scaling="MINMAX", data_type="IMG", ), "ANNO": DataInfo( file_path=IMGMAPPING, data_type="ANNOTATION", ), }, ), ) imagix = acx.Imagix(config=imgconfig) result = imagix.run() backup_datset = result.datasets

```
Given image size is possible, rescaling images to: 64x64
Successfully loaded 2000 images for IMG
calling normalize image in _process_ing_to_img_case
anno key: IMG
Converting 1400 images to torch.float32 tensors...
Converting 400 images to torch.float32 tensors...
Converting 200 images to torch.float32 tensors...
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 15.0112 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 25 - Train Loss: 446.9194
Sub-losses: recon_loss: 446.9194, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 25 - Valid Loss: 502.1609
Sub-losses: recon_loss: 502.1609, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 50 - Train Loss: 393.6598
Sub-losses: recon_loss: 393.6594, var_loss: 0.0004, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 50 - Valid Loss: 494.4483
Sub-losses: recon_loss: 494.4479, var_loss: 0.0004, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 75 - Train Loss: 368.1934
Sub-losses: recon_loss: 368.1889, var_loss: 0.0045, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 75 - Valid Loss: 502.7059
Sub-losses: recon_loss: 502.7012, var_loss: 0.0047, anneal_factor: 0.0001, effecti
... [output truncated]
```

## 2) Understand Imagix Specific Steps[¶](#2-understand-imagix-specific-steps)

Since `Imagix` is jus a `Varix` for images, there are no extra steps for this pipeline.

## 3) Access Imagix Specific Results[¶](#3-access-imagix-specific-results)

The `result` object follows our standard interface. Refer to [1] for more details.\
We don't have any `Imagix` specific results, but you can directly visualize the reconstructions as images, as shown in the code below:

[1] [Tutorials/DeepDives/PipelineOutputTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/PipelineOutputTutorial.ipynb)

In \[5\]:

Copied!

```
import matplotlib.pyplot as plt

sample_img = result.reconstructions.get(split="test", epoch=-1)
sample_img = sample_img[0, :, :, :]
sample_img = sample_img.squeeze()
sample_img.shape

plt.imshow(sample_img, cmap="grey")
```

import matplotlib.pyplot as plt sample_img = result.reconstructions.get(split="test", epoch=-1) sample_img = sample_img[0, :, :, :] sample_img = sample_img.squeeze() sample_img.shape plt.imshow(sample_img, cmap="grey")

Out\[5\]:

```
<matplotlib.image.AxesImage at 0x7fd9f62a78e0>
```

## 4) Visualize Imagix Results[¶](#4-visualize-imagix-results)

Since the results of `Imagix` are visually interpretable, we add an additional visualization:\
We show a grid of original images and reconstructed images with label information. See the code below for how to obtain this visualization.

In \[6\]:

Copied!

```
imagix.visualizer.show_image_recon_grid(result=imagix.result, n_samples=5)
```

imagix.visualizer.show_image_recon_grid(result=imagix.result, n_samples=5)

**Standard Visualizations**\
We also offer the standard visualizations as for all other pipelines. You can pass one or more column names from the annotation data. These columns will be used to color the visualizations accordingly. In this example, we use the `label` parameter.

In \[7\]:

Copied!

```
imagix.show_result(params=["label"])
```

imagix.show_result(params=["label"])

```
Creating plots ...
```

## 5) Customize Imagix[¶](#5-customize-imagix)

Via the `Config` object we can apply customize the pipeline by chaning the `loss function` or the number of `epoch` or preprocessing. To get an full overview of the adjustable parameters, please refere to [2].\
In Addition, there is a way to use another architecture for the image autoencoder. We currently offer two architectures:`ImageVAEArchitecture` and `ImageVAEFastArchitecture`. The fast architecture does not has BatchNorm and inplace activation functions, which can speed up the training process. The `ImageVAEArchitecture` is the default architecture and based on [Yang & Uhler](https://arxiv.org/abs/1902.03515).

We can simply pass the architecture as type to the `model_type` parameter of the `Imagix` pipeline.

[2] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb).

In \[8\]:

Copied!

```
from autoencodix.modeling._imgfast_architecture import ImageVAEFastArchitecture
from autoencodix.modeling._imagevae_architecture import ImageVAEArchitecture


imgconfig = DefaultConfig(
    data_case=DataCase.IMG_TO_IMG,
    checkpoint_interval=25,
    epochs=250,
    reconstruction_loss="bce",
    beta=0.005,
    scaling="MINMAX",
    anneal_function="logistic-late",
    data_config=DataConfig(
        data_info={
            "IMG": DataInfo(
                file_path=IMGROOT,
                scaling="MINMAX",
                data_type="IMG",
            ),
            "ANNO": DataInfo(
                file_path=IMGMAPPING,
                data_type="ANNOTATION",
            ),
        },
    ),
)

imagix = acx.Imagix(config=imgconfig, model_type=ImageVAEFastArchitecture)
result = imagix.run()
backup_datset = result.datasets
```

from autoencodix.modeling.\_imgfast_architecture import ImageVAEFastArchitecture from autoencodix.modeling.\_imagevae_architecture import ImageVAEArchitecture imgconfig = DefaultConfig( data_case=DataCase.IMG_TO_IMG, checkpoint_interval=25, epochs=250, reconstruction_loss="bce", beta=0.005, scaling="MINMAX", anneal_function="logistic-late", data_config=DataConfig( data_info={ "IMG": DataInfo( file_path=IMGROOT, scaling="MINMAX", data_type="IMG", ), "ANNO": DataInfo( file_path=IMGMAPPING, data_type="ANNOTATION", ), }, ), ) imagix = acx.Imagix(config=imgconfig, model_type=ImageVAEFastArchitecture) result = imagix.run() backup_datset = result.datasets

```
Given image size is possible, rescaling images to: 64x64
Successfully loaded 2000 images for IMG
calling normalize image in _process_ing_to_img_case
anno key: IMG
Converting 1400 images to torch.float32 tensors...
Converting 400 images to torch.float32 tensors...
Converting 200 images to torch.float32 tensors...
Epoch 25 - Train Loss: 467.5948
Sub-losses: recon_loss: 467.5943, var_loss: 0.0005, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 25 - Valid Loss: 525.0574
Sub-losses: recon_loss: 525.0569, var_loss: 0.0005, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 50 - Train Loss: 402.6618
Sub-losses: recon_loss: 402.6540, var_loss: 0.0078, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 50 - Valid Loss: 524.4652
Sub-losses: recon_loss: 524.4579, var_loss
... [output truncated]
```

## 6) Load Save and Reuse Imagix[¶](#6-load-save-and-reuse-imagix)

There are not `Imagix` specific steps here. See the `Getting Started - Vanillix` for details. Below is a basic save/load usecase.

In \[9\]:

Copied!

```
import os
import glob

outpath = os.path.join("tutorial_res", "imagix.pkl")

imagix.save(file_path=outpath)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*imagix.pkl"))
model_files = glob.glob(os.path.join(folder, "*imagix.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
imagix_loaded = acx.Imagix.load(outpath)
```

import os import glob outpath = os.path.join("tutorial_res", "imagix.pkl") imagix.save(file_path=outpath) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*imagix.pkl")) model_files = glob.glob(os.path.join(folder, "\*imagix.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

imagix_loaded = acx.Imagix.load(outpath)

```
Preprocessor saved successfully.
saving memory efficient
Pipeline object saved successfully.
PKL files: ['tutorial_res/imagix.pkl']
Model files: []
Attempting to load a pipeline from tutorial_res/imagix.pkl...
Pipeline object loaded successfully. Actual type: Imagix
Preprocessor loaded successfully.
```

In \[10\]:

Copied!

```
result_predict = imagix_loaded.predict(data=backup_datset)
```

result_predict = imagix_loaded.predict(data=backup_datset)

```
Processed 400 / 400 samples
```

Visualize on the new predictions.

In \[11\]:

Copied!

```
imagix_loaded.show_result(params=["label"])
```

imagix_loaded.show_result(params=["label"])

```
Creating plots ...
Absolute loss plot not found in the plots dictionary
This happens, when you did not run visualize() or if you saved and loaded the model with `save_all=False`
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 249 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 249 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 249 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 249 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

In \[12\]:

Copied!

```
imagix_loaded.visualizer.show_image_recon_grid(result=imagix_loaded.result, n_samples=5)
```

imagix_loaded.visualizer.show_image_recon_grid(result=imagix_loaded.result, n_samples=5)

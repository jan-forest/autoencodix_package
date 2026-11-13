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
Changed to: /home/janew/ACX/autoencodix_package
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
```

```
Successfully loaded 2000 images for IMG
calling normalize image in _process_ing_to_img_case
anno key: IMG
Converting 1400 images to torch.float32 tensors...
Converting 400 images to torch.float32 tensors...
Converting 200 images to torch.float32 tensors...
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 14.8192 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 25 - Train Loss: 438.8208
Sub-losses: recon_loss: 438.8208, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 25 - Valid Loss: 508.9986
Sub-losses: recon_loss: 508.9986, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
```

```
Epoch 50 - Train Loss: 386.8533
Sub-losses: recon_loss: 386.8529, var_loss: 0.0004, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 50 - Valid Loss: 500.0794
Sub-losses: recon_loss: 500.0790, var_loss: 0.0004, anneal_factor: 0.0000, effective_beta_factor: 0.0000
```

```
Epoch 75 - Train Loss: 366.8177
Sub-losses: recon_loss: 366.8130, var_loss: 0.0046, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 75 - Valid Loss: 520.9086
Sub-losses: recon_loss: 520.9041, var_loss: 0.0045, anneal_factor: 0.0001, effective_beta_factor: 0.0000
```

```
Epoch 100 - Train Loss: 352.3103
Sub-losses: recon_loss: 352.2635, var_loss: 0.0468, anneal_factor: 0.0008, effective_beta_factor: 0.0000
Epoch 100 - Valid Loss: 518.3987
Sub-losses: recon_loss: 518.3511, var_loss: 0.0476, anneal_factor: 0.0008, effective_beta_factor: 0.0000
```

```
Epoch 125 - Train Loss: 346.8793
Sub-losses: recon_loss: 346.4524, var_loss: 0.4269, anneal_factor: 0.0062, effective_beta_factor: 0.0000
Epoch 125 - Valid Loss: 531.1880
Sub-losses: recon_loss: 530.7356, var_loss: 0.4525, anneal_factor: 0.0062, effective_beta_factor: 0.0000
```

```
Epoch 150 - Train Loss: 345.2943
Sub-losses: recon_loss: 342.3082, var_loss: 2.9862, anneal_factor: 0.0439, effective_beta_factor: 0.0002
Epoch 150 - Valid Loss: 527.9324
Sub-losses: recon_loss: 524.8586, var_loss: 3.0738, anneal_factor: 0.0439, effective_beta_factor: 0.0002
```

```
Epoch 175 - Train Loss: 350.2575
Sub-losses: recon_loss: 342.3741, var_loss: 7.8835, anneal_factor: 0.2535, effective_beta_factor: 0.0013
Epoch 175 - Valid Loss: 549.7746
Sub-losses: recon_loss: 542.1984, var_loss: 7.5762, anneal_factor: 0.2535, effective_beta_factor: 0.0013
```

```
Epoch 200 - Train Loss: 355.2111
Sub-losses: recon_loss: 345.3760, var_loss: 9.8352, anneal_factor: 0.7150, effective_beta_factor: 0.0036
Epoch 200 - Valid Loss: 556.9980
Sub-losses: recon_loss: 547.4770, var_loss: 9.5209, anneal_factor: 0.7150, effective_beta_factor: 0.0036
```

```
Epoch 225 - Train Loss: 356.9684
Sub-losses: recon_loss: 346.8836, var_loss: 10.0849, anneal_factor: 0.9488, effective_beta_factor: 0.0047
Epoch 225 - Valid Loss: 575.4523
Sub-losses: recon_loss: 565.9695, var_loss: 9.4828, anneal_factor: 0.9488, effective_beta_factor: 0.0047
```

```
Epoch 250 - Train Loss: 354.4498
Sub-losses: recon_loss: 344.6672, var_loss: 9.7825, anneal_factor: 0.9928, effective_beta_factor: 0.0050
Epoch 250 - Valid Loss: 563.1205
Sub-losses: recon_loss: 553.5057, var_loss: 9.6148, anneal_factor: 0.9928, effective_beta_factor: 0.0050
Processed 32 / 400 samples
Processed 64 / 400 samples
Processed 96 / 400 samples
Processed 128 / 400 samples
Processed 160 / 400 samples
Processed 192 / 400 samples
Processed 224 / 400 samples
Processed 256 / 400 samples
Processed 288 / 400 samples
Processed 320 / 400 samples
Processed 352 / 400 samples
Processed 384 / 400 samples
Processed 400 / 400 samples
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
<matplotlib.image.AxesImage at 0x7b2cf738b230>
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
```

```
Successfully loaded 2000 images for IMG
calling normalize image in _process_ing_to_img_case
anno key: IMG
Converting 1400 images to torch.float32 tensors...
Converting 400 images to torch.float32 tensors...
Converting 200 images to torch.float32 tensors...
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 5.8905 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 25 - Train Loss: 467.6268
Sub-losses: recon_loss: 467.6263, var_loss: 0.0005, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 25 - Valid Loss: 539.3104
Sub-losses: recon_loss: 539.3099, var_loss: 0.0005, anneal_factor: 0.0000, effective_beta_factor: 0.0000
```

```
Epoch 50 - Train Loss: 403.6699
Sub-losses: recon_loss: 403.6620, var_loss: 0.0079, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 50 - Valid Loss: 555.0461
Sub-losses: recon_loss: 555.0381, var_loss: 0.0080, anneal_factor: 0.0000, effective_beta_factor: 0.0000
```

```
Epoch 75 - Train Loss: 372.1672
Sub-losses: recon_loss: 372.0882, var_loss: 0.0790, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 75 - Valid Loss: 575.3615
Sub-losses: recon_loss: 575.2818, var_loss: 0.0796, anneal_factor: 0.0001, effective_beta_factor: 0.0000
```

```
Epoch 100 - Train Loss: 357.7046
Sub-losses: recon_loss: 357.1626, var_loss: 0.5420, anneal_factor: 0.0008, effective_beta_factor: 0.0000
Epoch 100 - Valid Loss: 586.2407
Sub-losses: recon_loss: 585.7131, var_loss: 0.5276, anneal_factor: 0.0008, effective_beta_factor: 0.0000
```

```
Epoch 125 - Train Loss: 349.3904
Sub-losses: recon_loss: 347.6527, var_loss: 1.7377, anneal_factor: 0.0062, effective_beta_factor: 0.0000
Epoch 125 - Valid Loss: 591.0341
Sub-losses: recon_loss: 589.3687, var_loss: 1.6654, anneal_factor: 0.0062, effective_beta_factor: 0.0000
```

```
Epoch 150 - Train Loss: 351.3182
Sub-losses: recon_loss: 347.6052, var_loss: 3.7130, anneal_factor: 0.0439, effective_beta_factor: 0.0002
Epoch 150 - Valid Loss: 606.8701
Sub-losses: recon_loss: 603.1385, var_loss: 3.7315, anneal_factor: 0.0439, effective_beta_factor: 0.0002
```

```
Epoch 175 - Train Loss: 359.8177
Sub-losses: recon_loss: 352.1910, var_loss: 7.6266, anneal_factor: 0.2535, effective_beta_factor: 0.0013
Epoch 175 - Valid Loss: 617.9312
Sub-losses: recon_loss: 610.6943, var_loss: 7.2369, anneal_factor: 0.2535, effective_beta_factor: 0.0013
```

```
Epoch 200 - Train Loss: 367.0569
Sub-losses: recon_loss: 355.7136, var_loss: 11.3433, anneal_factor: 0.7150, effective_beta_factor: 0.0036
Epoch 200 - Valid Loss: 623.4192
Sub-losses: recon_loss: 612.4260, var_loss: 10.9932, anneal_factor: 0.7150, effective_beta_factor: 0.0036
```

```
Epoch 225 - Train Loss: 367.0073
Sub-losses: recon_loss: 355.2151, var_loss: 11.7922, anneal_factor: 0.9488, effective_beta_factor: 0.0047
Epoch 225 - Valid Loss: 641.4229
Sub-losses: recon_loss: 629.6204, var_loss: 11.8024, anneal_factor: 0.9488, effective_beta_factor: 0.0047
```

```
Epoch 250 - Train Loss: 363.2370
Sub-losses: recon_loss: 351.8541, var_loss: 11.3829, anneal_factor: 0.9928, effective_beta_factor: 0.0050
Epoch 250 - Valid Loss: 630.8686
Sub-losses: recon_loss: 619.5691, var_loss: 11.2994, anneal_factor: 0.9928, effective_beta_factor: 0.0050
Processed 32 / 400 samples
Processed 64 / 400 samples
Processed 96 / 400 samples
Processed 128 / 400 samples
Processed 160 / 400 samples
Processed 192 / 400 samples
Processed 224 / 400 samples
Processed 256 / 400 samples
Processed 288 / 400 samples
Processed 320 / 400 samples
Processed 352 / 400 samples
Processed 384 / 400 samples
Processed 400 / 400 samples
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
model_files = glob.glob(os.path.join(folder, "*imagix*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
imagix_loaded = acx.Imagix.load(outpath)
```

import os import glob outpath = os.path.join("tutorial_res", "imagix.pkl") imagix.save(file_path=outpath) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*imagix.pkl")) model_files = glob.glob(os.path.join(folder, "\*imagix\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

imagix_loaded = acx.Imagix.load(outpath)

```
Preprocessor saved successfully.
```

```
saving memory efficient
```

```
Pipeline object saved successfully.
PKL files: ['tutorial_res/imagix.pkl']
Model files: ['tutorial_res/imagix3d_model.pth', 'tutorial_res/imagix_model.pth']
Attempting to load a pipeline from tutorial_res/imagix.pkl...
```

```
Pipeline object loaded successfully. Actual type: Imagix
```

```
Preprocessor loaded successfully.
```

In \[10\]:

Copied!

```
result_predict = imagix_loaded.predict(data=backup_datset)
```

result_predict = imagix_loaded.predict(data=backup_datset)

```
Processed 32 / 400 samples
Processed 64 / 400 samples
Processed 96 / 400 samples
Processed 128 / 400 samples
Processed 160 / 400 samples
Processed 192 / 400 samples
Processed 224 / 400 samples
Processed 256 / 400 samples
Processed 288 / 400 samples
Processed 320 / 400 samples
Processed 352 / 400 samples
Processed 384 / 400 samples
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

In \[12\]:

Copied!

```
imagix_loaded.visualizer.show_image_recon_grid(result=imagix_loaded.result, n_samples=5)
```

imagix_loaded.visualizer.show_image_recon_grid(result=imagix_loaded.result, n_samples=5)

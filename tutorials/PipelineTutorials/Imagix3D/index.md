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

# How To Use Imagix3D[¶](#how-to-use-imagix3d)

Imagix3D is an implementation of a variational autoencoder for 3D image data (volumes).\
This tutorial follows the structure of [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Imagix3D/Vanillix.ipynb), and repeats many of the parts therein which cover the general functionaliy of the **AUTOENCODIX** framework. The aim is to provide a tutorial for Imagix3D that simultaneously functions as

- a general demonstration of basic **AUTOENCODIX** functionalities
- a specific tutorial about the 3D image pipeline workflow
- an end-to-end test for Imagix3D that covers all its functionalities

**NOTE:** This tutorial is very similar to the one for [`Getting Started - Imagix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Imagix3D/Imagix.ipynb). So if you already did the `Imagix` tutorial, this one should be easy to follow.

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
Changed to: /home/ewald/Github/autoencodix_package
```

## 1) Initialize the Imagix3D Pipeline[¶](#1-initialize-the-imagix3d-pipeline)

`Imagix3D` is a VAE implementation for 3D image data. Other data modalities are not allowed for `Imagix3D` (including 2D image data).\
This tutorial will be restricted to 3D human medical imaging data. However, the pipeline also works for other 3D data, provided that they come in a data type which is allwed (see down below). To run the pipeline we need to prepare two things:

1. A directory with image files. The following extensions are allowed: `".nii", ".nii.gz"` (for the time being - this might change in the future).
1. An annotation file with metadata, where we map `sample_ids` to image paths. For this, we need to provide the name of the column where the image path information is stored. This is done via the `img_path_col` config parameter.
1. A configuration object

### ❗❗Requirements ❗❗[¶](#requirements)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/mnist](https://huggingface.co/datasets/autoencodix/mnist)) and is downloaded (and extracted) automatically in the cell below on first run.

#### 1.1 📊 The Dataset[¶](#11-the-dataset)

Here we use a subsample of the OrganMNIST3D dataset, which is part of MedMNIST collection (see <https://medmnist.com/>). The original OrganMNIST3D is comprised of abdominal computed tomography (CT) scans and includes the organ types: femure-right/left, bladder, liver, spleen, kidney-right/left, lung-right/left, pancreas and heart. The subsample used here only contains the organs femur-right and bladder, and was drawn across train, validation and test splits.

**data:** 345 images, where 174 are bladder and 171 are femure-right organs, 64x64x64 grayscale. The annotation column `group` gives the organ category for each image.

**NOTE:** Since the data were drawn across pre-defined splits and this pipeline does not use custom-splits, the train\_, valid\_ and test\_ prefixes do not indicate in which split they will be assigned to in the Imagix3D pipeline. They were merely drawn across splits to have enough data of a dataset with a binary category of organ labels that are very distinct.

**A Look Inside the Annotation File**:

In \[3\]:

Copied!

```
import os
import zipfile
import pandas as pd
import autoencodix as acx
from huggingface_hub import hf_hub_download
from autoencodix.configs.imagix3d_config import Imagix3DConfig
from autoencodix.configs.default_config import (
    DataConfig,
    DataCase,
    DataInfo
)


IMGMAPPING = hf_hub_download(
    repo_id="autoencodix/OrganMNIST3D-subsample",
    repo_type="dataset",
    filename="organmnist3d_mappings.csv",
)

zip_path = hf_hub_download(
    repo_id="autoencodix/OrganMNIST3D-subsample",
    repo_type="dataset",
    filename="organmnist3d_images.zip",
)

VOLIMAGES = os.path.join(os.path.dirname(zip_path), "images")

os.makedirs(VOLIMAGES, exist_ok=True)

with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(VOLIMAGES)

anno_df = pd.read_csv(IMGMAPPING, sep=",", index_col=0)
anno_df.head()
```

import os import zipfile import pandas as pd import autoencodix as acx from huggingface_hub import hf_hub_download from autoencodix.configs.imagix3d_config import Imagix3DConfig from autoencodix.configs.default_config import ( DataConfig, DataCase, DataInfo ) IMGMAPPING = hf_hub_download( repo_id="autoencodix/OrganMNIST3D-subsample", repo_type="dataset", filename="organmnist3d_mappings.csv", ) zip_path = hf_hub_download( repo_id="autoencodix/OrganMNIST3D-subsample", repo_type="dataset", filename="organmnist3d_images.zip", ) VOLIMAGES = os.path.join(os.path.dirname(zip_path), "images") os.makedirs(VOLIMAGES, exist_ok=True) with zipfile.ZipFile(zip_path) as zf: zf.extractall(VOLIMAGES) anno_df = pd.read_csv(IMGMAPPING, sep=",", index_col=0) anno_df.head()

Out\[3\]:

|             | filepath           | group       |
| ----------- | ------------------ | ----------- |
| sample_id   |                    |             |
| train_00000 | train_00000.nii.gz | femur-right |
| train_00001 | train_00001.nii.gz | bladder     |
| train_00011 | train_00011.nii.gz | femur-right |
| train_00017 | train_00017.nii.gz | bladder     |
| train_00018 | train_00018.nii.gz | femur-right |

#### 1.2 ⚙️ Configuration Object[¶](#12-configuration-object)

We use the default parameters in `Imagix3DConfig`.\
Additionally, we define a `DataCase`, which tells the pipeline what kind of data to expect.\
In this case: `IMG`.

**Define Config**

In \[4\]:

Copied!

```
volconfig = Imagix3DConfig(
    data_case=DataCase.IMG_TO_IMG,
    img_path_col="filepath",
    spatial_shape_policy="crop_or_pad_to_shape",
    target_shape_3d=(64,64,64),
    checkpoint_interval=10,
    epochs=100,
    learning_rate=0.0005,
    reconstruction_loss="mse",
    beta=0.001,
    scaling="MINMAX",
    anneal_function="logistic-late",
    data_config=DataConfig(
        data_info={
            "IMG": DataInfo(
                file_path=os.path.join(VOLIMAGES, "organmnist3d_subsample"),
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

imagix3d = acx.Imagix3D(config=volconfig)
result = imagix3d.run()
backup_dataset = result.datasets
```

volconfig = Imagix3DConfig( data_case=DataCase.IMG_TO_IMG, img_path_col="filepath", spatial_shape_policy="crop_or_pad_to_shape", target_shape_3d=(64,64,64), checkpoint_interval=10, epochs=100, learning_rate=0.0005, reconstruction_loss="mse", beta=0.001, scaling="MINMAX", anneal_function="logistic-late", data_config=DataConfig( data_info={ "IMG": DataInfo( file_path=os.path.join(VOLIMAGES, "organmnist3d_subsample"), scaling="MINMAX", data_type="IMG", ), "ANNO": DataInfo( file_path=IMGMAPPING, data_type="ANNOTATION", ), }, ), ) imagix3d = acx.Imagix3D(config=volconfig) result = imagix3d.run() backup_dataset = result.datasets

```
Successfully loaded 345 images for IMG
anno key: IMG
Converting 241 images to torch.float32 tensors...
Converting 70 images to torch.float32 tensors...
Converting 34 images to torch.float32 tensors...
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 52.4849 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 8310.2237
Sub-losses: recon_loss: 8310.2237, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 8580.0268
Sub-losses: recon_loss: 8580.0268, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 20 - Train Loss: 6737.2059
Sub-losses: recon_loss: 6737.2059, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 7556.0366
Sub-losses: recon_loss: 7556.0366, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 30 - Train Loss: 6339.6828
Sub-losses: recon_loss: 6339.6828, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 30 - Valid Loss: 7303.8807
Sub-losses: recon_loss: 7303.8807, var_loss: 0.0000, anneal_factor: 0.0
... [output truncated]
```

## 2) Understand Imagix3D Specific Steps[¶](#2-understand-imagix3d-specific-steps)

Since `Imagix3D` is just a `Imagix` for volumes, which in turn is just a `Varix` for images, there are no extra steps for this pipeline.

## 3) Visualize Imagix3D Results[¶](#3-visualize-imagix3d-results)

Since the results of `Imagix3D` are visually interpretable, we add an additional visualization:\
We show a grid of

- original images
- reconstructed images
- and error images (subtraction between the two) with label information on top.

Because these are 3D images, the 2D depictions only show slices from a certain orientation. Therefore the 3D images are shown from the

- axial ('top to bottom'),
- coronal ('front to back'), and
- sagittal ('right to left') orientation.

See the code below for how to obtain this visualization.

In \[5\]:

Copied!

```
imagix3d.visualizer.show_image_recon_grid(
    result=imagix3d.result, 
    n_samples=5)
```

imagix3d.visualizer.show_image_recon_grid( result=imagix3d.result, n_samples=5)

**Standard Visualizations**\
We also offer the standard visualizations as for all other pipelines. You can call the `show_result` method to display the actual plots. By passing the keyword argument `params` and providing a list of columns from the annotation data, you get visualizations based on these columns. In this example, we use the `group` parameter.\
Refer to [5] for more details on how to work with plots.

______________________________________________________________________

[5] Tutorials/DeepDives/VisualizeTutorial.ipynb

In \[6\]:

Copied!

```
imagix3d.show_result(params=["group"])
```

imagix3d.show_result(params=["group"])

```
Creating plots ...
```

## 3) Access Imagix Specific Results[¶](#3-access-imagix-specific-results)

In contrast to `Imagix`, `Imagix3D` does provide pipeline specific results for volumes.

#### Comparing reconstructions to originals, wrt the best, median and worst examples[¶](#comparing-reconstructions-to-originals-wrt-the-best-median-and-worst-examples)

Instead of looking at reconstrunction success from randomly drawn samples, we can here check for the best, median and worst reconstruction cases from our pipeline

In \[7\]:

Copied!

```
imagix3d.visualizer.show_image_recon_grid(
    result=imagix3d.result,
    selection="bmw",
    metric="mse")
```

imagix3d.visualizer.show_image_recon_grid( result=imagix3d.result, selection="bmw", metric="mse")

#### Correlations between latent dimensions[¶](#correlations-between-latent-dimensions)

This visualization shows pairwise `correlations between latent dimensions` to assess how strongly they encode overlapping information.

The optional table and correlation matrix provide a numerical and visual overview of these relationships, where high absolute correlations indicate less independent latent dimensions.

In \[8\]:

Copied!

```
imagix3d.visualizer.show_latent_corr(
    result=imagix3d.result,
    show_table=False,
    show_corrmat=True)
```

imagix3d.visualizer.show_latent_corr( result=imagix3d.result, show_table=False, show_corrmat=True)

#### Inspect latent dimension "activity"[¶](#inspect-latent-dimension-activity)

First defined by Burda et al. (2015)[1](#fn1) and later taken up by Sinha & Dieng (2021)[2](#fn2) the **activity** of the dimension of the latent variables **z** is (Var(x) > \\delta) , where (\\delta) is a user-defined threshold set to 0.01 in the papers. An active dimension is termed an *active unit* in Sinha & Dieng. If the variance of a given dimension is very low it indicates that this very dimension is not used / inactive. The higher the number of latent active units, the better the learned representations.

**compute latent activity...**

In \[9\]:

Copied!

```
imagix3d.results = imagix3d.evaluator.compute_latent_activity(
    result=imagix3d.result,
    threshold=0.01,
    splits=("train", "valid"),
    include_test=True)
```

imagix3d.results = imagix3d.evaluator.compute_latent_activity( result=imagix3d.result, threshold=0.01, splits=("train", "valid"), include_test=True)

**...and show the results**

In \[10\]:

Copied!

```
imagix3d.visualizer.show_latent_activity(
    result=imagix3d.result,
    show_table=False,
    show_dim_table=False)
```

imagix3d.visualizer.show_latent_activity( result=imagix3d.result, show_table=False, show_dim_table=False)

#### Latent Traversal[¶](#latent-traversal)

We can use latent traversal as a qualitative visualization technique to inspect what individual latent dimensions have learned to represent. Starting from a fixed latent vector, one latent dimension is varied while all other dimensions are held constant, and the decoder generates the corresponding output images. If changing a specific latent dimension produces changes in the reconstructed volume, this suggests that the dimension captures a meaningful factor of variation in the data.

**Compute latent traversal...**

In \[11\]:

Copied!

```
imagix3d.result =imagix3d.evaluator.compute_latent_traversal(
    result=imagix3d.result,
    latent_dims="top_kl",
    n_latent_dims=8,
    value_mode="percentile",
    n_values=5)
```

imagix3d.result =imagix3d.evaluator.compute_latent_traversal( result=imagix3d.result, latent_dims="top_kl", n_latent_dims=8, value_mode="percentile", n_values=5)

**...and show the results**

In \[12\]:

Copied!

```
imagix3d.visualizer.show_latent_traversal(
    result=imagix3d.result,
    view="axial")
```

imagix3d.visualizer.show_latent_traversal( result=imagix3d.result, view="axial")

## 5) Customize Imagix3D[¶](#5-customize-imagix3d)

To customize the behavior of our pipeline, you adjust the configuration.\
There are two ways to work with the config:

1. Create a customized instance of the config class.
1. Provide a `yaml` file and use the config class to read it.

We will focus on option 1 and show a few examples. Via the `Config` object we can apply customize the pipeline by chaning the `loss function` or the number of `epoch` or preprocessing. To get an full overview of the adjustable parameters, please refere to [2].

In Addition, there is a way to use another architecture for the image autoencoder. We currently offer two architectures:`VolumeVAEArchitecture` and `VolumeVAEFastArchitecture`. The fast architecture does not has BatchNorm and inplace activation functions, which can speed up the training process. The `VolumeVAEArchitecture` is the default architecture.

We can simply pass the architecture as type to the `model_type` parameter of the `Imagix3D` pipeline.

[2] Tutorials/DeepDives/ConfigTutorial.ipynb.

In \[14\]:

Copied!

```
from autoencodix.modeling._volfast_architecture import VolumeVAEFastArchitecture
from autoencodix.modeling._volumevae_architecture import VolumeVAEArchitecture

custom_volconfig = Imagix3DConfig(
    data_case=DataCase.IMG_TO_IMG,
    img_path_col="filepath",
    spatial_shape_policy="crop_or_pad_to_shape",
    target_shape_3d=(64,64,64),
    checkpoint_interval=10,
    epochs=100,
    learning_rate=0.001,
    reconstruction_loss="mse",
    beta=0.005,
    scaling="MINMAX",
    anneal_function="logistic-late",
    data_config=DataConfig(
        data_info={
            "IMG": DataInfo(
                file_path=os.path.join(VOLIMAGES, "organmnist3d_subsample"),
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

imagix3d = acx.Imagix3D(
    config=custom_volconfig, 
    model_type=VolumeVAEFastArchitecture
)
result = imagix3d.run()
backup_dataset = result.datasets
```

from autoencodix.modeling.\_volfast_architecture import VolumeVAEFastArchitecture from autoencodix.modeling.\_volumevae_architecture import VolumeVAEArchitecture custom_volconfig = Imagix3DConfig( data_case=DataCase.IMG_TO_IMG, img_path_col="filepath", spatial_shape_policy="crop_or_pad_to_shape", target_shape_3d=(64,64,64), checkpoint_interval=10, epochs=100, learning_rate=0.001, reconstruction_loss="mse", beta=0.005, scaling="MINMAX", anneal_function="logistic-late", data_config=DataConfig( data_info={ "IMG": DataInfo( file_path=os.path.join(VOLIMAGES, "organmnist3d_subsample"), scaling="MINMAX", data_type="IMG", ), "ANNO": DataInfo( file_path=IMGMAPPING, data_type="ANNOTATION", ), }, ), ) imagix3d = acx.Imagix3D( config=custom_volconfig, model_type=VolumeVAEFastArchitecture ) result = imagix3d.run() backup_dataset = result.datasets

```
Successfully loaded 345 images for IMG
anno key: IMG
Converting 241 images to torch.float32 tensors...
Converting 70 images to torch.float32 tensors...
Converting 34 images to torch.float32 tensors...
Epoch 10 - Train Loss: 18228.9796
Sub-losses: recon_loss: 18228.9796, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 16439.0481
Sub-losses: recon_loss: 16439.0481, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 20 - Train Loss: 12809.1013
Sub-losses: recon_loss: 12809.1013, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 11979.7325
Sub-losses: recon_loss: 11979.7325, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 30 - Train Loss: 11307.0336
... [output truncated]
```

## 6) Load Save and Reuse Imagix3D[¶](#6-load-save-and-reuse-imagix3d)

Lastly, we show how to re-use a trained pipeline, as well as how to save and load it.\
You can do the following:

- Use the trained pipeline to run the `predict` step with new data.
- Obtain the latent space, apply transformations, and run the `decode` step of the trained pipeline.
- Generate new data by computing the mean and standard deviation of all encoded latent vectors in the chosen split and epoch, fitting a diagonal Gaussian to those statistics, and sampling new latent points by adding scaled Gaussian noise to that mean.
- Access the underlying PyTorch model and re-use it in any way you like.
- Store the pipeline on disk or in the cloud, then reload it and continue using it

#### Save and Load[¶](#save-and-load)

Best practice is to provide a file path without extension, we set the extensions internally and handle all saving and loading.

In \[15\]:

Copied!

```
import os
import glob

outpath = os.path.join("tutorial_res", "imagix3d.pkl")

imagix3d.save(file_path=outpath,
             save_all=True)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*imagix3d.pkl"))
model_files = glob.glob(os.path.join(folder, "*imagix3d.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
imagix3d_loaded = acx.Imagix3D.load(outpath)
```

import os import glob outpath = os.path.join("tutorial_res", "imagix3d.pkl") imagix3d.save(file_path=outpath, save_all=True) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*imagix3d.pkl")) model_files = glob.glob(os.path.join(folder, "\*imagix3d.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

imagix3d_loaded = acx.Imagix3D.load(outpath)

```
Preprocessor saved successfully.
Pipeline object saved successfully.
PKL files: ['tutorial_res/imagix3d.pkl']
Model files: []
Attempting to load a pipeline from tutorial_res/imagix3d.pkl...
Pipeline object loaded successfully. Actual type: Imagix3D
Preprocessor loaded successfully.
```

**Visualize on new predictions**

In \[16\]:

Copied!

```
imagix3d_loaded.show_result(
    split="test",
    params=["group"])
```

imagix3d_loaded.show_result( split="test", params=["group"])

```
Creating plots ...
```

In \[17\]:

Copied!

```
imagix3d_loaded.visualizer.show_image_recon_grid(
    result=imagix3d_loaded.result, 
    n_samples=5)
```

imagix3d_loaded.visualizer.show_image_recon_grid( result=imagix3d_loaded.result, n_samples=5)

**References**

1 @misc{burda2016importanceweightedautoencoders, title={Importance Weighted Autoencoders}, author={Yuri Burda and Roger Grosse and Ruslan Salakhutdinov}, year={2016}, eprint={1509.00519}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={[https://arxiv.org/abs/1509.00519}](https://arxiv.org/abs/1509.00519%7D), } [↩] (#fn1)

2 @misc{sinha2022consistencyregularizationvariationalautoencoders, title={Consistency Regularization for Variational Auto-Encoders}, author={Samarth Sinha and Adji B. Dieng}, year={2022}, eprint={2105.14859}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={[https://arxiv.org/abs/2105.14859}](https://arxiv.org/abs/2105.14859%7D), } [↩] (#fn2)

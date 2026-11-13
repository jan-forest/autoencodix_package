> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[1\]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
# Needed for the TabPFN example below. Must be set before scipy or scikit-learn are imported.
%env SCIPY_ARRAY_API=1
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

# Needed for the TabPFN example below. Must be set before scipy or scikit-learn are imported.

%env SCIPY_ARRAY_API=1

```
env: CUBLAS_WORKSPACE_CONFIG=:16:8
env: SCIPY_ARRAY_API=1
```

# Evaluate Overview[¶](#evaluate-overview)

This tutorial provides an overview of the in-built functionalities to evaluate the learned embeddings for downstream machine learning tasks leveraging metadata information.

Evaluation is designed as additional pipeline step after `.run()` (which includes preprocessing, training, predict on test data and visualization).

When you work with a pre-trained model, make sure that you perform the `.predict()` step such that embeddings are calculated for newly provided test-samples.

## Stackix example[¶](#stackix-example)

We will use a Stackix as an example to show options in embedding evaluation.

In \[2\]:

Copied!

```
# Train the Stackix similar to pipeline tutorial
import autoencodix as acx
from autoencodix.configs.default_config import DataCase
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK

# Stackix has its own config class
# instead of passing a pandas DataFrame, we use a pre-defined DataPackage object directly.
# this time with single cell data
print("Input data:")
print(EXAMPLE_MULTI_BULK)
print("-" * 50)
my_config = StackixConfig(
    epochs=100,
    checkpoint_interval=5,
    default_vae_loss="kl",  
    data_case=DataCase.MULTI_BULK,
)
print("\n")
print("Starting Pipeline")
print("-" * 50)
print("-" * 50)
stackix = acx.Stackix(data=EXAMPLE_MULTI_BULK, config=my_config)
result = stackix.run()
```

# Train the Stackix similar to pipeline tutorial

import autoencodix as acx from autoencodix.configs.default_config import DataCase from autoencodix.configs.stackix_config import StackixConfig from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK

# Stackix has its own config class

# instead of passing a pandas DataFrame, we use a pre-defined DataPackage object directly.

# this time with single cell data

print("Input data:") print(EXAMPLE_MULTI_BULK) print("-" * 50) my_config = StackixConfig( epochs=100, checkpoint_interval=5, default_vae_loss="kl", data_case=DataCase.MULTI_BULK, ) print("\\n") print("Starting Pipeline") print("-" * 50) print("-" * 50) stackix = acx.Stackix(data=EXAMPLE_MULTI_BULK, config=my_config) result = stackix.run()

```
Input data:
multi_bulk:
  transcriptomics: 500 samples × 100 features
  proteomics: 500 samples × 80 features
annotation:
  transcriptomics: 500 samples × 3 features
  proteomics: 500 samples × 3 features
--------------------------------------------------


Starting Pipeline
--------------------------------------------------
--------------------------------------------------
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: transcriptomics
anno key: proteomics
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
Training each modality model...
Training modality: transcriptomics
Training modality: transcriptomics
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 12.6849 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 108.0478
Sub-losses: recon_loss: 108.0478, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 5 - Valid Loss: 88.4555
Sub-losses: recon_loss: 88.4555, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
```

```
Epoch 10 - Train Loss: 103.0511
Sub-losses: recon_loss: 103.0510, var_loss: 0.0001, anneal_factor: 0.0003, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 87.0261
Sub-losses: recon_loss: 87.0261, var_loss: 0.0001, anneal_factor: 0.0003, effective_beta_factor: 0.0000
```

```
Epoch 15 - Train Loss: 99.6199
Sub-losses: recon_loss: 99.6194, var_loss: 0.0005, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 15 - Valid Loss: 84.7559
Sub-losses: recon_loss: 84.7557, var_loss: 0.0003, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 20 - Train Loss: 96.6244
Sub-losses: recon_loss: 96.6227, var_loss: 0.0017, anneal_factor: 0.0020, effective_beta_factor: 0.0002
Epoch 20 - Valid Loss: 80.5584
Sub-losses: recon_loss: 80.5572, var_loss: 0.0012, anneal_factor: 0.0020, effective_beta_factor: 0.0002
```

```
Epoch 25 - Train Loss: 92.3356
Sub-losses: recon_loss: 92.3295, var_loss: 0.0061, anneal_factor: 0.0055, effective_beta_factor: 0.0005
Epoch 25 - Valid Loss: 78.4712
Sub-losses: recon_loss: 78.4674, var_loss: 0.0038, anneal_factor: 0.0055, effective_beta_factor: 0.0005
```

```
Epoch 30 - Train Loss: 90.3097
Sub-losses: recon_loss: 90.2879, var_loss: 0.0219, anneal_factor: 0.0148, effective_beta_factor: 0.0015
Epoch 30 - Valid Loss: 76.3916
Sub-losses: recon_loss: 76.3779, var_loss: 0.0138, anneal_factor: 0.0148, effective_beta_factor: 0.0015
```

```
Epoch 35 - Train Loss: 87.2004
Sub-losses: recon_loss: 87.1304, var_loss: 0.0701, anneal_factor: 0.0392, effective_beta_factor: 0.0039
Epoch 35 - Valid Loss: 74.4117
Sub-losses: recon_loss: 74.3605, var_loss: 0.0512, anneal_factor: 0.0392, effective_beta_factor: 0.0039
```

```
Epoch 40 - Train Loss: 86.4785
Sub-losses: recon_loss: 86.2575, var_loss: 0.2211, anneal_factor: 0.0998, effective_beta_factor: 0.0100
Epoch 40 - Valid Loss: 73.4207
Sub-losses: recon_loss: 73.2544, var_loss: 0.1663, anneal_factor: 0.0998, effective_beta_factor: 0.0100
```

```
Epoch 45 - Train Loss: 84.9065
Sub-losses: recon_loss: 84.3197, var_loss: 0.5868, anneal_factor: 0.2315, effective_beta_factor: 0.0231
Epoch 45 - Valid Loss: 71.6050
Sub-losses: recon_loss: 71.1700, var_loss: 0.4350, anneal_factor: 0.2315, effective_beta_factor: 0.0231
```

```
Epoch 50 - Train Loss: 84.5510
Sub-losses: recon_loss: 83.3356, var_loss: 1.2153, anneal_factor: 0.4502, effective_beta_factor: 0.0450
Epoch 50 - Valid Loss: 71.7988
Sub-losses: recon_loss: 70.8895, var_loss: 0.9093, anneal_factor: 0.4502, effective_beta_factor: 0.0450
```

```
Epoch 55 - Train Loss: 84.3188
Sub-losses: recon_loss: 82.4651, var_loss: 1.8537, anneal_factor: 0.6900, effective_beta_factor: 0.0690
Epoch 55 - Valid Loss: 70.9629
Sub-losses: recon_loss: 69.7004, var_loss: 1.2625, anneal_factor: 0.6900, effective_beta_factor: 0.0690
```

```
Epoch 60 - Train Loss: 83.9415
Sub-losses: recon_loss: 81.8715, var_loss: 2.0700, anneal_factor: 0.8581, effective_beta_factor: 0.0858
Epoch 60 - Valid Loss: 70.6853
Sub-losses: recon_loss: 69.1945, var_loss: 1.4908, anneal_factor: 0.8581, effective_beta_factor: 0.0858
```

```
Epoch 65 - Train Loss: 83.2220
Sub-losses: recon_loss: 81.1184, var_loss: 2.1036, anneal_factor: 0.9427, effective_beta_factor: 0.0943
Epoch 65 - Valid Loss: 70.3990
Sub-losses: recon_loss: 68.8988, var_loss: 1.5002, anneal_factor: 0.9427, effective_beta_factor: 0.0943
```

```
Epoch 70 - Train Loss: 81.9846
Sub-losses: recon_loss: 79.9617, var_loss: 2.0229, anneal_factor: 0.9781, effective_beta_factor: 0.0978
Epoch 70 - Valid Loss: 69.3109
Sub-losses: recon_loss: 67.9043, var_loss: 1.4066, anneal_factor: 0.9781, effective_beta_factor: 0.0978
```

```
Epoch 75 - Train Loss: 80.8361
Sub-losses: recon_loss: 78.8616, var_loss: 1.9744, anneal_factor: 0.9918, effective_beta_factor: 0.0992
Epoch 75 - Valid Loss: 69.1859
Sub-losses: recon_loss: 67.8625, var_loss: 1.3234, anneal_factor: 0.9918, effective_beta_factor: 0.0992
```

```
Epoch 80 - Train Loss: 82.6688
Sub-losses: recon_loss: 80.6819, var_loss: 1.9869, anneal_factor: 0.9970, effective_beta_factor: 0.0997
Epoch 80 - Valid Loss: 68.1446
Sub-losses: recon_loss: 66.7382, var_loss: 1.4064, anneal_factor: 0.9970, effective_beta_factor: 0.0997
```

```
Epoch 85 - Train Loss: 80.3141
Sub-losses: recon_loss: 78.3306, var_loss: 1.9836, anneal_factor: 0.9989, effective_beta_factor: 0.0999
Epoch 85 - Valid Loss: 68.5901
Sub-losses: recon_loss: 67.2641, var_loss: 1.3260, anneal_factor: 0.9989, effective_beta_factor: 0.0999
```

```
Epoch 90 - Train Loss: 80.5890
Sub-losses: recon_loss: 78.5321, var_loss: 2.0569, anneal_factor: 0.9996, effective_beta_factor: 0.1000
Epoch 90 - Valid Loss: 67.4627
Sub-losses: recon_loss: 66.0958, var_loss: 1.3669, anneal_factor: 0.9996, effective_beta_factor: 0.1000
```

```
Epoch 95 - Train Loss: 80.8003
Sub-losses: recon_loss: 78.6748, var_loss: 2.1254, anneal_factor: 0.9998, effective_beta_factor: 0.1000
Epoch 95 - Valid Loss: 67.2400
Sub-losses: recon_loss: 65.8173, var_loss: 1.4227, anneal_factor: 0.9998, effective_beta_factor: 0.1000
```

```
Epoch 100 - Train Loss: 80.4735
Sub-losses: recon_loss: 78.3721, var_loss: 2.1014, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 100 - Valid Loss: 67.0705
Sub-losses: recon_loss: 65.6822, var_loss: 1.3883, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Training modality: proteomics
Training modality: proteomics
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.7820 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 86.5841
Sub-losses: recon_loss: 86.5841, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 5 - Valid Loss: 70.9752
Sub-losses: recon_loss: 70.9752, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
```

```
Epoch 10 - Train Loss: 83.0158
Sub-losses: recon_loss: 83.0157, var_loss: 0.0001, anneal_factor: 0.0003, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 69.4432
Sub-losses: recon_loss: 69.4431, var_loss: 0.0001, anneal_factor: 0.0003, effective_beta_factor: 0.0000
```

```
Epoch 15 - Train Loss: 80.4179
Sub-losses: recon_loss: 80.4174, var_loss: 0.0005, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 15 - Valid Loss: 66.5124
Sub-losses: recon_loss: 66.5121, var_loss: 0.0003, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 20 - Train Loss: 76.3860
Sub-losses: recon_loss: 76.3842, var_loss: 0.0018, anneal_factor: 0.0020, effective_beta_factor: 0.0002
Epoch 20 - Valid Loss: 63.9436
Sub-losses: recon_loss: 63.9426, var_loss: 0.0010, anneal_factor: 0.0020, effective_beta_factor: 0.0002
```

```
Epoch 25 - Train Loss: 73.7199
Sub-losses: recon_loss: 73.7129, var_loss: 0.0070, anneal_factor: 0.0055, effective_beta_factor: 0.0005
Epoch 25 - Valid Loss: 61.6779
Sub-losses: recon_loss: 61.6738, var_loss: 0.0041, anneal_factor: 0.0055, effective_beta_factor: 0.0005
```

```
Epoch 30 - Train Loss: 69.7901
Sub-losses: recon_loss: 69.7662, var_loss: 0.0239, anneal_factor: 0.0148, effective_beta_factor: 0.0015
Epoch 30 - Valid Loss: 59.3963
Sub-losses: recon_loss: 59.3819, var_loss: 0.0143, anneal_factor: 0.0148, effective_beta_factor: 0.0015
```

```
Epoch 35 - Train Loss: 68.2430
Sub-losses: recon_loss: 68.1613, var_loss: 0.0816, anneal_factor: 0.0392, effective_beta_factor: 0.0039
Epoch 35 - Valid Loss: 58.1488
Sub-losses: recon_loss: 58.1020, var_loss: 0.0468, anneal_factor: 0.0392, effective_beta_factor: 0.0039
```

```
Epoch 40 - Train Loss: 66.8286
Sub-losses: recon_loss: 66.5896, var_loss: 0.2390, anneal_factor: 0.0998, effective_beta_factor: 0.0100
Epoch 40 - Valid Loss: 57.0197
Sub-losses: recon_loss: 56.8704, var_loss: 0.1493, anneal_factor: 0.0998, effective_beta_factor: 0.0100
```

```
Epoch 45 - Train Loss: 65.7826
Sub-losses: recon_loss: 65.1619, var_loss: 0.6207, anneal_factor: 0.2315, effective_beta_factor: 0.0231
Epoch 45 - Valid Loss: 55.3971
Sub-losses: recon_loss: 55.0178, var_loss: 0.3793, anneal_factor: 0.2315, effective_beta_factor: 0.0231
```

```
Epoch 50 - Train Loss: 64.3450
Sub-losses: recon_loss: 63.1128, var_loss: 1.2322, anneal_factor: 0.4502, effective_beta_factor: 0.0450
Epoch 50 - Valid Loss: 55.5587
Sub-losses: recon_loss: 54.8268, var_loss: 0.7319, anneal_factor: 0.4502, effective_beta_factor: 0.0450
```

```
Epoch 55 - Train Loss: 64.8403
Sub-losses: recon_loss: 62.9955, var_loss: 1.8448, anneal_factor: 0.6900, effective_beta_factor: 0.0690
Epoch 55 - Valid Loss: 54.4682
Sub-losses: recon_loss: 53.3400, var_loss: 1.1282, anneal_factor: 0.6900, effective_beta_factor: 0.0690
```

```
Epoch 60 - Train Loss: 64.2589
Sub-losses: recon_loss: 62.1341, var_loss: 2.1248, anneal_factor: 0.8581, effective_beta_factor: 0.0858
Epoch 60 - Valid Loss: 53.9488
Sub-losses: recon_loss: 52.6301, var_loss: 1.3187, anneal_factor: 0.8581, effective_beta_factor: 0.0858
```

```
Epoch 65 - Train Loss: 63.5126
Sub-losses: recon_loss: 61.4315, var_loss: 2.0810, anneal_factor: 0.9427, effective_beta_factor: 0.0943
Epoch 65 - Valid Loss: 53.3113
Sub-losses: recon_loss: 52.0064, var_loss: 1.3049, anneal_factor: 0.9427, effective_beta_factor: 0.0943
```

```
Epoch 70 - Train Loss: 63.0010
Sub-losses: recon_loss: 60.8710, var_loss: 2.1300, anneal_factor: 0.9781, effective_beta_factor: 0.0978
Epoch 70 - Valid Loss: 53.3223
Sub-losses: recon_loss: 51.9459, var_loss: 1.3764, anneal_factor: 0.9781, effective_beta_factor: 0.0978
```

```
Epoch 75 - Train Loss: 63.4723
Sub-losses: recon_loss: 61.3764, var_loss: 2.0958, anneal_factor: 0.9918, effective_beta_factor: 0.0992
Epoch 75 - Valid Loss: 53.0144
Sub-losses: recon_loss: 51.5550, var_loss: 1.4594, anneal_factor: 0.9918, effective_beta_factor: 0.0992
```

```
Epoch 80 - Train Loss: 62.8066
Sub-losses: recon_loss: 60.7046, var_loss: 2.1020, anneal_factor: 0.9970, effective_beta_factor: 0.0997
Epoch 80 - Valid Loss: 52.2245
Sub-losses: recon_loss: 50.8046, var_loss: 1.4199, anneal_factor: 0.9970, effective_beta_factor: 0.0997
```

```
Epoch 85 - Train Loss: 62.9928
Sub-losses: recon_loss: 60.9297, var_loss: 2.0632, anneal_factor: 0.9989, effective_beta_factor: 0.0999
Epoch 85 - Valid Loss: 51.5968
Sub-losses: recon_loss: 50.1624, var_loss: 1.4344, anneal_factor: 0.9989, effective_beta_factor: 0.0999
```

```
Epoch 90 - Train Loss: 62.3466
Sub-losses: recon_loss: 60.2128, var_loss: 2.1338, anneal_factor: 0.9996, effective_beta_factor: 0.1000
Epoch 90 - Valid Loss: 50.9410
Sub-losses: recon_loss: 49.4521, var_loss: 1.4889, anneal_factor: 0.9996, effective_beta_factor: 0.1000
```

```
Epoch 95 - Train Loss: 61.9138
Sub-losses: recon_loss: 59.7570, var_loss: 2.1569, anneal_factor: 0.9998, effective_beta_factor: 0.1000
Epoch 95 - Valid Loss: 52.8541
Sub-losses: recon_loss: 51.3344, var_loss: 1.5197, anneal_factor: 0.9998, effective_beta_factor: 0.1000
```

```
Epoch 100 - Train Loss: 61.4132
Sub-losses: recon_loss: 59.2459, var_loss: 2.1673, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 100 - Valid Loss: 50.9114
Sub-losses: recon_loss: 49.3939, var_loss: 1.5175, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Found 350 common samples for the stacked autoencoder.
Found 50 common samples for the stacked autoencoder.
finished training each modality model
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 35.5739 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 112.4400
Sub-losses: recon_loss: 112.4399, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 5 - Valid Loss: 83.1644
Sub-losses: recon_loss: 83.1643, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
```

```
Epoch 10 - Train Loss: 101.7353
Sub-losses: recon_loss: 101.7352, var_loss: 0.0001, anneal_factor: 0.0003, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 76.8087
Sub-losses: recon_loss: 76.8087, var_loss: 0.0001, anneal_factor: 0.0003, effective_beta_factor: 0.0000
```

```
Epoch 15 - Train Loss: 95.3655
Sub-losses: recon_loss: 95.3651, var_loss: 0.0004, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 15 - Valid Loss: 73.0526
Sub-losses: recon_loss: 73.0524, var_loss: 0.0002, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 20 - Train Loss: 91.1731
Sub-losses: recon_loss: 91.1720, var_loss: 0.0012, anneal_factor: 0.0020, effective_beta_factor: 0.0002
Epoch 20 - Valid Loss: 69.6160
Sub-losses: recon_loss: 69.6155, var_loss: 0.0005, anneal_factor: 0.0020, effective_beta_factor: 0.0002
```

```
Epoch 25 - Train Loss: 88.7470
Sub-losses: recon_loss: 88.7434, var_loss: 0.0036, anneal_factor: 0.0055, effective_beta_factor: 0.0005
Epoch 25 - Valid Loss: 68.5087
Sub-losses: recon_loss: 68.5070, var_loss: 0.0017, anneal_factor: 0.0055, effective_beta_factor: 0.0005
```

```
Epoch 30 - Train Loss: 85.2641
Sub-losses: recon_loss: 85.2521, var_loss: 0.0120, anneal_factor: 0.0148, effective_beta_factor: 0.0015
Epoch 30 - Valid Loss: 67.0588
Sub-losses: recon_loss: 67.0536, var_loss: 0.0052, anneal_factor: 0.0148, effective_beta_factor: 0.0015
```

```
Epoch 35 - Train Loss: 80.5045
Sub-losses: recon_loss: 80.4628, var_loss: 0.0417, anneal_factor: 0.0392, effective_beta_factor: 0.0039
Epoch 35 - Valid Loss: 63.2416
Sub-losses: recon_loss: 63.2224, var_loss: 0.0193, anneal_factor: 0.0392, effective_beta_factor: 0.0039
```

```
Epoch 40 - Train Loss: 76.7371
Sub-losses: recon_loss: 76.5946, var_loss: 0.1425, anneal_factor: 0.0998, effective_beta_factor: 0.0100
Epoch 40 - Valid Loss: 63.1165
Sub-losses: recon_loss: 63.0564, var_loss: 0.0601, anneal_factor: 0.0998, effective_beta_factor: 0.0100
```

```
Epoch 45 - Train Loss: 74.5967
Sub-losses: recon_loss: 74.2104, var_loss: 0.3863, anneal_factor: 0.2315, effective_beta_factor: 0.0231
Epoch 45 - Valid Loss: 60.0691
Sub-losses: recon_loss: 59.8729, var_loss: 0.1961, anneal_factor: 0.2315, effective_beta_factor: 0.0231
```

```
Epoch 50 - Train Loss: 71.3007
Sub-losses: recon_loss: 70.4079, var_loss: 0.8928, anneal_factor: 0.4502, effective_beta_factor: 0.0450
Epoch 50 - Valid Loss: 58.6705
Sub-losses: recon_loss: 58.1986, var_loss: 0.4719, anneal_factor: 0.4502, effective_beta_factor: 0.0450
```

```
Epoch 55 - Train Loss: 72.9087
Sub-losses: recon_loss: 71.4725, var_loss: 1.4362, anneal_factor: 0.6900, effective_beta_factor: 0.0690
Epoch 55 - Valid Loss: 56.8277
Sub-losses: recon_loss: 56.0487, var_loss: 0.7790, anneal_factor: 0.6900, effective_beta_factor: 0.0690
```

```
Epoch 60 - Train Loss: 72.1475
Sub-losses: recon_loss: 70.3444, var_loss: 1.8030, anneal_factor: 0.8581, effective_beta_factor: 0.0858
Epoch 60 - Valid Loss: 56.2429
Sub-losses: recon_loss: 55.2860, var_loss: 0.9568, anneal_factor: 0.8581, effective_beta_factor: 0.0858
```

```
Epoch 65 - Train Loss: 71.0535
Sub-losses: recon_loss: 69.1609, var_loss: 1.8926, anneal_factor: 0.9427, effective_beta_factor: 0.0943
Epoch 65 - Valid Loss: 55.8167
Sub-losses: recon_loss: 54.7807, var_loss: 1.0360, anneal_factor: 0.9427, effective_beta_factor: 0.0943
```

```
Epoch 70 - Train Loss: 69.4062
Sub-losses: recon_loss: 67.4491, var_loss: 1.9571, anneal_factor: 0.9781, effective_beta_factor: 0.0978
Epoch 70 - Valid Loss: 55.7657
Sub-losses: recon_loss: 54.7775, var_loss: 0.9883, anneal_factor: 0.9781, effective_beta_factor: 0.0978
```

```
Epoch 75 - Train Loss: 69.8546
Sub-losses: recon_loss: 67.9446, var_loss: 1.9099, anneal_factor: 0.9918, effective_beta_factor: 0.0992
Epoch 75 - Valid Loss: 55.1678
Sub-losses: recon_loss: 54.1349, var_loss: 1.0329, anneal_factor: 0.9918, effective_beta_factor: 0.0992
```

```
Epoch 80 - Train Loss: 68.2038
Sub-losses: recon_loss: 66.3589, var_loss: 1.8449, anneal_factor: 0.9970, effective_beta_factor: 0.0997
Epoch 80 - Valid Loss: 54.4344
Sub-losses: recon_loss: 53.4368, var_loss: 0.9976, anneal_factor: 0.9970, effective_beta_factor: 0.0997
```

```
Epoch 85 - Train Loss: 68.4435
Sub-losses: recon_loss: 66.5764, var_loss: 1.8671, anneal_factor: 0.9989, effective_beta_factor: 0.0999
Epoch 85 - Valid Loss: 53.7548
Sub-losses: recon_loss: 52.6583, var_loss: 1.0965, anneal_factor: 0.9989, effective_beta_factor: 0.0999
```

```
Epoch 90 - Train Loss: 67.4005
Sub-losses: recon_loss: 65.5439, var_loss: 1.8566, anneal_factor: 0.9996, effective_beta_factor: 0.1000
Epoch 90 - Valid Loss: 53.7878
Sub-losses: recon_loss: 52.6902, var_loss: 1.0976, anneal_factor: 0.9996, effective_beta_factor: 0.1000
```

```
Epoch 95 - Train Loss: 69.2402
Sub-losses: recon_loss: 67.2834, var_loss: 1.9568, anneal_factor: 0.9998, effective_beta_factor: 0.1000
Epoch 95 - Valid Loss: 53.5047
Sub-losses: recon_loss: 52.3869, var_loss: 1.1178, anneal_factor: 0.9998, effective_beta_factor: 0.1000
```

```
Epoch 100 - Train Loss: 68.4316
Sub-losses: recon_loss: 66.5503, var_loss: 1.8813, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 100 - Valid Loss: 52.5694
Sub-losses: recon_loss: 51.4780, var_loss: 1.0914, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 100 samples
Processed 64 / 100 samples
Processed 96 / 100 samples
Processed 100 / 100 samples
Processed 32 / 100 samples
Processed 64 / 100 samples
Processed 96 / 100 samples
Processed 100 / 100 samples
Found 100 common samples for the stacked autoencoder.
<autoencodix.data._numeric_dataset.NumericDataset object at 0x76d4b01ebb60>
Processed 32 / 100 samples
Processed 64 / 100 samples
Processed 96 / 100 samples
Processed 100 / 100 samples
Successfully created annotated latent space object (adata_latent).
```

Before we perform embedding evaluation on a metadata parameter. Let's check visually if latent space shows discrimination regarding this parameter.

In \[3\]:

Copied!

```
my_target_param = "condition"  

stackix.visualizer.show_latent_space(
	result= result,
	plot_type="Ridgeline",
	param= [my_target_param]
)
```

my_target_param = "condition" stackix.visualizer.show_latent_space( result= result, plot_type="Ridgeline", param= [my_target_param] )

We can see a separation of conditions along several dimension. Hence, we can expect that embeddings are useful to classify.

TODO make example stackix better

Now we can call the pipeline step `stackix.evaluate()`

In \[4\]:

Copied!

```
result = stackix.evaluate(params=[my_target_param])

print("Evaluation results:")
print(result.embedding_evaluation)
```

result = stackix.evaluate(params=[my_target_param]) print("Evaluation results:") print(result.embedding_evaluation)

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
Evaluation results:
  score_split CLINIC_PARAM       metric     value              ML_ALG  \
0       train    condition  roc_auc_ovo  0.741726  LogisticRegression   
1       valid    condition  roc_auc_ovo  0.655958  LogisticRegression   
2        test    condition  roc_auc_ovo  0.572573  LogisticRegression   

          ML_TYPE ML_TASK ML_SUBTASK  
0  classification  Latent     Latent  
1  classification  Latent     Latent  
2  classification  Latent     Latent
```

```
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
```

From the dataframe containing the results, we can see that default configuration of `.evaluate()` uses linear machine learning models for regression and classification tasks.

But evaluate offers many options for a detailed inspection:

- With `ml_model_class` and `ml_model_regression` user can provide any machine learning method if they are using a sklearn-like interface.
- `params` - Is the list of metadata parameter on which downstream tasks will be performed. If not a list, but the string "all" is provided, all parameters will be evaluated. Type of task classification vs. regression will be auto-detected. Make sure to avoid providing class information as integers or floats.
- With `metric_class` and `metric_regression` the metric for evaluation is specified for classification (Default: `roc_auc_ovo`) or regression (Default: `r2`). Must be one as described in sklearn: <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names>
- `split_type` - Specifies how data should be splitted for training and testing downstream ML models. Default is `use-split` which will use established split and will train on the train split. Alternatively and in case there is no train split, cross-validation can be performed by using a string like "LOOCV", "CV-5", ... "CV-10", where the number specifies the number of folds.
- `n_downsample` - To speed up evaluation on large data sets, evaluation can be performed on a random subset. Default is 10000. You may need to increase in case of extreme class-imbalance.
- `reference_methods` - For cross-refence of embedding performance you can check (provided as list):
  - "RandomFeature" will pick randomly pick input features in the same number as `latent_dim` and provides a lower performance baseline. Will be repeated five times.
  - "PCA", "UMAP" or "TSNE" will perform dimension reduction to `latent_dim` to compare autoencoder embeddings to other techniques.

Let's check this with an advanced example:

In \[5\]:

Copied!

```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)  # Use a custom classifier from sklearn including hyperparameters
metric_class = "accuracy"  # Choose another metric to evaluate classification performance
split_type = "CV-5" # Use 5-fold cross-validation instead of the predefined train/test split
reference_methods = ["RandomFeature", "PCA"]  # Compare results to additional reference methods

result = stackix.evaluate(
	params=[my_target_param],
    ml_model_class=rf,
    metric_class=metric_class,
    split_type=split_type,
    reference_methods=reference_methods
)
```

from sklearn.ensemble import RandomForestClassifier rf = RandomForestClassifier(n_estimators=100) # Use a custom classifier from sklearn including hyperparameters metric_class = "accuracy" # Choose another metric to evaluate classification performance split_type = "CV-5" # Use 5-fold cross-validation instead of the predefined train/test split reference_methods = ["RandomFeature", "PCA"] # Compare results to additional reference methods result = stackix.evaluate( params=[my_target_param], ml_model_class=rf, metric_class=metric_class, split_type=split_type, reference_methods=reference_methods )

```
Perform ML task with feature df: RandomFeature
RandomFeature_R1
Perform ML task for target parameter: condition
```

```
RandomFeature_R2
Perform ML task for target parameter: condition
```

```
RandomFeature_R3
Perform ML task for target parameter: condition
```

```
RandomFeature_R4
Perform ML task for target parameter: condition
```

```
RandomFeature_R5
Perform ML task for target parameter: condition
```

```
Perform ML task with feature df: PCA
PCA
Perform ML task for target parameter: condition
```

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
```

Note, that evaluation results will be updated with additional evaluate calls:

In \[6\]:

Copied!

```
print("Evaluation results:")
print(result.embedding_evaluation)
```

print("Evaluation results:") print(result.embedding_evaluation)

```
Evaluation results:
   score_split CLINIC_PARAM       metric     value                  ML_ALG  \
0        train    condition  roc_auc_ovo  0.741726      LogisticRegression   
1        valid    condition  roc_auc_ovo  0.655958      LogisticRegression   
2         test    condition  roc_auc_ovo  0.572573      LogisticRegression   
0         test    condition     accuracy  0.700000  RandomForestClassifier   
1         test    condition     accuracy  0.740000  RandomForestClassifier   
..         ...          ...          ...       ...                     ...   
5        train    condition     accuracy  1.000000  RandomForestClassifier   
6        train    condition     accuracy  1.000000  RandomForestClassifier   
7        train    condition     accuracy  1.000000  RandomForestClassifier
... [output truncated]
```

Results can be visualized as bar plots:

In \[7\]:

Copied!

```
fig_condition_eval_linear = stackix.visualizer.show_evaluation(
	param=my_target_param, # The parameter to visualize
	metric="accuracy", # The metric we specified
	ml_alg=str(rf) # The string representation of the classifier used
)
```

fig_condition_eval_linear = stackix.visualizer.show_evaluation( param=my_target_param, # The parameter to visualize metric="accuracy", # The metric we specified ml_alg=str(rf) # The string representation of the classifier used )

## Using evaluate with non-sklearn models[¶](#using-evaluate-with-non-sklearn-models)

We can use other machine learning models which are not from sklearn, but have the same interface. An recent and powerful example is the foundation model TabPFN <https://github.com/PriorLabs/TabPFN>

TabPFN is not installed together with AUTOENCODIX. To run this section, install it first (we tested version 2.1.3):

```
pip install tabpfn==2.1.3
```

Note: the example needs a GPU with more than 4 GB VRAM and is skipped otherwise.

In \[8\]:

Copied!

```
## SCIPY_ARRAY_API=1 is required to import TabPFN. It has to be set before scipy or
## scikit-learn are imported, which is why we already set it in the first cell of this notebook.
import os
print(os.environ.get("SCIPY_ARRAY_API"))
```

## SCIPY_ARRAY_API=1 is required to import TabPFN. It has to be set before scipy or

## scikit-learn are imported, which is why we already set it in the first cell of this notebook.

import os print(os.environ.get("SCIPY_ARRAY_API"))

```
1
```

In \[9\]:

Copied!

```
import torch

# TabPFN needs a GPU with more than 4 GB VRAM
use_tabpfn = (
	torch.cuda.is_available()
	and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory > 4 * 1024**3
)

if use_tabpfn:
	from tabpfn import TabPFNClassifier, TabPFNRegressor

	tabpfn_class = TabPFNClassifier()

	result = stackix.evaluate(
		params=[my_target_param],
		ml_model_class=tabpfn_class,
		metric_class=metric_class,
		split_type=split_type,
		reference_methods=reference_methods
	)
else:
	print("Get a GPU with >4GB VRAM to run TabPFN models, you poor soul.")
```

import torch

# TabPFN needs a GPU with more than 4 GB VRAM

use_tabpfn = ( torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory > 4 * 1024\*\*3 ) if use_tabpfn: from tabpfn import TabPFNClassifier, TabPFNRegressor tabpfn_class = TabPFNClassifier() result = stackix.evaluate( params=[my_target_param], ml_model_class=tabpfn_class, metric_class=metric_class, split_type=split_type, reference_methods=reference_methods ) else: print("Get a GPU with >4GB VRAM to run TabPFN models, you poor soul.")

```
Perform ML task with feature df: RandomFeature
RandomFeature_R1
Perform ML task for target parameter: condition
```

```
RandomFeature_R2
Perform ML task for target parameter: condition
```

```
RandomFeature_R3
Perform ML task for target parameter: condition
```

```
RandomFeature_R4
Perform ML task for target parameter: condition
```

```
RandomFeature_R5
Perform ML task for target parameter: condition
```

```
Perform ML task with feature df: PCA
PCA
Perform ML task for target parameter: condition
```

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
```

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
```

In \[10\]:

Copied!

```
if use_tabpfn:
	fig_condition_eval_tabpfn = stackix.visualizer.show_evaluation(
		param=my_target_param, # The parameter to visualize
		metric="accuracy", # The metric we specified
		ml_alg=str(tabpfn_class) # The string representation of the classifier used
	)
```

if use_tabpfn: fig_condition_eval_tabpfn = stackix.visualizer.show_evaluation( param=my_target_param, # The parameter to visualize metric="accuracy", # The metric we specified ml_alg=str(tabpfn_class) # The string representation of the classifier used )

# XModalix specifics[¶](#xmodalix-specifics)

## Embedding evaluation[¶](#embedding-evaluation)

Since XModalix is composed of multiple VAE for each modality, evaluate behaves slightly different. In the case embedding evaluation it will perform evaluation for each VAE (modality) and test its performance for the specified downstream task.

## Pure VAE comparison[¶](#pure-vae-comparison)

The function `xmodalix.evaluator.pure_vae_comparison()` is designed to evaluate the reconstruction capability of a XModalix (reference and translated) against a normal Imagix on the image target modality. This evaluation provides insights whether the enforced latent space alignment and coupled training in the XModalix decreases or improves the reconstruction capability in comparison to image-only autoencoder.

The prerequisite for this comparison is the prior training of an imagix. An example is provided in the [XModalix tutorial](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/PipelineTutorials/XModalix.ipynb).

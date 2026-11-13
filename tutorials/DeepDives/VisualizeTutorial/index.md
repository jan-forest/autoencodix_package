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

# Overview of visualization options[¶](#overview-of-visualization-options)

This tutorial gives a comprehensive overview of in-built visualization functions and options.

Content:

1. Model & training
   - Model weight visualization
   - Loss curves
1. Latent space
   - Heatmap representation
   - Ridgeline representation
   - 2D representation
   - Coverage and total correlation
1. Embedding evaluation
   - Performance comparison via bar plots
1. Saving and customization
1. XModalix and Imagix specialities
   - Differences in standard visualizations
   - Input vs. translation 2D check
   - Visualizing image translation capabilities

## 1. Model & training[¶](#1-model-training)

We provide a heatmap representation of model weights and loss curves over epochs to gain insights into the autoencoder training and debugging.

Model weight visualization is limited to small models (input features \<3000) for computational and visual reasons.

Let us start with a Disentanglix to show model weight and loss curve plotting.

In \[2\]:

Copied!

```
from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK
from autoencodix.configs.disentanglix_config import DisentanglixConfig
from autoencodix.configs.default_config import DataCase
import autoencodix as acx


my_cfg = DisentanglixConfig(
	data_case=DataCase.MULTI_BULK,
	loss_reduction="sum",
	k_filter=80,
	latent_dim=6,
	scaling="STANDARD",
	epochs=100,
	learning_rate=0.001,
	batch_size = 128,
	beta_mi = 10,
	beta_tc = 500,
	beta_dimKL= 0.5,
	use_mss = True,
	drop_p =0.1,
	global_seed=42,
	checkpoint_interval=1,
	n_layers=1)

disent = acx.Disentanglix(data=EXAMPLE_MULTI_BULK, config=my_cfg)

result = disent.run()
```

from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK from autoencodix.configs.disentanglix_config import DisentanglixConfig from autoencodix.configs.default_config import DataCase import autoencodix as acx my_cfg = DisentanglixConfig( data_case=DataCase.MULTI_BULK, loss_reduction="sum", k_filter=80, latent_dim=6, scaling="STANDARD", epochs=100, learning_rate=0.001, batch_size = 128, beta_mi = 10, beta_tc = 500, beta_dimKL= 0.5, use_mss = True, drop_p =0.1, global_seed=42, checkpoint_interval=1, n_layers=1) disent = acx.Disentanglix(data=EXAMPLE_MULTI_BULK, config=my_cfg) result = disent.run()

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: transcriptomics
anno key: proteomics
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 31.6446 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 1 - Train Loss: 97.2630
Sub-losses: recon_loss: 97.2621, mut_info_loss: 0.0004, tot_corr_loss: 0.0005, dimwise_kl_loss: 0.0001, anneal_factor: 0.0000, effective_beta_mi_factor: 0.0005, effective_beta_tc_factor: 0.0227, effective_beta_dimKL_factor: 0.0000
Epoch 1 - Valid Loss: 89.8245
Sub-losses: recon_loss: 89.8230, mut_info_loss: 0.0003, tot_corr_loss: 0.0012, dimwise_kl_loss: 0.0000, anneal_factor: 0.0000, effective_beta_mi_factor: 0.0005, effective_beta_tc_factor: 0.0227, effective_beta_dimKL_factor: 0.0000
Epoch 2 - Train Loss: 95.9207
Sub-losses: recon_loss: 95.9201, mut_info_loss: 0.0004, tot_corr_loss: 0.0002, dimwise_kl_loss: 0.0001, anneal_factor: 0.0001, effective_beta_mi_factor: 0.0006, effective_beta_tc_factor: 0.0277, effective_beta_dimKL_factor: 0.0000
Epoch 2 - Valid
... [output truncated]
```

```
Epoch 4 - Train Loss: 91.8892
Sub-losses: recon_loss: 91.8866, mut_info_loss: 0.0006, tot_corr_loss: 0.0018, dimwise_kl_loss: 0.0001, anneal_factor: 0.0001, effective_beta_mi_factor: 0.0008, effective_beta_tc_factor: 0.0414, effective_beta_dimKL_factor: 0.0000
Epoch 4 - Valid Loss: 85.0144
Sub-losses: recon_loss: 85.0138, mut_info_loss: 0.0004, tot_corr_loss: 0.0000, dimwise_kl_loss: 0.0001, anneal_factor: 0.0001, effective_beta_mi_factor: 0.0008, effective_beta_tc_factor: 0.0414, effective_beta_dimKL_factor: 0.0000
Epoch 5 - Train Loss: 91.2830
Sub-losses: recon_loss: 91.2797, mut_info_loss: 0.0007, tot_corr_loss: 0.0024, dimwise_kl_loss: 0.0001, anneal_factor: 0.0001, effective_beta_mi_factor: 0.0010, effective_beta_tc_factor: 0.0505, effective_beta_dimKL_factor: 0.0001
Epoch 5 - Valid
... [output truncated]
```

```
Epoch 8 - Train Loss: 87.7770
Sub-losses: recon_loss: 87.7718, mut_info_loss: 0.0016, tot_corr_loss: 0.0033, dimwise_kl_loss: 0.0003, anneal_factor: 0.0002, effective_beta_mi_factor: 0.0018, effective_beta_tc_factor: 0.0920, effective_beta_dimKL_factor: 0.0001
Epoch 8 - Valid Loss: 80.1388
Sub-losses: recon_loss: 80.1360, mut_info_loss: 0.0006, tot_corr_loss: 0.0020, dimwise_kl_loss: 0.0002, anneal_factor: 0.0002, effective_beta_mi_factor: 0.0018, effective_beta_tc_factor: 0.0920, effective_beta_dimKL_factor: 0.0001
Epoch 9 - Train Loss: 87.6653
Sub-losses: recon_loss: 87.6577, mut_info_loss: 0.0018, tot_corr_loss: 0.0054, dimwise_kl_loss: 0.0004, anneal_factor: 0.0002, effective_beta_mi_factor: 0.0022, effective_beta_tc_factor: 0.1124, effective_beta_dimKL_factor: 0.0001
Epoch 9 - Valid
... [output truncated]
```

```
Epoch 10 - Train Loss: 85.9561
Sub-losses: recon_loss: 85.9481, mut_info_loss: 0.0021, tot_corr_loss: 0.0055, dimwise_kl_loss: 0.0004, anneal_factor: 0.0003, effective_beta_mi_factor: 0.0027, effective_beta_tc_factor: 0.1373, effective_beta_dimKL_factor: 0.0001
Epoch 10 - Valid Loss: 80.7765
Sub-losses: recon_loss: 80.7726, mut_info_loss: 0.0020, tot_corr_loss: 0.0016, dimwise_kl_loss: 0.0004, anneal_factor: 0.0003, effective_beta_mi_factor: 0.0027, effective_beta_tc_factor: 0.1373, effective_beta_dimKL_factor: 0.0001
Epoch 11 - Train Loss: 85.2333
Sub-losses: recon_loss: 85.2196, mut_info_loss: 0.0028, tot_corr_loss: 0.0103, dimwise_kl_loss: 0.0006, anneal_factor: 0.0003, effective_beta_mi_factor: 0.0034, effective_beta_tc_factor: 0.1677, effective_beta_dimKL_factor: 0.0002
Epoch 11 - Va
... [output truncated]
```

```
Epoch 13 - Train Loss: 84.2506
Sub-losses: recon_loss: 84.2208, mut_info_loss: 0.0048, tot_corr_loss: 0.0242, dimwise_kl_loss: 0.0008, anneal_factor: 0.0005, effective_beta_mi_factor: 0.0050, effective_beta_tc_factor: 0.2501, effective_beta_dimKL_factor: 0.0003
Epoch 13 - Valid Loss: 78.0658
Sub-losses: recon_loss: 78.0437, mut_info_loss: 0.0040, tot_corr_loss: 0.0173, dimwise_kl_loss: 0.0008, anneal_factor: 0.0005, effective_beta_mi_factor: 0.0050, effective_beta_tc_factor: 0.2501, effective_beta_dimKL_factor: 0.0003
Epoch 14 - Train Loss: 83.1435
Sub-losses: recon_loss: 83.1138, mut_info_loss: 0.0062, tot_corr_loss: 0.0225, dimwise_kl_loss: 0.0011, anneal_factor: 0.0006, effective_beta_mi_factor: 0.0061, effective_beta_tc_factor: 0.3054, effective_beta_dimKL_factor: 0.0003
Epoch 14 - Va
... [output truncated]
```

```
Epoch 16 - Train Loss: 83.0944
Sub-losses: recon_loss: 83.0580, mut_info_loss: 0.0084, tot_corr_loss: 0.0264, dimwise_kl_loss: 0.0015, anneal_factor: 0.0009, effective_beta_mi_factor: 0.0091, effective_beta_tc_factor: 0.4555, effective_beta_dimKL_factor: 0.0005
Epoch 16 - Valid Loss: 78.1649
Sub-losses: recon_loss: 78.1158, mut_info_loss: 0.0059, tot_corr_loss: 0.0411, dimwise_kl_loss: 0.0022, anneal_factor: 0.0009, effective_beta_mi_factor: 0.0091, effective_beta_tc_factor: 0.4555, effective_beta_dimKL_factor: 0.0005
Epoch 17 - Train Loss: 80.9695
Sub-losses: recon_loss: 80.8874, mut_info_loss: 0.0106, tot_corr_loss: 0.0697, dimwise_kl_loss: 0.0019, anneal_factor: 0.0011, effective_beta_mi_factor: 0.0111, effective_beta_tc_factor: 0.5563, effective_beta_dimKL_factor: 0.0006
Epoch 17 - Va
... [output truncated]
```

```
Epoch 19 - Train Loss: 79.6351
Sub-losses: recon_loss: 79.5224, mut_info_loss: 0.0168, tot_corr_loss: 0.0930, dimwise_kl_loss: 0.0029, anneal_factor: 0.0017, effective_beta_mi_factor: 0.0166, effective_beta_tc_factor: 0.8294, effective_beta_dimKL_factor: 0.0008
Epoch 19 - Valid Loss: 73.1812
Sub-losses: recon_loss: 73.0716, mut_info_loss: 0.0176, tot_corr_loss: 0.0887, dimwise_kl_loss: 0.0033, anneal_factor: 0.0017, effective_beta_mi_factor: 0.0166, effective_beta_tc_factor: 0.8294, effective_beta_dimKL_factor: 0.0008
Epoch 20 - Train Loss: 78.7569
Sub-losses: recon_loss: 78.5630, mut_info_loss: 0.0254, tot_corr_loss: 0.1641, dimwise_kl_loss: 0.0044, anneal_factor: 0.0020, effective_beta_mi_factor: 0.0203, effective_beta_tc_factor: 1.0127, effective_beta_dimKL_factor: 0.0010
Epoch 20 - Va
... [output truncated]
```

```
Epoch 21 - Train Loss: 78.4371
Sub-losses: recon_loss: 78.2043, mut_info_loss: 0.0298, tot_corr_loss: 0.1980, dimwise_kl_loss: 0.0049, anneal_factor: 0.0025, effective_beta_mi_factor: 0.0247, effective_beta_tc_factor: 1.2363, effective_beta_dimKL_factor: 0.0012
Epoch 21 - Valid Loss: 71.7727
Sub-losses: recon_loss: 71.5929, mut_info_loss: 0.0279, tot_corr_loss: 0.1473, dimwise_kl_loss: 0.0046, anneal_factor: 0.0025, effective_beta_mi_factor: 0.0247, effective_beta_tc_factor: 1.2363, effective_beta_dimKL_factor: 0.0012
Epoch 22 - Train Loss: 77.8029
Sub-losses: recon_loss: 77.5365, mut_info_loss: 0.0333, tot_corr_loss: 0.2268, dimwise_kl_loss: 0.0064, anneal_factor: 0.0030, effective_beta_mi_factor: 0.0302, effective_beta_tc_factor: 1.5092, effective_beta_dimKL_factor: 0.0015
Epoch 22 - Va
... [output truncated]
```

```
Epoch 23 - Train Loss: 76.8611
Sub-losses: recon_loss: 76.4996, mut_info_loss: 0.0446, tot_corr_loss: 0.3091, dimwise_kl_loss: 0.0078, anneal_factor: 0.0037, effective_beta_mi_factor: 0.0368, effective_beta_tc_factor: 1.8421, effective_beta_dimKL_factor: 0.0018
Epoch 23 - Valid Loss: 70.3202
Sub-losses: recon_loss: 69.8966, mut_info_loss: 0.0426, tot_corr_loss: 0.3740, dimwise_kl_loss: 0.0071, anneal_factor: 0.0037, effective_beta_mi_factor: 0.0368, effective_beta_tc_factor: 1.8421, effective_beta_dimKL_factor: 0.0018
Epoch 24 - Train Loss: 76.5867
Sub-losses: recon_loss: 76.0853, mut_info_loss: 0.0566, tot_corr_loss: 0.4345, dimwise_kl_loss: 0.0102, anneal_factor: 0.0045, effective_beta_mi_factor: 0.0450, effective_beta_tc_factor: 2.2481, effective_beta_dimKL_factor: 0.0022
Epoch 24 - Va
... [output truncated]
```

```
Epoch 26 - Train Loss: 75.8177
Sub-losses: recon_loss: 75.0388, mut_info_loss: 0.0862, tot_corr_loss: 0.6777, dimwise_kl_loss: 0.0150, anneal_factor: 0.0067, effective_beta_mi_factor: 0.0669, effective_beta_tc_factor: 3.3464, effective_beta_dimKL_factor: 0.0033
Epoch 26 - Valid Loss: 69.2326
Sub-losses: recon_loss: 68.1728, mut_info_loss: 0.0923, tot_corr_loss: 0.9521, dimwise_kl_loss: 0.0154, anneal_factor: 0.0067, effective_beta_mi_factor: 0.0669, effective_beta_tc_factor: 3.3464, effective_beta_dimKL_factor: 0.0033
Epoch 27 - Train Loss: 75.3926
Sub-losses: recon_loss: 74.3642, mut_info_loss: 0.1068, tot_corr_loss: 0.9031, dimwise_kl_loss: 0.0186, anneal_factor: 0.0082, effective_beta_mi_factor: 0.0816, effective_beta_tc_factor: 4.0813, effective_beta_dimKL_factor: 0.0041
Epoch 27 - Va
... [output truncated]
```

```
Epoch 29 - Train Loss: 74.6660
Sub-losses: recon_loss: 72.9704, mut_info_loss: 0.1657, tot_corr_loss: 1.4977, dimwise_kl_loss: 0.0322, anneal_factor: 0.0121, effective_beta_mi_factor: 0.1213, effective_beta_tc_factor: 6.0642, effective_beta_dimKL_factor: 0.0061
Epoch 29 - Valid Loss: 70.7034
Sub-losses: recon_loss: 68.6773, mut_info_loss: 0.1456, tot_corr_loss: 1.8532, dimwise_kl_loss: 0.0274, anneal_factor: 0.0121, effective_beta_mi_factor: 0.1213, effective_beta_tc_factor: 6.0642, effective_beta_dimKL_factor: 0.0061
Epoch 30 - Train Loss: 74.5555
Sub-losses: recon_loss: 72.2716, mut_info_loss: 0.2107, tot_corr_loss: 2.0349, dimwise_kl_loss: 0.0383, anneal_factor: 0.0148, effective_beta_mi_factor: 0.1477, effective_beta_tc_factor: 7.3870, effective_beta_dimKL_factor: 0.0074
Epoch 30 - Va
... [output truncated]
```

```
Epoch 31 - Valid Loss: 68.2373
Sub-losses: recon_loss: 66.2576, mut_info_loss: 0.2616, tot_corr_loss: 1.6620, dimwise_kl_loss: 0.0561, anneal_factor: 0.0180, effective_beta_mi_factor: 0.1799, effective_beta_tc_factor: 8.9931, effective_beta_dimKL_factor: 0.0090
Epoch 32 - Train Loss: 74.3804
Sub-losses: recon_loss: 71.3400, mut_info_loss: 0.3314, tot_corr_loss: 2.6480, dimwise_kl_loss: 0.0608, anneal_factor: 0.0219, effective_beta_mi_factor: 0.2188, effective_beta_tc_factor: 10.9406, effective_beta_dimKL_factor: 0.0109
Epoch 32 - Valid Loss: 69.2479
Sub-losses: recon_loss: 65.5636, mut_info_loss: 0.3644, tot_corr_loss: 3.2705, dimwise_kl_loss: 0.0494, anneal_factor: 0.0219, effective_beta_mi_factor: 0.2188, effective_beta_tc_factor: 10.9406, effective_beta_dimKL_factor: 0.0109
```

```
Epoch 33 - Train Loss: 74.0530
Sub-losses: recon_loss: 71.1994, mut_info_loss: 0.3935, tot_corr_loss: 2.3851, dimwise_kl_loss: 0.0751, anneal_factor: 0.0266, effective_beta_mi_factor: 0.2660, effective_beta_tc_factor: 13.2985, effective_beta_dimKL_factor: 0.0133
Epoch 33 - Valid Loss: 68.4378
Sub-losses: recon_loss: 65.2790, mut_info_loss: 0.3775, tot_corr_loss: 2.7034, dimwise_kl_loss: 0.0779, anneal_factor: 0.0266, effective_beta_mi_factor: 0.2660, effective_beta_tc_factor: 13.2985, effective_beta_dimKL_factor: 0.0133
Epoch 34 - Train Loss: 74.3519
Sub-losses: recon_loss: 70.2587, mut_info_loss: 0.5084, tot_corr_loss: 3.4889, dimwise_kl_loss: 0.0959, anneal_factor: 0.0323, effective_beta_mi_factor: 0.3230, effective_beta_tc_factor: 16.1477, effective_beta_dimKL_factor: 0.0161
Epoch 34 -
... [output truncated]
```

```
Epoch 36 - Train Loss: 74.8447
Sub-losses: recon_loss: 69.8170, mut_info_loss: 0.7336, tot_corr_loss: 4.1642, dimwise_kl_loss: 0.1298, anneal_factor: 0.0474, effective_beta_mi_factor: 0.4743, effective_beta_tc_factor: 23.7129, effective_beta_dimKL_factor: 0.0237
Epoch 36 - Valid Loss: 67.0970
Sub-losses: recon_loss: 64.4941, mut_info_loss: 0.7253, tot_corr_loss: 1.7296, dimwise_kl_loss: 0.1480, anneal_factor: 0.0474, effective_beta_mi_factor: 0.4743, effective_beta_tc_factor: 23.7129, effective_beta_dimKL_factor: 0.0237
Epoch 37 - Train Loss: 74.4632
Sub-losses: recon_loss: 69.3246, mut_info_loss: 0.9075, tot_corr_loss: 4.0828, dimwise_kl_loss: 0.1484, anneal_factor: 0.0573, effective_beta_mi_factor: 0.5732, effective_beta_tc_factor: 28.6621, effective_beta_dimKL_factor: 0.0287
Epoch 37 -
... [output truncated]
```

```
Epoch 38 - Train Loss: 75.3557
Sub-losses: recon_loss: 69.7785, mut_info_loss: 1.0348, tot_corr_loss: 4.3535, dimwise_kl_loss: 0.1889, anneal_factor: 0.0691, effective_beta_mi_factor: 0.6914, effective_beta_tc_factor: 34.5692, effective_beta_dimKL_factor: 0.0346
Epoch 38 - Valid Loss: 71.1237
Sub-losses: recon_loss: 64.4789, mut_info_loss: 0.9456, tot_corr_loss: 5.5304, dimwise_kl_loss: 0.1687, anneal_factor: 0.0691, effective_beta_mi_factor: 0.6914, effective_beta_tc_factor: 34.5692, effective_beta_dimKL_factor: 0.0346
Epoch 39 - Train Loss: 74.9344
Sub-losses: recon_loss: 69.4592, mut_info_loss: 1.2372, tot_corr_loss: 4.0093, dimwise_kl_loss: 0.2288, anneal_factor: 0.0832, effective_beta_mi_factor: 0.8317, effective_beta_tc_factor: 41.5863, effective_beta_dimKL_factor: 0.0416
Epoch 39 -
... [output truncated]
```

```
Epoch 40 - Train Loss: 76.5879
Sub-losses: recon_loss: 70.0437, mut_info_loss: 1.3934, tot_corr_loss: 4.9002, dimwise_kl_loss: 0.2506, anneal_factor: 0.0998, effective_beta_mi_factor: 0.9975, effective_beta_tc_factor: 49.8752, effective_beta_dimKL_factor: 0.0499
Epoch 40 - Valid Loss: 74.3364
Sub-losses: recon_loss: 65.6392, mut_info_loss: 1.0731, tot_corr_loss: 7.3706, dimwise_kl_loss: 0.2537, anneal_factor: 0.0998, effective_beta_mi_factor: 0.9975, effective_beta_tc_factor: 49.8752, effective_beta_dimKL_factor: 0.0499
Epoch 41 - Train Loss: 75.2999
Sub-losses: recon_loss: 69.6112, mut_info_loss: 1.7859, tot_corr_loss: 3.5959, dimwise_kl_loss: 0.3069, anneal_factor: 0.1192, effective_beta_mi_factor: 1.1920, effective_beta_tc_factor: 59.6015, effective_beta_dimKL_factor: 0.0596
Epoch 41 -
... [output truncated]
```

```
Epoch 43 - Train Loss: 80.1728
Sub-losses: recon_loss: 69.9777, mut_info_loss: 2.5847, tot_corr_loss: 7.1957, dimwise_kl_loss: 0.4148, anneal_factor: 0.1680, effective_beta_mi_factor: 1.6798, effective_beta_tc_factor: 83.9908, effective_beta_dimKL_factor: 0.0840
Epoch 43 - Valid Loss: 82.7314
Sub-losses: recon_loss: 64.6755, mut_info_loss: 2.3854, tot_corr_loss: 15.3031, dimwise_kl_loss: 0.3674, anneal_factor: 0.1680, effective_beta_mi_factor: 1.6798, effective_beta_tc_factor: 83.9908, effective_beta_dimKL_factor: 0.0840
Epoch 44 - Train Loss: 78.2347
Sub-losses: recon_loss: 69.2527, mut_info_loss: 2.9230, tot_corr_loss: 5.5478, dimwise_kl_loss: 0.5112, anneal_factor: 0.1978, effective_beta_mi_factor: 1.9782, effective_beta_tc_factor: 98.9081, effective_beta_dimKL_factor: 0.0989
Epoch 44
... [output truncated]
```

```
Epoch 45 - Valid Loss: 73.1917
Sub-losses: recon_loss: 66.7859, mut_info_loss: 2.6890, tot_corr_loss: 3.1904, dimwise_kl_loss: 0.5264, anneal_factor: 0.2315, effective_beta_mi_factor: 2.3148, effective_beta_tc_factor: 115.7376, effective_beta_dimKL_factor: 0.1157
Epoch 46 - Train Loss: 81.9470
Sub-losses: recon_loss: 69.8083, mut_info_loss: 4.1858, tot_corr_loss: 7.2694, dimwise_kl_loss: 0.6835, anneal_factor: 0.2689, effective_beta_mi_factor: 2.6894, effective_beta_tc_factor: 134.4707, effective_beta_dimKL_factor: 0.1345
Epoch 46 - Valid Loss: 88.1466
Sub-losses: recon_loss: 63.2516, mut_info_loss: 3.3779, tot_corr_loss: 20.9333, dimwise_kl_loss: 0.5838, anneal_factor: 0.2689, effective_beta_mi_factor: 2.6894, effective_beta_tc_factor: 134.4707, effective_beta_dimKL_factor: 0.1345
Epoch
... [output truncated]
```

```
Epoch 48 - Train Loss: 86.0678
Sub-losses: recon_loss: 69.9718, mut_info_loss: 4.6030, tot_corr_loss: 10.6224, dimwise_kl_loss: 0.8706, anneal_factor: 0.3543, effective_beta_mi_factor: 3.5434, effective_beta_tc_factor: 177.1718, effective_beta_dimKL_factor: 0.1772
Epoch 48 - Valid Loss: 91.3572
Sub-losses: recon_loss: 66.7590, mut_info_loss: 4.0452, tot_corr_loss: 19.6441, dimwise_kl_loss: 0.9088, anneal_factor: 0.3543, effective_beta_mi_factor: 3.5434, effective_beta_tc_factor: 177.1718, effective_beta_dimKL_factor: 0.1772
Epoch 49 - Train Loss: 82.1495
Sub-losses: recon_loss: 70.1274, mut_info_loss: 5.3413, tot_corr_loss: 5.7300, dimwise_kl_loss: 0.9509, anneal_factor: 0.4013, effective_beta_mi_factor: 4.0131, effective_beta_tc_factor: 200.6562, effective_beta_dimKL_factor: 0.2007
Epoch
... [output truncated]
```

```
Epoch 50 - Train Loss: 83.8004
Sub-losses: recon_loss: 70.4932, mut_info_loss: 5.8680, tot_corr_loss: 6.3600, dimwise_kl_loss: 1.0793, anneal_factor: 0.4502, effective_beta_mi_factor: 4.5017, effective_beta_tc_factor: 225.0830, effective_beta_dimKL_factor: 0.2251
Epoch 50 - Valid Loss: 97.0654
Sub-losses: recon_loss: 64.5984, mut_info_loss: 6.8122, tot_corr_loss: 24.8334, dimwise_kl_loss: 0.8214, anneal_factor: 0.4502, effective_beta_mi_factor: 4.5017, effective_beta_tc_factor: 225.0830, effective_beta_dimKL_factor: 0.2251
Epoch 51 - Train Loss: 90.0042
Sub-losses: recon_loss: 70.5591, mut_info_loss: 7.0733, tot_corr_loss: 11.2063, dimwise_kl_loss: 1.1655, anneal_factor: 0.5000, effective_beta_mi_factor: 5.0000, effective_beta_tc_factor: 250.0000, effective_beta_dimKL_factor: 0.2500
Epoch
... [output truncated]
```

```
Epoch 52 - Train Loss: 93.5470
Sub-losses: recon_loss: 70.5511, mut_info_loss: 7.0240, tot_corr_loss: 14.8224, dimwise_kl_loss: 1.1495, anneal_factor: 0.5498, effective_beta_mi_factor: 5.4983, effective_beta_tc_factor: 274.9170, effective_beta_dimKL_factor: 0.2749
Epoch 52 - Valid Loss: 82.4857
Sub-losses: recon_loss: 65.6722, mut_info_loss: 6.4622, tot_corr_loss: 8.9595, dimwise_kl_loss: 1.3919, anneal_factor: 0.5498, effective_beta_mi_factor: 5.4983, effective_beta_tc_factor: 274.9170, effective_beta_dimKL_factor: 0.2749
Epoch 53 - Train Loss: 87.2136
Sub-losses: recon_loss: 70.7870, mut_info_loss: 7.8787, tot_corr_loss: 7.1709, dimwise_kl_loss: 1.3769, anneal_factor: 0.5987, effective_beta_mi_factor: 5.9869, effective_beta_tc_factor: 299.3438, effective_beta_dimKL_factor: 0.2993
Epoch
... [output truncated]
```

```
Epoch 54 - Train Loss: 87.0905
Sub-losses: recon_loss: 70.9596, mut_info_loss: 7.9307, tot_corr_loss: 6.8607, dimwise_kl_loss: 1.3396, anneal_factor: 0.6457, effective_beta_mi_factor: 6.4566, effective_beta_tc_factor: 322.8282, effective_beta_dimKL_factor: 0.3228
Epoch 54 - Valid Loss: 84.2423
Sub-losses: recon_loss: 64.8320, mut_info_loss: 8.3383, tot_corr_loss: 9.5243, dimwise_kl_loss: 1.5477, anneal_factor: 0.6457, effective_beta_mi_factor: 6.4566, effective_beta_tc_factor: 322.8282, effective_beta_dimKL_factor: 0.3228
Epoch 55 - Train Loss: 96.8598
Sub-losses: recon_loss: 71.3424, mut_info_loss: 8.7232, tot_corr_loss: 15.3915, dimwise_kl_loss: 1.4027, anneal_factor: 0.6900, effective_beta_mi_factor: 6.8997, effective_beta_tc_factor: 344.9872, effective_beta_dimKL_factor: 0.3450
Epoch
... [output truncated]
```

```
Epoch 56 - Train Loss: 90.2877
Sub-losses: recon_loss: 71.0023, mut_info_loss: 8.6498, tot_corr_loss: 9.1175, dimwise_kl_loss: 1.5181, anneal_factor: 0.7311, effective_beta_mi_factor: 7.3106, effective_beta_tc_factor: 365.5293, effective_beta_dimKL_factor: 0.3655
Epoch 56 - Valid Loss: 93.9985
Sub-losses: recon_loss: 66.9015, mut_info_loss: 8.2302, tot_corr_loss: 17.4525, dimwise_kl_loss: 1.4143, anneal_factor: 0.7311, effective_beta_mi_factor: 7.3106, effective_beta_tc_factor: 365.5293, effective_beta_dimKL_factor: 0.3655
Epoch 57 - Train Loss: 91.5067
Sub-losses: recon_loss: 70.6195, mut_info_loss: 9.5738, tot_corr_loss: 9.7692, dimwise_kl_loss: 1.5443, anneal_factor: 0.7685, effective_beta_mi_factor: 7.6852, effective_beta_tc_factor: 384.2624, effective_beta_dimKL_factor: 0.3843
Epoch
... [output truncated]
```

```
Epoch 58 - Train Loss: 103.0473
Sub-losses: recon_loss: 70.1937, mut_info_loss: 9.9474, tot_corr_loss: 21.4687, dimwise_kl_loss: 1.4375, anneal_factor: 0.8022, effective_beta_mi_factor: 8.0218, effective_beta_tc_factor: 401.0919, effective_beta_dimKL_factor: 0.4011
Epoch 58 - Valid Loss: 103.7288
Sub-losses: recon_loss: 66.4020, mut_info_loss: 7.4149, tot_corr_loss: 28.5762, dimwise_kl_loss: 1.3357, anneal_factor: 0.8022, effective_beta_mi_factor: 8.0218, effective_beta_tc_factor: 401.0919, effective_beta_dimKL_factor: 0.4011
Epoch 59 - Train Loss: 96.6896
Sub-losses: recon_loss: 70.4628, mut_info_loss: 9.6132, tot_corr_loss: 14.9872, dimwise_kl_loss: 1.6264, anneal_factor: 0.8320, effective_beta_mi_factor: 8.3202, effective_beta_tc_factor: 416.0092, effective_beta_dimKL_factor: 0.4160
Ep
... [output truncated]
```

```
Epoch 60 - Valid Loss: 86.6360
Sub-losses: recon_loss: 68.2904, mut_info_loss: 8.1885, tot_corr_loss: 8.8312, dimwise_kl_loss: 1.3259, anneal_factor: 0.8581, effective_beta_mi_factor: 8.5815, effective_beta_tc_factor: 429.0745, effective_beta_dimKL_factor: 0.4291
Epoch 61 - Train Loss: 97.6042
Sub-losses: recon_loss: 71.4607, mut_info_loss: 9.6424, tot_corr_loss: 14.9599, dimwise_kl_loss: 1.5412, anneal_factor: 0.8808, effective_beta_mi_factor: 8.8080, effective_beta_tc_factor: 440.3985, effective_beta_dimKL_factor: 0.4404
Epoch 61 - Valid Loss: 96.6897
Sub-losses: recon_loss: 65.2630, mut_info_loss: 10.8814, tot_corr_loss: 19.0992, dimwise_kl_loss: 1.4461, anneal_factor: 0.8808, effective_beta_mi_factor: 8.8080, effective_beta_tc_factor: 440.3985, effective_beta_dimKL_factor: 0.4404
Epoc
... [output truncated]
```

```
Epoch 64 - Train Loss: 93.2314
Sub-losses: recon_loss: 71.5963, mut_info_loss: 9.9739, tot_corr_loss: 10.1044, dimwise_kl_loss: 1.5568, anneal_factor: 0.9309, effective_beta_mi_factor: 9.3086, effective_beta_tc_factor: 465.4308, effective_beta_dimKL_factor: 0.4654
Epoch 64 - Valid Loss: 103.2011
Sub-losses: recon_loss: 66.4837, mut_info_loss: 9.5765, tot_corr_loss: 25.5565, dimwise_kl_loss: 1.5844, anneal_factor: 0.9309, effective_beta_mi_factor: 9.3086, effective_beta_tc_factor: 465.4308, effective_beta_dimKL_factor: 0.4654
Epoch 65 - Train Loss: 99.5502
Sub-losses: recon_loss: 71.2203, mut_info_loss: 10.2932, tot_corr_loss: 16.5170, dimwise_kl_loss: 1.5198, anneal_factor: 0.9427, effective_beta_mi_factor: 9.4268, effective_beta_tc_factor: 471.3379, effective_beta_dimKL_factor: 0.4713
Ep
... [output truncated]
```

```
Epoch 67 - Train Loss: 84.5203
Sub-losses: recon_loss: 71.8514, mut_info_loss: 9.8388, tot_corr_loss: 1.3172, dimwise_kl_loss: 1.5129, anneal_factor: 0.9608, effective_beta_mi_factor: 9.6083, effective_beta_tc_factor: 480.4171, effective_beta_dimKL_factor: 0.4804
Epoch 67 - Valid Loss: 85.8307
Sub-losses: recon_loss: 68.6088, mut_info_loss: 9.3761, tot_corr_loss: 6.5546, dimwise_kl_loss: 1.2911, anneal_factor: 0.9608, effective_beta_mi_factor: 9.6083, effective_beta_tc_factor: 480.4171, effective_beta_dimKL_factor: 0.4804
Epoch 68 - Train Loss: 87.0336
Sub-losses: recon_loss: 71.1648, mut_info_loss: 10.0797, tot_corr_loss: 4.4167, dimwise_kl_loss: 1.3725, anneal_factor: 0.9677, effective_beta_mi_factor: 9.6770, effective_beta_tc_factor: 483.8523, effective_beta_dimKL_factor: 0.4839
Epoch
... [output truncated]
```

```
Epoch 69 - Valid Loss: 75.9944
Sub-losses: recon_loss: 67.1105, mut_info_loss: 7.7932, tot_corr_loss: 0.0105, dimwise_kl_loss: 1.0802, anneal_factor: 0.9734, effective_beta_mi_factor: 9.7340, effective_beta_tc_factor: 486.7015, effective_beta_dimKL_factor: 0.4867
Epoch 70 - Train Loss: 86.3831
Sub-losses: recon_loss: 72.6830, mut_info_loss: 9.2155, tot_corr_loss: 3.1605, dimwise_kl_loss: 1.3241, anneal_factor: 0.9781, effective_beta_mi_factor: 9.7812, effective_beta_tc_factor: 489.0594, effective_beta_dimKL_factor: 0.4891
Epoch 70 - Valid Loss: 76.0902
Sub-losses: recon_loss: 69.3783, mut_info_loss: 5.7063, tot_corr_loss: 0.0000, dimwise_kl_loss: 1.0055, anneal_factor: 0.9781, effective_beta_mi_factor: 9.7812, effective_beta_tc_factor: 489.0594, effective_beta_dimKL_factor: 0.4891
Epoch 7
... [output truncated]
```

```
Epoch 71 - Valid Loss: 95.2337
Sub-losses: recon_loss: 67.4021, mut_info_loss: 8.1023, tot_corr_loss: 18.8244, dimwise_kl_loss: 0.9050, anneal_factor: 0.9820, effective_beta_mi_factor: 9.8201, effective_beta_tc_factor: 491.0069, effective_beta_dimKL_factor: 0.4910
Epoch 72 - Train Loss: 90.7506
Sub-losses: recon_loss: 72.4338, mut_info_loss: 9.4707, tot_corr_loss: 7.5724, dimwise_kl_loss: 1.2737, anneal_factor: 0.9852, effective_beta_mi_factor: 9.8523, effective_beta_tc_factor: 492.6130, effective_beta_dimKL_factor: 0.4926
Epoch 72 - Valid Loss: 84.3481
Sub-losses: recon_loss: 67.3124, mut_info_loss: 9.6243, tot_corr_loss: 6.2754, dimwise_kl_loss: 1.1360, anneal_factor: 0.9852, effective_beta_mi_factor: 9.8523, effective_beta_tc_factor: 492.6130, effective_beta_dimKL_factor: 0.4926
Epoch
... [output truncated]
```

```
Epoch 74 - Train Loss: 93.1610
Sub-losses: recon_loss: 72.8364, mut_info_loss: 9.1127, tot_corr_loss: 9.9368, dimwise_kl_loss: 1.2751, anneal_factor: 0.9900, effective_beta_mi_factor: 9.9005, effective_beta_tc_factor: 495.0241, effective_beta_dimKL_factor: 0.4950
Epoch 74 - Valid Loss: 79.6376
Sub-losses: recon_loss: 67.6724, mut_info_loss: 8.7075, tot_corr_loss: 2.2205, dimwise_kl_loss: 1.0372, anneal_factor: 0.9900, effective_beta_mi_factor: 9.9005, effective_beta_tc_factor: 495.0241, effective_beta_dimKL_factor: 0.4950
Epoch 75 - Train Loss: 82.4008
Sub-losses: recon_loss: 72.5079, mut_info_loss: 8.3105, tot_corr_loss: 0.4214, dimwise_kl_loss: 1.1610, anneal_factor: 0.9918, effective_beta_mi_factor: 9.9184, effective_beta_tc_factor: 495.9187, effective_beta_dimKL_factor: 0.4959
Epoch 7
... [output truncated]
```

```
Epoch 76 - Train Loss: 87.9451
Sub-losses: recon_loss: 72.5374, mut_info_loss: 9.4792, tot_corr_loss: 4.7286, dimwise_kl_loss: 1.2000, anneal_factor: 0.9933, effective_beta_mi_factor: 9.9331, effective_beta_tc_factor: 496.6536, effective_beta_dimKL_factor: 0.4967
Epoch 76 - Valid Loss: 78.9826
Sub-losses: recon_loss: 70.3897, mut_info_loss: 7.3071, tot_corr_loss: 0.0000, dimwise_kl_loss: 1.2858, anneal_factor: 0.9933, effective_beta_mi_factor: 9.9331, effective_beta_tc_factor: 496.6536, effective_beta_dimKL_factor: 0.4967
Epoch 77 - Train Loss: 89.5216
Sub-losses: recon_loss: 72.2952, mut_info_loss: 9.5595, tot_corr_loss: 6.5786, dimwise_kl_loss: 1.0884, anneal_factor: 0.9945, effective_beta_mi_factor: 9.9451, effective_beta_tc_factor: 497.2568, effective_beta_dimKL_factor: 0.4973
Epoch 7
... [output truncated]
```

```
Epoch 78 - Train Loss: 82.2348
Sub-losses: recon_loss: 72.0957, mut_info_loss: 9.0222, tot_corr_loss: 0.0000, dimwise_kl_loss: 1.1168, anneal_factor: 0.9955, effective_beta_mi_factor: 9.9550, effective_beta_tc_factor: 497.7519, effective_beta_dimKL_factor: 0.4978
Epoch 78 - Valid Loss: 87.1744
Sub-losses: recon_loss: 69.7229, mut_info_loss: 8.1062, tot_corr_loss: 8.5022, dimwise_kl_loss: 0.8431, anneal_factor: 0.9955, effective_beta_mi_factor: 9.9550, effective_beta_tc_factor: 497.7519, effective_beta_dimKL_factor: 0.4978
Epoch 79 - Train Loss: 89.9588
Sub-losses: recon_loss: 72.2015, mut_info_loss: 9.2950, tot_corr_loss: 7.4510, dimwise_kl_loss: 1.0113, anneal_factor: 0.9963, effective_beta_mi_factor: 9.9632, effective_beta_tc_factor: 498.1579, effective_beta_dimKL_factor: 0.4982
Epoch 7
... [output truncated]
```

```
Epoch 80 - Train Loss: 91.6172
Sub-losses: recon_loss: 72.3698, mut_info_loss: 8.9747, tot_corr_loss: 9.2352, dimwise_kl_loss: 1.0375, anneal_factor: 0.9970, effective_beta_mi_factor: 9.9698, effective_beta_tc_factor: 498.4908, effective_beta_dimKL_factor: 0.4985
Epoch 80 - Valid Loss: 81.8506
Sub-losses: recon_loss: 68.4658, mut_info_loss: 6.3162, tot_corr_loss: 6.2711, dimwise_kl_loss: 0.7975, anneal_factor: 0.9970, effective_beta_mi_factor: 9.9698, effective_beta_tc_factor: 498.4908, effective_beta_dimKL_factor: 0.4985
Epoch 81 - Train Loss: 86.9672
Sub-losses: recon_loss: 72.9607, mut_info_loss: 8.6346, tot_corr_loss: 4.3912, dimwise_kl_loss: 0.9807, anneal_factor: 0.9975, effective_beta_mi_factor: 9.9753, effective_beta_tc_factor: 498.7637, effective_beta_dimKL_factor: 0.4988
Epoch 8
... [output truncated]
```

```
Epoch 82 - Valid Loss: 78.6833
Sub-losses: recon_loss: 70.4530, mut_info_loss: 7.4035, tot_corr_loss: 0.0000, dimwise_kl_loss: 0.8268, anneal_factor: 0.9980, effective_beta_mi_factor: 9.9797, effective_beta_tc_factor: 498.9873, effective_beta_dimKL_factor: 0.4990
Epoch 83 - Train Loss: 82.9687
Sub-losses: recon_loss: 72.8425, mut_info_loss: 8.2981, tot_corr_loss: 0.9045, dimwise_kl_loss: 0.9237, anneal_factor: 0.9983, effective_beta_mi_factor: 9.9834, effective_beta_tc_factor: 499.1706, effective_beta_dimKL_factor: 0.4992
Epoch 83 - Valid Loss: 83.8420
Sub-losses: recon_loss: 68.8101, mut_info_loss: 8.2259, tot_corr_loss: 6.1191, dimwise_kl_loss: 0.6869, anneal_factor: 0.9983, effective_beta_mi_factor: 9.9834, effective_beta_tc_factor: 499.1706, effective_beta_dimKL_factor: 0.4992
```

```
Epoch 84 - Train Loss: 87.2128
Sub-losses: recon_loss: 73.4307, mut_info_loss: 7.8590, tot_corr_loss: 5.0234, dimwise_kl_loss: 0.8998, anneal_factor: 0.9986, effective_beta_mi_factor: 9.9864, effective_beta_tc_factor: 499.3207, effective_beta_dimKL_factor: 0.4993
Epoch 84 - Valid Loss: 90.8062
Sub-losses: recon_loss: 68.0658, mut_info_loss: 8.3176, tot_corr_loss: 13.3451, dimwise_kl_loss: 1.0778, anneal_factor: 0.9986, effective_beta_mi_factor: 9.9864, effective_beta_tc_factor: 499.3207, effective_beta_dimKL_factor: 0.4993
Epoch 85 - Train Loss: 83.8772
Sub-losses: recon_loss: 73.0476, mut_info_loss: 7.5123, tot_corr_loss: 2.3248, dimwise_kl_loss: 0.9924, anneal_factor: 0.9989, effective_beta_mi_factor: 9.9889, effective_beta_tc_factor: 499.4437, effective_beta_dimKL_factor: 0.4994
Epoch
... [output truncated]
```

```
Epoch 86 - Train Loss: 83.2132
Sub-losses: recon_loss: 72.8768, mut_info_loss: 7.9314, tot_corr_loss: 1.5396, dimwise_kl_loss: 0.8654, anneal_factor: 0.9991, effective_beta_mi_factor: 9.9909, effective_beta_tc_factor: 499.5445, effective_beta_dimKL_factor: 0.4995
Epoch 86 - Valid Loss: 79.0142
Sub-losses: recon_loss: 68.6036, mut_info_loss: 7.7898, tot_corr_loss: 1.7784, dimwise_kl_loss: 0.8423, anneal_factor: 0.9991, effective_beta_mi_factor: 9.9909, effective_beta_tc_factor: 499.5445, effective_beta_dimKL_factor: 0.4995
Epoch 87 - Train Loss: 85.0278
Sub-losses: recon_loss: 73.7196, mut_info_loss: 7.7009, tot_corr_loss: 2.8127, dimwise_kl_loss: 0.7946, anneal_factor: 0.9993, effective_beta_mi_factor: 9.9925, effective_beta_tc_factor: 499.6270, effective_beta_dimKL_factor: 0.4996
Epoch 8
... [output truncated]
```

```
Epoch 88 - Train Loss: 83.4587
Sub-losses: recon_loss: 72.2079, mut_info_loss: 8.0611, tot_corr_loss: 2.2913, dimwise_kl_loss: 0.8984, anneal_factor: 0.9994, effective_beta_mi_factor: 9.9939, effective_beta_tc_factor: 499.6945, effective_beta_dimKL_factor: 0.4997
Epoch 88 - Valid Loss: 77.0886
Sub-losses: recon_loss: 70.1473, mut_info_loss: 6.1329, tot_corr_loss: 0.3012, dimwise_kl_loss: 0.5073, anneal_factor: 0.9994, effective_beta_mi_factor: 9.9939, effective_beta_tc_factor: 499.6945, effective_beta_dimKL_factor: 0.4997
Epoch 89 - Train Loss: 84.5285
Sub-losses: recon_loss: 71.6244, mut_info_loss: 8.3522, tot_corr_loss: 3.6434, dimwise_kl_loss: 0.9085, anneal_factor: 0.9995, effective_beta_mi_factor: 9.9950, effective_beta_tc_factor: 499.7499, effective_beta_dimKL_factor: 0.4997
Epoch 8
... [output truncated]
```

```
Epoch 90 - Train Loss: 87.3430
Sub-losses: recon_loss: 73.3738, mut_info_loss: 7.7194, tot_corr_loss: 5.3455, dimwise_kl_loss: 0.9043, anneal_factor: 0.9996, effective_beta_mi_factor: 9.9959, effective_beta_tc_factor: 499.7952, effective_beta_dimKL_factor: 0.4998
Epoch 90 - Valid Loss: 83.9068
Sub-losses: recon_loss: 67.4485, mut_info_loss: 7.6275, tot_corr_loss: 7.9016, dimwise_kl_loss: 0.9291, anneal_factor: 0.9996, effective_beta_mi_factor: 9.9959, effective_beta_tc_factor: 499.7952, effective_beta_dimKL_factor: 0.4998
Epoch 91 - Train Loss: 88.1063
Sub-losses: recon_loss: 72.0004, mut_info_loss: 8.4354, tot_corr_loss: 6.7294, dimwise_kl_loss: 0.9411, anneal_factor: 0.9997, effective_beta_mi_factor: 9.9966, effective_beta_tc_factor: 499.8323, effective_beta_dimKL_factor: 0.4998
Epoch 9
... [output truncated]
```

```
Epoch 92 - Train Loss: 84.8958
Sub-losses: recon_loss: 73.0536, mut_info_loss: 7.4892, tot_corr_loss: 3.6573, dimwise_kl_loss: 0.6957, anneal_factor: 0.9997, effective_beta_mi_factor: 9.9973, effective_beta_tc_factor: 499.8627, effective_beta_dimKL_factor: 0.4999
Epoch 92 - Valid Loss: 76.5450
Sub-losses: recon_loss: 68.7457, mut_info_loss: 7.4105, tot_corr_loss: 0.0000, dimwise_kl_loss: 0.3888, anneal_factor: 0.9997, effective_beta_mi_factor: 9.9973, effective_beta_tc_factor: 499.8627, effective_beta_dimKL_factor: 0.4999
Epoch 93 - Train Loss: 83.5068
Sub-losses: recon_loss: 72.8491, mut_info_loss: 8.1053, tot_corr_loss: 1.7707, dimwise_kl_loss: 0.7818, anneal_factor: 0.9998, effective_beta_mi_factor: 9.9978, effective_beta_tc_factor: 499.8876, effective_beta_dimKL_factor: 0.4999
Epoch 9
... [output truncated]
```

```
Epoch 94 - Train Loss: 85.0296
Sub-losses: recon_loss: 73.4607, mut_info_loss: 6.9923, tot_corr_loss: 3.8193, dimwise_kl_loss: 0.7572, anneal_factor: 0.9998, effective_beta_mi_factor: 9.9982, effective_beta_tc_factor: 499.9080, effective_beta_dimKL_factor: 0.4999
Epoch 94 - Valid Loss: 78.6786
Sub-losses: recon_loss: 68.6881, mut_info_loss: 7.1639, tot_corr_loss: 2.2091, dimwise_kl_loss: 0.6175, anneal_factor: 0.9998, effective_beta_mi_factor: 9.9982, effective_beta_tc_factor: 499.9080, effective_beta_dimKL_factor: 0.4999
Epoch 95 - Train Loss: 84.3195
Sub-losses: recon_loss: 72.5847, mut_info_loss: 7.9074, tot_corr_loss: 3.0866, dimwise_kl_loss: 0.7407, anneal_factor: 0.9998, effective_beta_mi_factor: 9.9985, effective_beta_tc_factor: 499.9247, effective_beta_dimKL_factor: 0.4999
Epoch 9
... [output truncated]
```

```
Epoch 96 - Train Loss: 88.4913
Sub-losses: recon_loss: 71.2444, mut_info_loss: 8.5593, tot_corr_loss: 8.0128, dimwise_kl_loss: 0.6748, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9988, effective_beta_tc_factor: 499.9383, effective_beta_dimKL_factor: 0.4999
Epoch 96 - Valid Loss: 83.3672
Sub-losses: recon_loss: 68.1217, mut_info_loss: 7.1219, tot_corr_loss: 7.3576, dimwise_kl_loss: 0.7660, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9988, effective_beta_tc_factor: 499.9383, effective_beta_dimKL_factor: 0.4999
Epoch 97 - Train Loss: 87.1056
Sub-losses: recon_loss: 71.9243, mut_info_loss: 8.3404, tot_corr_loss: 6.0125, dimwise_kl_loss: 0.8284, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9990, effective_beta_tc_factor: 499.9495, effective_beta_dimKL_factor: 0.4999
Epoch 9
... [output truncated]
```

```
Epoch 98 - Train Loss: 85.6697
Sub-losses: recon_loss: 71.3407, mut_info_loss: 7.6256, tot_corr_loss: 5.9911, dimwise_kl_loss: 0.7123, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9992, effective_beta_tc_factor: 499.9586, effective_beta_dimKL_factor: 0.5000
Epoch 98 - Valid Loss: 86.1850
Sub-losses: recon_loss: 65.6213, mut_info_loss: 8.4881, tot_corr_loss: 11.2176, dimwise_kl_loss: 0.8580, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9992, effective_beta_tc_factor: 499.9586, effective_beta_dimKL_factor: 0.5000
Epoch 99 - Train Loss: 86.7496
Sub-losses: recon_loss: 72.7149, mut_info_loss: 7.5741, tot_corr_loss: 5.7289, dimwise_kl_loss: 0.7318, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9993, effective_beta_tc_factor: 499.9661, effective_beta_dimKL_factor: 0.5000
Epoch
... [output truncated]
```

```
Epoch 100 - Train Loss: 82.2820
Sub-losses: recon_loss: 71.5371, mut_info_loss: 7.4815, tot_corr_loss: 2.5783, dimwise_kl_loss: 0.6850, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9994, effective_beta_tc_factor: 499.9723, effective_beta_dimKL_factor: 0.5000
Epoch 100 - Valid Loss: 80.3944
Sub-losses: recon_loss: 68.8209, mut_info_loss: 5.6982, tot_corr_loss: 5.2653, dimwise_kl_loss: 0.6099, anneal_factor: 0.9999, effective_beta_mi_factor: 9.9994, effective_beta_tc_factor: 499.9723, effective_beta_dimKL_factor: 0.5000
Processed 100 / 100 samples
```

After conducting the pipeline with `.run()`, the pipeline function `.visualize()` is already called and created the plots for model weights and loss curves.

We have now three ways to display them:

- Loss curves, but not model weights, are part of the general `show_result()` function.
- Specific display functions for loss `show_loss()` and model weights `show_weights()`
- accessing figure handles stored in `disent.visualizer.plots` dictionary

In \[3\]:

Copied!

```
# Using show_result
disent.show_result()
```

# Using show_result

disent.show_result()

```
Creating plots ...
```

In \[4\]:

Copied!

```
# Using specific display functions via disent.visualizer
fig_weights = disent.visualizer.show_weights()
fig_loss_abs = disent.visualizer.show_loss(plot_type="absolute") # Displays all loss components and factors over epochs
fig_loss_rel = disent.visualizer.show_loss(plot_type="relative") # Displays contribution of loss components to total loss over epochs
```

# Using specific display functions via disent.visualizer

fig_weights = disent.visualizer.show_weights() fig_loss_abs = disent.visualizer.show_loss(plot_type="absolute") # Displays all loss components and factors over epochs fig_loss_rel = disent.visualizer.show_loss(plot_type="relative") # Displays contribution of loss components to total loss over epochs

In \[5\]:

Copied!

```
# All figure handles are also accessible via disent.visualizer.plots dictionary
disent.visualizer.plots
```

# All figure handles are also accessible via disent.visualizer.plots dictionary

disent.visualizer.plots

Out\[5\]:

```
defaultdict(<function autoencodix.utils._utils.nested_dict()>,
            {'ModelWeights': <Figure size 3000x1500 with 8 Axes>,
             'loss_absolute': <Figure size 4500x500 with 9 Axes>,
             'loss_relative': <seaborn._core.plot.Plot at 0x7cd5410ddb80>,
             'Ridgeline': defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                         {99: defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                                      {'all': defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                                                   {'user_label': <seaborn.axisgrid.FacetGrid at 0x7cd5412c3d70>})})}),
             '2D-scatter': defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                         {99
... [output truncated]
```

## 2. Latent space visualization[¶](#2-latent-space-visualization)

We offer different views on the latent space for different tasks and insights which are all accessible via `show_latent_space()` and the options `plot_type=`:

- `2D-scatter` - Uses UMAP as dimension reduction if `latent_dim>2` to display a 2D scatter (samples) plot representation of the latent space
- `Ridgeline` - Distributions (per sample class) of latent intensities for each latent dimension
- `Clustermap`- Heatmap of meant latent intensities per sample class and latent dimension. Classes are clustered by correlation along latent dimensions.
- `Coverage-Correlation` - Displays average coverage and total correlation across latent dimensions to inspect denseness and disentanglement over epochs.

In \[6\]:

Copied!

```
disent._datasets.test.metadata
```

disent.\_datasets.test.metadata

Out\[6\]:

```
{'transcriptomics':               condition    batch  quality_score
 sample_1    condition_0  batch_2       0.884084
 sample_103  condition_2  batch_3       0.914741
 sample_107  condition_2  batch_2       0.969119
 sample_110  condition_1  batch_3       0.950450
 sample_111  condition_1  batch_2       0.678535
 ...                 ...      ...            ...
 sample_76   condition_0  batch_2       0.665877
 sample_81   condition_2  batch_3       0.957749
 sample_83   condition_1  batch_2       0.974214
 sample_91   condition_0  batch_1       0.872111
 sample_97   condition_0  batch_1       0.912413
 
 [100 rows x 3 columns],
 'proteomics':               condition    batch  quality_score
 sample_1    condition_0  batch_2       0.884084
 sample_103  condition_2  batch_3       0.914741
 sam
... [output truncated]
```

In \[7\]:

Copied!

```
params = ["condition"] # Specify metadata parameters to color/label the plots

fig_latent_2D = disent.visualizer.show_latent_space(result=result, plot_type="2D-scatter", param=params)
fig_latent_ridge = disent.visualizer.show_latent_space(result=result, plot_type="Ridgeline", param=params)
fig_latent_clustermap = disent.visualizer.show_latent_space(result=result, plot_type="Clustermap", param=params)
```

params = ["condition"] # Specify metadata parameters to color/label the plots fig_latent_2D = disent.visualizer.show_latent_space(result=result, plot_type="2D-scatter", param=params) fig_latent_ridge = disent.visualizer.show_latent_space(result=result, plot_type="Ridgeline", param=params) fig_latent_clustermap = disent.visualizer.show_latent_space(result=result, plot_type="Clustermap", param=params)

In \[8\]:

Copied!

```
# To increase resolution over epochs, set the config parameter 'checkpoint_interval=1' before training
fig_latent_cov_tc = disent.visualizer.show_latent_space(result=result, plot_type="Coverage-Correlation")
# TODO Coverage calculation fails for small valid split
```

# To increase resolution over epochs, set the config parameter 'checkpoint_interval=1' before training

fig_latent_cov_tc = disent.visualizer.show_latent_space(result=result, plot_type="Coverage-Correlation")

# TODO Coverage calculation fails for small valid split

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Covera
... [output truncated]
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Covera
... [output truncated]
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Covera
... [output truncated]
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Covera
... [output truncated]
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:946: UserWarning: Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins.
  warnings.warn(
```

### Advanced options in latent space visualization[¶](#advanced-options-in-latent-space-visualization)

For a deep analysis of the latent space we offer additional options in `show_latent_space()`:

- `param` - List of parameters to color plots and already provided as metadata before training. Strings must match column names. If not a list, string "all" is expected for convenient way to make plots for all parameters available.
- `labels` - List or pd.Series of labels of each sample to color the latent space. If provided as list this must match the order of samples.
- You can use instead the config parameter `annotation_columns` to specify a default list of parameters to be used for plotting
- `split` - The data split to visualize. Options are "train", "valid", "test", and "all". Default is "all".
- `epoch` - The epoch number to visualize. If None, the last epoch is used (default). Warning: this only works for epochs which are checkpointed, see config parameter `checkpoint_interval`.
- `n_downsample` - To speed up visualization, visualization is performed only on a random subset. Default is 10000.

Example: we want to display the latent space at the very beginning of the training only for samples in the `train` split.

Further, we have new labels which we will provide as a pd.Series.

In \[9\]:

Copied!

```
import pandas as pd
disease_labels = ["influenza"] * (len(disent._datasets.train.sample_ids)//2) + ["covid"] * (len(disent._datasets.train.sample_ids)//2) 
train_samples = pd.Series(data=disease_labels, index=disent._datasets.train.sample_ids, name="disease")

fig_latent_epoch0 = disent.visualizer.show_latent_space(
	result=result,
	plot_type="Ridgeline",
	labels=train_samples,
	split="train",
	epoch=0
)
```

import pandas as pd disease_labels = ["influenza"] * (len(disent.\_datasets.train.sample_ids)//2) + ["covid"] * (len(disent.\_datasets.train.sample_ids)//2) train_samples = pd.Series(data=disease_labels, index=disent.\_datasets.train.sample_ids, name="disease") fig_latent_epoch0 = disent.visualizer.show_latent_space( result=result, plot_type="Ridgeline", labels=train_samples, split="train", epoch=0 )

# 3. Embedding evaluation[¶](#3-embedding-evaluation)

Check out the Evaluate Tutorial for a deep dive into how to do the evaluation.

Here, is a quick example on how to plot the results after embedding evaluation.

In \[10\]:

Copied!

```
# Perform embedding evaluation (not part of .run())
params = ["condition", "batch"]
result = disent.evaluate(params=params)
```

# Perform embedding evaluation (not part of .run())

params = ["condition", "batch"] result = disent.evaluate(params=params)

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
Perform ML task for target parameter: batch
```

```
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
```

In \[11\]:

Copied!

```
from sklearn.linear_model import LogisticRegression

fig_condition_eval = disent.visualizer.show_evaluation(
	param=params[0], # We plot one parameter at a time
	metric="roc_auc_ovo", # This is the default metric for classification tasks
	ml_alg=str(LogisticRegression()) # This is the default ML algorithm for classification tasks
)
```

from sklearn.linear_model import LogisticRegression fig_condition_eval = disent.visualizer.show_evaluation( param=params[0], # We plot one parameter at a time metric="roc_auc_ovo", # This is the default metric for classification tasks ml_alg=str(LogisticRegression()) # This is the default ML algorithm for classification tasks )

# 4. Saving and customization[¶](#4-saving-and-customization)

## Saving[¶](#saving)

All plots in the dictionary in `disent.visualizer.plots` can be saved via the function `save_plots()`.

In case you only want to save a subset, you can use the option `which` and provide a list of plots to be saved.

In \[12\]:

Copied!

```
# All available plots
disent.visualizer.plots
```

# All available plots

disent.visualizer.plots

Out\[12\]:

```
defaultdict(<function autoencodix.utils._utils.nested_dict()>,
            {'ModelWeights': <Figure size 3000x1500 with 8 Axes>,
             'loss_absolute': <Figure size 4500x500 with 9 Axes>,
             'loss_relative': <seaborn._core.plot.Plot at 0x7cd5410ddb80>,
             'Ridgeline': defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                         {99: defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                                      {'all': defaultdict(<function autoencodix.utils._utils.nested_dict()>,
                                                   {'user_label': <seaborn.axisgrid.FacetGrid at 0x7cd5412c3d70>,
                                                    'condition': <seaborn.axisgrid.FacetGrid at 0x7cd5410be630>})}),
... [output truncated]
```

In \[13\]:

Copied!

```
my_selection = ["loss_relative", "Ridgeline", "ML_Evaluation"]

disent.visualizer.save_plots(
	path="./tutorial_res/myplots/",
	which=my_selection, # Options are "all" or a list of plot names as in disent.visualizer.plots.keys()
	format="png" # Options are "png", "pdf, "svg"
)
```

my_selection = ["loss_relative", "Ridgeline", "ML_Evaluation"] disent.visualizer.save_plots( path="./tutorial_res/myplots/", which=my_selection, # Options are "all" or a list of plot names as in disent.visualizer.plots.keys() format="png" # Options are "png", "pdf, "svg" )

## Customization[¶](#customization)

The stored figure handles can be recalled to customize labels and styles to your liking.

In \[14\]:

Copied!

```
import matplotlib.pyplot as plt

fig_weights = disent.visualizer.plots["ModelWeights"]

# change the title
fig_weights.suptitle("My custom title for model weights")

fig_weights.figure
```

import matplotlib.pyplot as plt fig_weights = disent.visualizer.plots["ModelWeights"]

# change the title

fig_weights.suptitle("My custom title for model weights") fig_weights.figure

Out\[14\]:

# 5. XModalix and Imagix specialities[¶](#5-xmodalix-and-imagix-specialities)

XModalix and Imagix behave differently for some visualizations. This is related to the fact they are composed of multiple autoencoders (XModalix) and support images as data modality.

## Differences in standard visualizations[¶](#differences-in-standard-visualizations)

- We do not provive a model weight visualization for more complex autoencoders (...yet...)
- Similarly, coverage and total correlation `Coverage-Correlation` is not supported (...yet...)
- Supported latent space visualizations are `Ridgeline` and `2D-scatter` and show each VAE as subplot
- Loss curves behave as before and are expanded to each loss term for complex autoencoders

## Additional plots for XModalix[¶](#additional-plots-for-xmodalix)

At the core XModalix tries learn a joint latent space to enable translation across modalities. To check the capabilities of translation, we provide additional visualizations:

1. `xmodalix.visualizer.show_2D_translation()` - This plot compares on the test-split the `translated_modality` reconstruction by the XModalix with the original input in joint 2D repesentation calculated used a specified `reducer` method (default `UMAP`). A well trained XModalix and good translated modality (right plot) should as similar as possible to the original input (left plot).
1. `xmodalix.visualizer.show_image_translation()` - For images as target modality of translation, translation capability can be checked by visual comparison of test samples, possibly across classes specified under `param`, from a) original b) translated `from_key` modality to `to_key` modality and c) reference reconstruction of the image VAE inside the XModalix.

For an example checkout the [XModalix Tutorial](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/PipelineTutorials/XModalix.ipynb)

## Additional plot for Imagix[¶](#additional-plot-for-imagix)

Analogously to XModalix, `imagix.visualizer.show_image_recon_grid()` provides a grid of test sample images showing a comparison of original images and reconstructed images by the Imagix.

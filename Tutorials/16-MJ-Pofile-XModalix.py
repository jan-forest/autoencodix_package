#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T06:12:07.432Z
"""

# ## Profiling and Benchmarking XModalix
# XModalix is rather slow compared to Varix, we assume this is because of the MultiModalDataSet and the CustomSampler.
# Before improving the code we profile and benchmark the components of the XModalix pipeline. We think the following modules could be relevant:
# - MultiModalDataset
#   - which is comprised of NumericDataset, or ImageDataset
# - CoverageEnsuringSampler
# - XModlixTrainer
#   - which calls multiple GeneralTrainers
#     - which is a child of BaseTrainer
#
#
# We should also profile for four different cases:
#   - single cell vs standard tabular
#   - paired vs unpaired
#   - image vs standard tabulas
#   - image vs single cell

import os


import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function
import autoencodix as acx
from autoencodix.trainers import _xmodal_trainer, _general_trainer
from autoencodix.base import BaseTrainer, BaseDataset
from autoencodix.data import NumericDataset, MultiModalDataset, ImageDataset
from autoencodix.data._multimodal_dataset import (
    create_multimodal_collate_fn,
    CoverageEnsuringSampler,
)


from autoencodix.data._multimodal_dataset import CoverageEnsuringSampler
from autoencodix.utils.example_data import EXAMPLE_MULTI_SC, EXAMPLE_MULTI_BULK
from autoencodix.configs.xmodalix_config import XModalixConfig
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase


import torch.utils.benchmark as benchmark

if __name__ == "__main__":
    rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet")
    img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake")


    #


    clin_file = os.path.join("./data/XModalix-Tut-data/combined_clin_formatted.parquet")
    rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet")
    img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake")

    xmodalix_config = XModalixConfig(
        checkpoint_interval=100,
        class_param="CANCER_TYPE",
        epochs=1,
        beta=0.1,
        gamma=10,
        delta_class=100,
        delta_pair=300,
        latent_dim=6,
        k_filter=1000,
        batch_size=512,
        profiling=True,
        learning_rate=0.0005,
        requires_paired=False,
        float_precision="16-mixed",
        loss_reduction="sum",
        data_case=DataCase.IMG_TO_BULK,
        data_config=DataConfig(
            data_info={
                "img": DataInfo(
                    file_path=img_root,
                    img_height_resize=32,
                    img_width_resize=32,
                    data_type="IMG",
                    scaling="STANDARD",
                    translate_direction="to",
                    pretrain_epochs=0,
                ),
                "rna": DataInfo(
                    file_path=rna_file,
                    data_type="NUMERIC",
                    scaling="STANDARD",
                    pretrain_epochs=0,
                    translate_direction="from",
                ),
                "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\t"),
            },
            annotation_columns=["CANCER_TYPE_ACRONYM"],
        ),
    )

    xmodalix = acx.XModalix(config=xmodalix_config)
    result = xmodalix.run()


    # #### Getting attributes I want to profiler


    model = result.model
    forward_fn = xmodalix._trainer._modalities_forward
    loader = xmodalix._trainer._trainloader
    dataset: MultiModalDataset = CoverageEnsuringSampler(
        multimodal_dataset=loader.dataset, batch_size=xmodalix_config.batch_size
    )
    sampler = xmodalix._trainer
    collate_fn = create_multimodal_collate_fn(multimodal_dataset=dataset)


    activities = [ProfilerActivity.CPU]
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        activities += [ProfilerActivity.CUDA]


    sort_by_keyword = device + "_time_total"

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        with record_function("model_inference"):
            xmodalix.fit()
            # forward_fn(next(iter(loader)))

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

    print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=2))

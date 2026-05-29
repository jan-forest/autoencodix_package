import torch
import numpy as np
import scipy
import pandas as pd
from autoencodix.modeling._captum_forward import CaptumForward
from typing import Optional, Union
from captum.attr import (
    DeepLiftShap,
    IntegratedGradients,
)
import warnings

warnings.filterwarnings(
    "ignore",
    message="Setting forward, backward hooks and attributes on non-linear",
    category=UserWarning,
)

warnings.filterwarnings("ignore")


class FeatureImportanceExplainer:
    """
    More complete version with:
      - method selection (DeepLiftShap, IG, etc.)
      - baseline construction (mean / random / grouped)
      - subset sampling
      - reproducible randomness
    """

    def __init__(
        self,
        adata,
        model,
        method: str = "DeepLiftShap",
        sel_latent_dim: Union[
            list, int, None
        ] = None,  # list or int of latent dimensions to explain, if None explain all
        n_subset: int = 100,  # Randomly sample in input and baseline space for computational efficiency, if n_subset is None use all samples
        seed_int: int = 12,
        input_type: str = "random",  # "random" or "grouped" for selecting subset of inputs to explain
        input_group: Optional[
            str
        ] = None,  # column in .obs for grouping inputs must be provided if input_type is "grouped"
        baseline_type: str = "random",  # "random" or "mean" or "grouped" for selecting baseline samples
        baseline_group: str = None,  # column in .obs for grouping inputs must be provided if input_type is "grouped", optional for "mean", ignored for "random"
        anno_col: Optional[str] = None,  # column in .obs for grouping
    ):
        super(FeatureImportanceExplainer, self).__init__()
        self.adata_ACX = adata
        self.model = model
        self.latent_dim = model.config.latent_dim

        self.method = method
        if self.method not in {"DeepLiftShap", "IntegratedGradients"}:
            raise ValueError(f"Invalid method {method}.")
        self.n_subset = n_subset
        self.seed_int = seed_int
        self.baseline_type = baseline_type
        self.baseline_group = baseline_group
        self.anno_col = anno_col
        self.input_type = input_type
        self.input_group = input_group
        self.sel_latent_dim = sel_latent_dim

        torch.manual_seed(seed_int)
        np.random.seed(seed_int)

    def explain(self):
        adata_ACX = self.adata_ACX
        gene_names = adata_ACX.var_names
        inputs, baselines = return_inputs_baseline(
            adata_ACX,
            self.baseline_group,
            self.baseline_type,
            self.anno_col,
            self.input_type,
            self.input_group,
        )
        # set n_samples to whatever is lowest from: inputs, baselines, n_subset (if not None)
        n_samples = min(inputs.shape[0], baselines.shape[0])
        if self.n_subset is not None:
            n_samples = min(n_samples, self.n_subset)

        # Downsample input
        if self.n_subset is not None:
            indices_keep = np.random.choice(
                inputs.shape[0], size=n_samples, replace=False
            )
        else:
            indices_keep = np.arange(inputs.shape[0])
        # Downsample baselines in the same way
        if self.n_subset is not None:
            indices_keep_baseline = np.random.choice(
                baselines.shape[0], size=n_samples, replace=False
            )
        else:
            indices_keep_baseline = np.arange(baselines.shape[0])
        # for latent_dim in range(self.latent_dim):
        all_attr = []
        if isinstance(
            self.sel_latent_dim, list
        ):  # Provide list of latent dimensions to explain
            latent_dims = self.sel_latent_dim
        elif isinstance(
            self.sel_latent_dim, int
        ):  # Provide single latent dimension to explain
            latent_dims = [self.sel_latent_dim]
        elif self.sel_latent_dim is None:  # Explain all latent dimensions
            latent_dims = list(range(self.latent_dim))
        else:
            raise ValueError(
                f"Invalid sel_latent_dim {self.sel_latent_dim}. Must be int, list of ints, or None."
            )

        for latent_dim in latent_dims:
            print("Calculating attributions for latent dimension:", latent_dim)
            cp_forward_dim = CaptumForward(model=self.model, dim=latent_dim)
            if self.method == "DeepLiftShap":
                cp_explainer = DeepLiftShap(cp_forward_dim)
            if self.method == "IntegratedGradients":
                cp_explainer = IntegratedGradients(cp_forward_dim)
            attributions = cp_explainer.attribute(
                inputs=inputs[indices_keep].float(),
                baselines=baselines[indices_keep_baseline].float(),
                return_convergence_delta=False,
            )
            avg_abs_attributions = attributions.abs().mean(dim=0)
            all_attr.append(avg_abs_attributions.detach().cpu())
        attr_matrix = torch.stack(all_attr).T.numpy()
        if hasattr(self.model, "ontologies") and self.model.ontologies is not None:
            cols = [list(self.model.ontologies[0].keys())[i] for i in latent_dims]
        else:
            cols = [f"Latent_Dim_{i}" for i in latent_dims]
        df_attributions = pd.DataFrame(
            attr_matrix,
            index=list(gene_names),
            columns=cols,
        )

        return df_attributions


def return_inputs_baseline(
    adata, baseline_group, baseline_type, anno_col, input_type, input_group
):
    baseline_torch_all = torch.tensor(
        adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    )
    ## Define inputs
    if input_type == "grouped":
        if anno_col is None or input_group is None:
            raise ValueError(
                "If input_type is 'grouped', anno_col and input_group must be provided to specify the grouping column in .obs."
            )
        input_adata = adata[adata.obs[anno_col] == input_group]
        inputs = torch.tensor(
            input_adata.X.toarray()
            if scipy.sparse.issparse(input_adata.X)
            else input_adata.X
        )
    elif input_type == "random":
        inputs = torch.tensor(
            adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        )

    ## Define baselines
    if baseline_type == "random":
        # Generates a bootstrap random from input in the same size
        baseline_random = baseline_torch_all[
            torch.randint(0, baseline_torch_all.size(0), (1,)).item()
        ]
        baselines = torch.tensor(
            np.tile(baseline_random, (baseline_torch_all.shape[0], 1))
        )
    elif baseline_type == "grouped":
        if anno_col is None or baseline_group is None:
            raise ValueError(
                "If baseline_type is 'grouped', anno_col and baseline_group must be provided to specify the grouping column in .obs."
            )
        base_adata_filtered = adata[adata.obs[anno_col] == baseline_group]
        base_filtered = torch.tensor(
            base_adata_filtered.X.toarray()
            if scipy.sparse.issparse(base_adata_filtered.X)
            else base_adata_filtered.X
        )
        # Generates a bootstrap random from the filtered input in the same size
        baseline_grouped = base_filtered[
            torch.randint(0, base_filtered.size(0), (1,)).item()
        ]
        baselines = torch.tensor(
            np.tile(baseline_grouped, (base_adata_filtered.shape[0], 1))
        )
    elif baseline_type == "mean":
        if baseline_group == None:
            baseline_mean = baseline_torch_all.mean(axis=0)  # gene_means
        else:
            if anno_col is None:
                raise ValueError(
                    "If baseline_group is not 'all', anno_col must be provided to specify the grouping column in .obs."
                )
            base_adata_filtered = adata[adata.obs[anno_col] == baseline_group]
            base_filtered = torch.tensor(
                base_adata_filtered.X.toarray()
                if scipy.sparse.issparse(base_adata_filtered.X)
                else base_adata_filtered.X
            )
            baseline_mean = base_filtered.mean(axis=0)  # gene_means
        baselines = torch.tensor(
            np.tile(baseline_mean, (baseline_torch_all.shape[0], 1))
        )
    else:
        raise ValueError(f"Invalid baseline_type {baseline_type}.")

    # inputs = torch.tensor(
    #     input_adata.X.toarray()
    #     if scipy.sparse.issparse(input_adata.X)
    #     else input_adata.X
    # )
    # if baseline_group == "all":
    #     if baseline_type == "mean":
    #         baseline_mean = inputs.mean(axis=0)  # gene_means
    #         baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
    #     if baseline_type == "random_sample":
    #         baseline_random = inputs[torch.randint(0, inputs.size(0), (1,)).item()]
    #         baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    # else:
    #     if anno_col is None:
    #         raise ValueError(
    #             "If baseline_group is not 'all', anno_col must be provided to specify the grouping column in .obs."
    #         )
    #     input_adata_filtered = input_adata[input_adata.obs[anno_col] == baseline_group]
    #     inputs_filtered = torch.tensor(
    #         input_adata_filtered.X.toarray()
    #         if scipy.sparse.issparse(input_adata_filtered.X)
    #         else input_adata_filtered.X
    #     )
    #     if baseline_type == "mean":
    #         baseline_mean = inputs_filtered.mean(axis=0)  # gene_means
    #         baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
    #     if baseline_type == "random_sample":
    #         baseline_random = inputs_filtered[
    #             torch.randint(0, inputs_filtered.size(0), (1,)).item()
    #         ]
    #         baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    return inputs, baselines

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# from autoencodix.utils._result import Result
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils.default_config import DefaultConfig



class Visualizer(BaseVisualizer):


    # def visualize(self, result: Result, config: DefaultConfig) -> Result:
    def visualize(self, result, config: DefaultConfig):

        ## Make Model Weights plot
        result.plots["ModelWeights"] = self.plot_model_weights(model=result.model)

        ## Make long format of losses 
        loss_df_melt = self.make_loss_format(result=result,config=config)

        ## Make plot loss absolute
        result.plots["loss_absolute"] = self.make_loss_plot(df_plot=loss_df_melt, type="absolute")
        ## Make plot loss relative
        result.plots["loss_relative"] = self.make_loss_plot(df_plot=loss_df_melt, type="relative")

        return result
    
    @staticmethod
    def plot_model_weights(model):
        """
        Visualization of model weights in encoder and decoder layers as heatmap for each layer as subplot.
        ARGS:
            model (torch.nn.Module): PyTorch model instance.
            filepath (str): Path specifying save name and location.
        RETURNS:
            fig (matplotlib.figure): Figure handle (of last plot)
        """
        all_weights = []
        names = []
        for name, param in model.named_parameters():
            if "weight" in name and len(param.shape) == 2:
                if not "var" in name:  ## For VAE plot only mu weights
                    all_weights.append(param.detach().cpu().numpy())
                    names.append(name[:-7])

        layers = int(len(all_weights) / 2)
        fig, axes = plt.subplots(2, layers, sharex=False, figsize=(20, 10))

        for l in range(layers):
            ## Encoder Layer
            if layers > 1:
                sns.heatmap(
                    all_weights[l],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[0, l],
                ).set(title=names[l])
                ## Decoder Layer
                sns.heatmap(
                    all_weights[layers + l],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[1, l],
                ).set(title=names[layers + l])
                axes[1, l].set_xlabel("In Node", size=12)
            else:
                sns.heatmap(
                    all_weights[l],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[l],
                ).set(title=names[l])
                ## Decoder Layer
                sns.heatmap(
                    all_weights[l + 2],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[l + 1],
                ).set(title=names[l + 2])
                axes[1].set_xlabel("In Node", size=12)

        if layers > 1:
            axes[1, 0].set_ylabel("Out Node", size=12)
            axes[0, 0].set_ylabel("Out Node", size=12)
        else:
            axes[1].set_ylabel("Out Node", size=12)
            axes[0].set_ylabel("Out Node", size=12)

        ## Add title
        fig.suptitle("Model Weights", size=20)
        plt.close()
        return fig

    def plot_2D(
        embedding,
        labels,
        param=None,
        layer="latent space",
        figsize=(24, 15),
        center=True,
        plot_numeric=False,
        xlim=None,
        ylim=None,
        scale=None,
        no_leg=False,
    ):
        """
        Creates a 2D visualization of the 2D embedding of the latent space.
        ARGS:
            embedding (pd.DataFrame): embedding on which is visualized. Assumes prior 2D dimension reduction.
            labels (list): Clinical parameters or cluster labels to colorize samples (points)
            layer (str): Label for plot title to indicate which network layer is represented by UMAP/TSNE
            figsize (tuple): Figure size specification.
            center (boolean): If True (default) centers of clusters/groups are visualized as stars.
        RETURNS:
            fig (matplotlib.figure): Figure handle

        """
        
        numeric = False
        if not (type(labels[0]) is str):
            if len(np.unique(labels)) > 3:
                if not plot_numeric:
                    print(
                        f"The provided label column is numeric and converted to categories."
                    )
                    labels = pd.qcut(
                        labels, q=4, labels=["1stQ", "2ndQ", f"3rdQ", f"4thQ"]
                    ).astype(str)
                else:
                    center = False  ## Disable centering for numeric params
                    numeric = True
            else:
                labels = [str(x) for x in labels]

        fig, ax1 = plt.subplots(figsize=figsize)

        # check if label or embedding is longerm and duplicate the shorter one
        if len(labels) < embedding.shape[0]:
            print(
                "Given labels do not have the same length as given sample size. Labels will be duplicated."
            )
            labels = [
                label for label in labels for _ in range(embedding.shape[0] // len(labels))
            ]
        elif len(labels) > embedding.shape[0]:
            labels = list(set(labels))

        if numeric:
            ax2 = sns.scatterplot(
                x=embedding.iloc[:, 0],
                y=embedding.iloc[:, 1],
                hue=labels,
                palette="bwr",
                s=40,
                alpha=0.5,
                ec="black",
            )
        else:
            ax2 = sns.scatterplot(
                x=embedding.iloc[:, 0],
                y=embedding.iloc[:, 1],
                hue=labels,
                hue_order=np.unique(labels),
                s=40,
                alpha=0.5,
                ec="black",
            )
        if center:
            means = embedding.groupby(by=labels).mean()

            ax2 = sns.scatterplot(
                x=means.iloc[:, 0],
                y=means.iloc[:, 1],
                hue=np.unique(labels),
                hue_order=np.unique(labels),
                s=200,
                ec="black",
                alpha=0.9,
                marker="*",
                legend=False,
                ax=ax2,
            )

        if not xlim == None:
            ax2.set_xlim(xlim[0], xlim[1])

        if not ylim == None:
            ax2.set_ylim(ylim[0], ylim[1])

        if not scale == None:
            plt.yscale(scale)
            plt.xscale(scale)
        ax2.set_xlabel("Dim 1")
        ax2.set_ylabel("Dim 2")
        legend_cols = 1
        if len(np.unique(labels)) > 10:
            legend_cols = 2

        if no_leg:
            plt.legend([], [], frameon=False)
        else:
            sns.move_legend(
                ax2,
                "upper left",
                bbox_to_anchor=(1, 1),
                ncol=legend_cols,
                title=param,
                frameon=False,
            )

        # Add title to the plot
        ax2.set_title(layer)
        
        plt.close()
        return fig

    def plot_latent_ridge(lat_space, label_list=None, param=None):
        """
        Creates a ridge line plot of latent space dimension where each row shows the density of a latent dimension and groups (ridges).
        ARGS:
            lat_space (pd.DataFrame): DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
            label_list (list): List of labels for each sample. If None, all samples are considered as one group. 
            param (str): Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        RETURNS:
            fig (matplotlib.figure): Figure handle (of last plot)
        """
        sns.set_theme(
            style="white", rc={"axes.facecolor": (0, 0, 0, 0)}
        )  ## Necessary to enforce overplotting

        df = pd.melt(lat_space, var_name="latent dim", value_name="latent intensity")
        df["sample"] = len(lat_space.columns) * list(lat_space.index)

        if label_list is None:
            param = "all"
            label_list = ["all"] * len(df)

        # print(labels[0])
        if not (type(label_list[0]) is str):
            if len(np.unique(label_list)) > 3:
                label_list = pd.qcut(
                    label_list, q=4, label_list=["1stQ", "2ndQ", f"3rdQ", f"4thQ"]
                ).astype(str)
            else:
                label_list = [str(x) for x in label_list]
        df[param] = len(lat_space.columns) * label_list

        exclude_missing_info = (df[param] == "unknown") | (df[param] == "nan")

        xmin = (
            df.loc[~exclude_missing_info, ["latent intensity", "latent dim", param]]
            .groupby([param, "latent dim"], observed=False)
            .quantile(0.05)
            .min()
        )
        xmax = (
            df.loc[~exclude_missing_info, ["latent intensity", "latent dim", param]]
            .groupby([param, "latent dim"], observed=False)
            .quantile(0.9)
            .max()
        )

        if len(np.unique(df[param])) > 8:
            cat_pal = sns.husl_palette(len(np.unique(df[param])))
        else:
            cat_pal = sns.color_palette(n_colors=len(np.unique(df[param])))

        g = sns.FacetGrid(
            df[~exclude_missing_info],
            row="latent dim",
            hue=param,
            aspect=12,
            height=0.8,
            xlim=(xmin.iloc[0], xmax.iloc[0]),
            palette=cat_pal,
        )

        g.map_dataframe(
            sns.kdeplot,
            "latent intensity",
            bw_adjust=0.5,
            clip_on=True,
            fill=True,
            alpha=0.5,
            warn_singular=False,
            ec="k",
            lw=1,
        )

        def label(data, color, label, text="latent dim"):
            ax = plt.gca()
            label_text = data[text].unique()[0]
            ax.text(
                0.0,
                0.2,
                label_text,
                fontweight="bold",
                ha="right",
                va="center",
                transform=ax.transAxes,
            )

        g.map_dataframe(label, text="latent dim")

        g.set(xlim=(xmin.iloc[0], xmax.iloc[0]))
        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-0.5)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        g.add_legend()

        plt.close()
        return g

    def make_loss_plot(self, df_plot, type):
        if type=="absolute":
            fig, axes = plt.subplots(1, len(df_plot['Loss Term'].unique()), figsize=(15, 5), sharey=False)
            ax = 0
            for term in df_plot['Loss Term'].unique():
                axes[ax] = sns.lineplot(data=df_plot[(df_plot['Loss Term'] == term)], x="Epoch", y="Loss Value", hue="Split", ax=axes[ax]).set_title(term)
                ax += 1
            
            plt.close()

        if type=="relative":
            exclude = df_plot['Loss Term'] != 'total_loss'

            fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

            ax = 0

            for split in df_plot['Split'].unique():
                axes[ax] = sns.kdeplot(data=df_plot[exclude & (df_plot['Split'] == split)], x="Epoch", hue="Loss Term", multiple="fill", weights="Loss Value",clip=[0,30], ax=axes[ax]).set_title(split)
                ax += 1
            
            plt.close()

        return fig
    
    # def make_loss_format(self, result: Result, config: DefaultConfig) -> pd.DataFrame:
    def make_loss_format(self, result, config: DefaultConfig) -> pd.DataFrame:
        loss_df_melt = pd.DataFrame()

        for term in result.sub_losses.keys():
            loss_df = pd.DataFrame.from_dict(
                result.sub_losses.get(key=term).get(),
                orient='index'
                )
            ## Make weighting of loss terms 
            if term == "var_loss":
                loss_df = loss_df *  config.beta

            loss_df['Epoch'] = loss_df.index +1
            loss_df['Loss Term'] = term

            # print(loss_df)
            loss_df_melt = pd.concat([	loss_df_melt, 
                                        loss_df.melt(id_vars=['Epoch','Loss Term'], var_name='Split', value_name='Loss Value')],
                                        axis=0).reset_index(drop=True)


        loss_df = pd.DataFrame.from_dict(
            result.losses.get(),
            orient='index'
                )
        loss_df['Epoch'] = loss_df.index +1
        loss_df['Loss Term'] = 'total_loss'
        loss_df_melt = pd.concat([	loss_df_melt, 
                                    loss_df.melt(id_vars=['Epoch','Loss Term'], var_name='Split', value_name='Loss Value')],
                                    axis=0).reset_index(drop=True)
                                    
        loss_df_melt['Loss Value'] = loss_df_melt['Loss Value'].astype(float)

        return loss_df_melt

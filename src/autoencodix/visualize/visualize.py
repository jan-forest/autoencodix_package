import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from autoencodix.utils._result import Result
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils.default_config import DefaultConfig



class Visualizer(BaseVisualizer):


    def visualize(self, result: Result, config: DefaultConfig) -> Result:

        ## Make long format of losses 
        loss_df_melt = self.make_loss_format(result=result,config=config)

        ## Make plot loss absolute
        result.plots["loss_absolute"] = self.make_loss_plot(df_plot=loss_df_melt, type="absolute")
        ## Make plot loss relative
        result.plots["loss_relative"] = self.make_loss_plot(df_plot=loss_df_melt, type="relative")

        return result
    
    
    def make_loss_plot(self, df_plot, type):
        if type=="absolute":
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
            ax = 0
            for split in df_plot['Split'].unique():
                axes[ax] = sns.lineplot(data=df_plot[(df_plot['Split'] == split)], x="Epoch", y="Loss Value", hue="Loss Term", ax=axes[ax]).set_title(split)
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
    
    def make_loss_format(self, result: Result, config: DefaultConfig) -> pd.DataFrame:
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

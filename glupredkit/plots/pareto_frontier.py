import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, normalize_results=True, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        metrics = ['rmse', 'temporal_gain', 'g_mean']
        data = []
        plots = []
        names = []

        # Creates results df
        for df in dfs:
            model_name = df['Model Name'][0]
            row = {"Model Name": model_name}

            for metric_name in metrics:
                score = df[f'{metric_name}_{prediction_horizon}'][0]
                row[metric_name] = score
            data.append(row)

        results_df = pd.DataFrame(data)
        if normalize_results:
            prediction_horizon = float(prediction_horizon)
            results_df['temporal_gain'] = (prediction_horizon - results_df['temporal_gain']) / prediction_horizon
            results_df['g_mean'] = 1 - results_df['g_mean']
            max_rmse = results_df['rmse'].max()
            if unit_config_manager.use_mgdl:
                results_df['rmse'] = results_df['rmse'].apply(lambda x: x / max(max_rmse, 18.018))
            else:
                results_df['rmse'] = results_df['rmse'].apply(lambda x: x / max(max_rmse, 1))

        def pareto_frontier(df, normalize_results):
            """
            Find the Pareto frontier from the DataFrame with metrics.
            This function returns the set of models that are not dominated by any other model.
            A model is dominated if another model has better (lower) performance across all metrics.
            """
            pareto_front = []

            # Change so that low value is better
            if not normalize_results:
                df['temporal_gain'] = -df['temporal_gain']
                df['g_mean'] = -df['g_mean']

            # Iterate through each model (row)
            for i, model in df.iterrows():
                is_dominated = False

                # Compare with every other model (row)
                for j, other_model in df.iterrows():
                    if (other_model < model).all():  # if all metrics of other_model are better (lower)
                        is_dominated = True
                        break  # Stop comparing this model (i) with further models, it's dominated

                if not is_dominated:
                    pareto_front.append(model)  # This model is not dominated, so add to Pareto front

            # Change back to original
            pareto_front_df = pd.DataFrame(pareto_front)
            if not normalize_results:
                pareto_front_df['temporal_gain'] = -pareto_front_df['temporal_gain']
                pareto_front_df['g_mean'] = -pareto_front_df['g_mean']
            return pareto_front_df

        pareto_front_df = pareto_frontier(results_df.copy(), normalize_results=normalize_results)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', '+', '|', '_']
        cmap = sns.color_palette("colorblind", as_cmap=True)  # Seaborn's colorblind-friendly palette

        def plot_2d(df, pareto_front_df, col1, col2, label1, label2):
            """
            Plot the Precision vs Recall and highlight the Pareto frontier
            """
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot()

            unique_models = df['Model Name'].unique()
            for i, model in enumerate(unique_models):
                model_df = df[df["Model Name"] == model]
                plt.scatter(model_df[col1], model_df[col2], color=cmap[i], marker=markers[i],
                            label=model, s=100, alpha=0.8)

            # Highlight non Pareto frontier
            not_pareto_front_df = df[~df["Model Name"].isin(pareto_front_df["Model Name"])]
            ax.scatter(not_pareto_front_df[col1], not_pareto_front_df[col2], color='black', label='Not on Pareto Frontier', s=100,
                       marker='x')

            if normalize_results:
                plt.scatter(0.0, 0.0, color='green', label='Optimal Point', s=150, marker='o')
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)

            plt.title(f'{label1.split("$")[0]} and {label2.split("$")[0]}')
            plt.xlabel(label1, labelpad=5)
            plt.ylabel(label2, labelpad=5)
            #plt.legend(loc='lower left')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot 2D
        plot_2d(results_df, pareto_front_df, 'rmse', 'g_mean', label1='RMSE$_{scaled}$', label2='GM$_{scaled}$')
        plot_2d(results_df, pareto_front_df, 'rmse', 'temporal_gain', label1='RMSE$_{scaled}$', label2='TG$_{scaled}$')
        plot_2d(results_df, pareto_front_df, 'g_mean', 'temporal_gain', label1='GM$_{scaled}$', label2='TG$_{scaled}$')

        def plot_3d(df, pareto_front_df):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot for all models, in / not in pareto frontier
            unique_models = df['Model Name'].unique()
            for i, model in enumerate(unique_models):
                model_df = df[df["Model Name"] == model]
                ax.scatter(model_df['rmse'], model_df['g_mean'], model_df['temporal_gain'],
                           color=cmap[i], marker=markers[i], label=model, s=100, alpha=1.0)

            not_pareto_front_df = df[~df["Model Name"].isin(pareto_front_df["Model Name"])]

            # Add vertical lines to the XY plane for better visibility
            for i, row in df.iterrows():
                ax.plot([row['rmse'], row['rmse']], [row['g_mean'], row['g_mean']], [0, row['temporal_gain']],
                        color=cmap[i], linestyle='dotted', alpha=0.6)
                ax.plot([row['rmse'], row['rmse']], [1, row['g_mean']], [row['temporal_gain'], row['temporal_gain']],
                        color=cmap[i], linestyle='dotted', alpha=0.6)

            # Highlight not Pareto frontier
            ax.scatter(not_pareto_front_df['rmse'], not_pareto_front_df['g_mean'], not_pareto_front_df['temporal_gain'],
                       color='black', label='Not Pareto Frontier', s=100, marker='x')

            # Highlight the optimal point
            if normalize_results:
                ax.scatter(0.0, 0.0, 0.0,
                           color='green', label='Optimal Point', s=150, marker='o')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
            else:
                ax.scatter(0.0, 1.0, float(prediction_horizon),
                           color='green', label='Optimal Point', s=150, marker='o')

            """
            # Add annotations (model names) for each point
            for i, model_name in enumerate(df['Model Name']):
                ax.text(df['rmse'].iloc[i] + 0.00, df['g_mean'].iloc[i] + 0.04, df['temporal_gain'].iloc[i] + 0.00,
                        model_name, color='black', fontsize=10)
            
            for i, model_name in enumerate(df['Model Name']):
                ax.text(df['rmse'].iloc[i] + 0.04, df['g_mean'].iloc[i] + 0.04, df['temporal_gain'].iloc[i],
                        model_name, fontsize=10, color='black',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            """
            ax.set_title('3D Visualization of the Pareto Frontier')
            ax.set_xlabel('RMSE$_{scaled}$', labelpad=15)
            ax.set_ylabel('GM$_{scaled}$', labelpad=15)
            ax.set_zlabel('TG$_{scaled}$', labelpad=15)
            ax.legend(loc='upper left')

        # Plot 3D visualization
        plot_3d(results_df, pareto_front_df)

        plot_name = f'pareto_frontier_ph_{prediction_horizon}'
        plots.append(plt.gcf())
        names.append(plot_name)

        if show_plot:
            plt.tight_layout()
            plt.show()
        plt.close()

        return plots, names


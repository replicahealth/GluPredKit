import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ast
from .base_plot import BasePlot
from matplotlib.lines import Line2D
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, plot_predictions=True, *args):
        """
        This plot plots predicted trajectories from the measured values. A random subsample of around 24 hours will
        be plotted.
        """
        n_samples = 12 * 4
        start_index = 12 * 9

        plots = []
        names = []

        for df in dfs:
            model_name = df['Model Name'][0]

            y_pred = get_list_from_string(df, f'y_pred_{prediction_horizon}')[start_index:start_index + n_samples]
            y_true = get_list_from_string(df, 'test_input_CGM')[start_index:start_index + n_samples]

            if unit_config_manager.use_mgdl:
                hypo = 70
                hyper = 180
            else:
                y_pred = [unit_config_manager.convert_value(val) for val in y_pred]
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                hypo = 3.9
                hyper = 10.0

            plt.figure(figsize=(16, 5))

            t = [i * 5 / 60 for i in range(len(y_true))]
            plt.scatter(t, y_true, color='black', label='Glucose Measurements')

            # Plot line for predicted values
            prediction_steps = int(prediction_horizon) // 5
            t_pred = [(i + prediction_steps) * 5 / 60 for i in range(len(y_pred) - prediction_steps)]
            plt.plot(t_pred, y_pred[:len(t_pred)], color='green', linestyle='--', label=f'{model_name} Predictions')

            # Add arrows to indicate prediction horizon
            interval = 12  # Interval is amount of measurements between each arrow, 6 is half an hour
            for i in range(0, len(y_true) - prediction_steps, interval):
                if plot_predictions:
                    x_start = t[i]
                    y_start = y_true[i]
                    x_end = t_pred[i]
                    y_end = y_pred[i]

                    dx = x_end - x_start
                    dy = y_end - y_start

                    arrow = patches.FancyArrowPatch(
                        (x_start, y_start), (x_start + dx, y_start + dy),
                        arrowstyle='-|>', color='blue', alpha=1.0,
                        mutation_scale=20, linewidth=1
                    )
                    plt.gca().add_patch(arrow)

            # Add the right y-axis
            ax_left = plt.gca()
            ax_right = ax_left.twinx()

            # Synchronize right y-axis with left y-axis values scaled by 18
            ax_right.set_ylim(ax_left.get_ylim()[0] * 18, ax_left.get_ylim()[1] * 18)
            ax_right.set_ylabel('Blood Glucose [mg/dL]', fontsize=18)

            # Match the ticks on the right y-axis to those on the left
            # left_ticks = ax_left.get_yticks()  # Get tick positions from the left axis
            # right_ticks = [tick * 18 for tick in left_ticks]  # Scale the left ticks by 18
            right_ticks = [75, 110, 145, 180, 215, 250, 285, 320]
            ax_right.set_yticks(right_ticks)  # Set these ticks for the right axis
            ax_right.tick_params(axis='y', labelsize=14)

            # Format right y-axis ticks to display no decimals
            ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{int(val)}"))

            # Customize legends
            glucose_legend = Line2D([0], [0], color='black', marker='o', linestyle='None', label='CGM Values')
            # prediction_legend = Line2D([0], [0], color='green', linestyle='--', label=f'{model_name} Predictions')
            prediction_legend = Line2D([0], [0], color='green', linestyle='--', label=f'Predictions')
            handles = [glucose_legend, prediction_legend]
            plt.legend(handles=handles, loc='upper right', fontsize=16)


            # Indicate hypo- and hyperglycemic range
            ax_left.axhline(y=hypo, color='black', linestyle='--', linewidth=1)
            ax_left.axhline(y=hyper, color='black', linestyle='--', linewidth=1)
            ax_left.text(x=-0.15, y=hypo + 0.2, s="Hypoglycemic threshold", color="black", ha="left", fontsize=13)
            #ax_left.text(x=-0.4, y=hyper + 0.2, s="Hyperglycemic threshold", color="black", ha="left", fontsize=13)

            # Labeling axes
            ax_left.set_xlabel('Time (hours)', fontsize=18)
            ax_left.set_ylabel('Blood Glucose [mmol/L]', fontsize=18, color='black')
            ax_left.tick_params(axis="x", labelsize=14)
            ax_left.tick_params(axis="y", labelsize=14, colors='black')

            # Add gridlines with vertical lines only, behind the plot elements
            ax_left.grid(True, which='both', axis='x', linestyle='--', zorder=-999)  # Grid behind everything

            # Set title
            plt.title(f"{model_name} {prediction_horizon}-minute PH", fontsize=20)
            plot_name = f'{model_name}_single_prediction_horizon_ph_{prediction_horizon}'
            plots.append(plt.gcf())
            names.append(plot_name)

            if show_plot:
                plt.tight_layout()
                plt.show()
            plt.close()

        return plots, names

def get_list_from_string(df, col):
    string_values = df[col][0]
    string_values = string_values.replace("nan", "None")
    list_values = ast.literal_eval(string_values)
    list_values = [np.nan if x is None else x for x in list_values]
    return list_values

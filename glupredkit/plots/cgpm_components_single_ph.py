import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ast
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.metrics import RMSE, GeoMean, TemporalGain


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, plot_predictions=True, *args):
        """
        This plot plots predicted trajectories from the measured values. A random subsample of around 24 hours will
        be plotted.
        """
        n_samples = 12 * 24 * 1
        start_index = 12 * 4

        plots = []
        names = []

        for df in dfs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [2, 2]})

            model_name = df['Model Name'][0]

            y_pred = get_list_from_string(df, f'y_pred_{prediction_horizon}')[start_index:start_index + n_samples]
            y_measured = get_list_from_string(df, 'test_input_CGM')[start_index:start_index + n_samples]
            y_true = get_list_from_string(df, f'target_{prediction_horizon}')[start_index:start_index + n_samples]

            if unit_config_manager.use_mgdl:
                hypo = 70
                hyper = 180
            else:
                y_pred = [unit_config_manager.convert_value(val) for val in y_pred]
                y_measured = [unit_config_manager.convert_value(val) for val in y_measured]
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                hypo = 3.9
                hyper = 10.0

            t = [i * 5 / 60 for i in range(len(y_measured))]  # Time in hours
            prediction_steps = int(prediction_horizon) // 5
            t_pred = [(i + prediction_steps) * 5 / 60 for i in range(len(y_pred) - prediction_steps)]

            # --- First Plot (Glucose Measurements & Predictions) ---
            ax1.scatter(t, y_measured, color='black', label='Glucose Measurements')
            ax1.plot(t_pred, y_pred[:len(t_pred)], color='green', linestyle='--', label=f'{model_name} Predictions')

            # Add arrows to indicate prediction horizon
            interval = 12
            for i in range(0, len(y_measured) - prediction_steps, interval):
                if plot_predictions:
                    x_start = t[i]
                    y_start = y_measured[i]
                    x_end = t_pred[i]
                    y_end = y_pred[i]

                    dx = x_end - x_start
                    dy = y_end - y_start

                    arrow = patches.FancyArrowPatch(
                        (x_start, y_start), (x_start + dx, y_start + dy),
                        arrowstyle='-|>', color='blue', alpha=1.0,
                        mutation_scale=20, linewidth=1
                    )
                    ax1.add_patch(arrow)

            ax1.set_ylabel("Glucose [mmol/L]", fontsize=14)
            ax1.legend(loc='upper right', fontsize=12)

            # Right y-axis for mg/dL
            ax_right = ax1.twinx()
            ax_right.set_ylim(ax1.get_ylim()[0] * 18, ax1.get_ylim()[1] * 18)
            ax_right.set_ylabel('Blood Glucose [mg/dL]', fontsize=14)
            right_ticks = [75, 110, 145, 180, 215, 250, 285, 320]
            ax_right.set_yticks(right_ticks)
            ax_right.tick_params(axis='y', labelsize=12)
            ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{int(val)}"))

            # Indicate hypo- and hyperglycemic range
            ax1.axhline(y=hypo, color='black', linestyle='--', linewidth=1)
            ax1.axhline(y=hyper, color='black', linestyle='--', linewidth=1)
            ax1.text(x=20.65, y=hypo + 0.3, s="Hypoglycemic threshold", color="black", ha="left", fontsize=13)
            ax1.text(x=20.5, y=hyper + 0.3, s="Hyperglycemic threshold", color="black", ha="left", fontsize=13)

            # --- Second Plot (Additional Information - Example: Prediction Errors) ---
            n_samples = len(y_pred)
            window_size = 6

            def get_error_list(metric_class, y_true_list, y_pred_list, n_samples, window_size=6, use_mg_dl=False):
                if use_mg_dl and not unit_config_manager.use_mgdl:
                    y_true_list = [val * 18.018 for val in y_true_list]
                    y_pred_list = [val * 18.018 for val in y_pred_list]

                error_moving = [
                    metric_class().__call__(
                        y_true_list[max(0, i - window_size): min(n_samples, i + window_size)],
                        y_pred_list[max(0, i - window_size): min(n_samples, i + window_size)],
                        prediction_horizon=int(prediction_horizon))
                    for i in range(n_samples)
                ]
                return error_moving

            rmse_moving = get_error_list(RMSE, y_true, y_pred, n_samples, window_size=window_size)
            gm_moving = get_error_list(GeoMean, y_true, y_pred, n_samples, window_size=window_size, use_mg_dl=True)
            #tg_moving = get_error_list(TemporalGain, y_true, y_pred, n_samples, window_size=window_size, use_mg_dl=True)

            #tg_val = TemporalGain().__call__(y_true, y_pred, prediction_horizon=int(prediction_horizon))
            tg_val = TemporalGain().__call__(get_list_from_string(df, f'target_{prediction_horizon}'),
                                             get_list_from_string(df, f'y_pred_{prediction_horizon}'),
                                             prediction_horizon=int(prediction_horizon))

            def scale_to_01(lst):
                arr = np.array(lst)
                min_val = np.min(arr)
                max_val = np.max(arr)
                scaled_arr = (arr - min_val) / (max_val - min_val)
                return scaled_arr.tolist()

            # Scale lists
            rmse_moving = scale_to_01(rmse_moving)
            gm_moving = [1 - val for val in gm_moving]
            tg_list = [(int(prediction_horizon) - tg_val) / int(prediction_horizon)] * n_samples
            #tg_moving = [(int(prediction_horizon) - val) / int(prediction_horizon) for val in tg_moving]

            # TODO: why is tg scaled 0
            # TODO: add horizontal line for each metric, same colour but dotted!

            from scipy.ndimage import gaussian_filter1d

            #prediction_errors = [y_pred[i] - y_true[i] for i in range(len(y_true))]
            ax2.plot(t_pred, rmse_moving[:len(t_pred)], color='red', linestyle='-', label='RMSE')
            ax2.plot(t_pred, rmse_moving[:len(t_pred)], color='red', linestyle='-', label='RMSE')
            ax2.plot(t_pred, gaussian_filter1d(gm_moving[:len(t_pred)], sigma=5), color='blue', linestyle='-', label='Geometric Mean')
            ax2.plot(t_pred, tg_list[:len(t_pred)], color='green', linestyle='-', label='Temporal Gain')
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)  # Baseline

            # TODO: make the plot pretty and name the curves in a good way

            # TODO: print table with the results

            ax2.set_ylabel("Prediction Error", fontsize=14)
            ax2.set_xlabel("Time (hours)", fontsize=14)
            ax2.legend(loc='upper right', fontsize=12)

            # TODO: make the map instead
            plt.title(f"{model_name}")

            # Adjust spacing
            plt.tight_layout()
            plt.show()

        return plots, names

def get_list_from_string(df, col):
    string_values = df[col][0]
    string_values = string_values.replace("nan", "None")
    list_values = ast.literal_eval(string_values)
    list_values = [np.nan if x is None else x for x in list_values]
    return list_values



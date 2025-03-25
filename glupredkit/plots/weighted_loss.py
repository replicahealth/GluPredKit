import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, *args):

        plots = []
        names = []

        # Plotting zone cost
        bg_values = np.linspace(18.6, 600, 500)
        cost_values = [zone_cost(bg) for bg in bg_values]
        plt.figure(figsize=(7, 4))
        plt.plot(bg_values, cost_values, label='Zone Cost', color='blue')
        plt.axvline(70, color='green', linestyle='--', label='Hypoglycemic Range (70)')
        plt.axvline(180, color='red', linestyle='--', label='Hyperglycemic Range (180)')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title('Zone Cost Function', fontsize=16)
        plt.xlabel('BG (Blood Glucose)', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plot_name = f'zone_cost'
        plots.append(plt.gcf())
        names.append(plot_name)
        plt.close()

        # Generate bg and delta_bg ranges
        bg_values = np.linspace(0, 600, 100)
        delta_bg_values = np.linspace(-18, 18, 100)
        bg_grid, delta_bg_grid = np.meshgrid(bg_values, delta_bg_values)
        cost_grid = np.array([[slope_cost(bg, delta_bg) for bg in bg_values] for delta_bg in delta_bg_values])
        plt.figure(figsize=(5, 4))
        plt.contourf(bg_grid, delta_bg_grid, cost_grid, cmap='RdYlGn_r', levels=50)
        plt.colorbar(label='Cost')
        plt.title('Slope Cost Function Heatmap', fontsize=16)
        plt.xlabel('BG (Blood Glucose)', fontsize=14)
        plt.ylabel('Delta BG', fontsize=14)

        plot_name = f'slope_cost'
        plots.append(plt.gcf())
        names.append(plot_name)

        if show_plot:
            plt.tight_layout()
            plt.show()
        plt.close()

        # TODO 3: Plot cost function in 3d

        for df in dfs:
            model_name = df['Model Name'][0]
            y_true = df[f'target_{prediction_horizon}'][0]  # PH doesnt really matter that much
            y_true = ast.literal_eval(y_true)
            y_pred = df[f'y_pred_{prediction_horizon}'][0]
            y_pred = ast.literal_eval(y_pred)[1:]
            y_diffs = [j-i for i, j in zip(y_true[:-1], y_true[1:])]
            y_true = y_true[1:]  # Removing first element because we don't have corresponding diff for that value
            costs = [slope_cost(bg, delta_bg) + zone_cost(bg) + 1 for bg, delta_bg in zip(y_true, y_diffs)]
            # important to add 1 to the costs because we never want them to multiply errors with 0.0!
            # Print the new cost function result for the weighted RMSE
            print(f"{model_name} {prediction_horizon} minutes: ", round(weighted_rmse(y_true, y_pred, weights=costs), 1))


        # Plot the results across different glucose regions
        df = pd.read_csv('data/raw/OhioT1DM.csv')
        df_train = df[~df['is_test']]
        y_true = df_train['CGM']
        y_diffs = [j - i for i, j in zip(y_true[:-1], y_true[1:])]
        y_true = y_true[1:]  # Removing first element because we don't have corresponding diff for that value
        costs = [slope_cost(bg, delta_bg) + zone_cost(bg) + 1 for bg, delta_bg in zip(y_true, y_diffs)]
        bins = [0, 70, 180, 600]
        #bins = list(range(0, 301, 20))
        bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
        data = pd.DataFrame({'BG': y_true, 'Cost': costs})
        data['Bin'] = pd.cut(data['BG'], bins=bins, labels=bin_labels, right=False)
        summed_costs = data.groupby('Bin')['Cost'].sum()
        plt.figure(figsize=(10, 6))
        summed_costs.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Summed Costs for Glucose Ranges', fontsize=16)
        plt.xlabel('Glucose Range (mg/dL)', fontsize=14)
        plt.ylabel('Summed Costs', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        return plots, names


def weighted_rmse(y_true, y_pred, weights):
    """
    Calculate the weighted RMSE.

    Parameters:
    - y_true: Array of true target values.
    - y_pred: Array of predicted values.
    - weights: The weights to assign to the errors.

    Returns:
    - Weighted RMSE.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute weighted RMSE
    squared_errors = weights * (y_pred - y_true) ** 2
    weighted_rmse = np.sqrt(np.sum(squared_errors) / np.sum(weights))

    return weighted_rmse



def slope_cost(bg, delta_bg):
    k = 18.0182
    # This function assumes mmol/L
    bg = bg / k
    delta_bg = delta_bg / k

    a = bg
    b = 15 - bg
    if b < 0:
        b = 0
    if a > 15:
        a = 15

    cost = (np.sign(delta_bg) + 1) / 2 * a * (delta_bg ** 2) - 2 * (np.sign(delta_bg) - 1) / 2 * b * (delta_bg ** 2)
    return cost


def zone_cost(bg, target=105):
    if bg < 1:
        bg = 1
    if bg > 600:
        bg = 600

    # This function assumes BG in mg / dL
    constant = 32.9170208165394
    left_weight = 19.0
    right_weight = 1.0

    if bg < target:
        risk = constant * left_weight * (np.log(bg) - np.log(target)) ** 2
    else:
        risk = constant * right_weight * (np.log(bg) - np.log(target)) ** 2

    return risk



def original_zone_cost(bg, target=105):
    if bg < 1:
        bg = 1
    if bg > 600:
        bg = 600

    # This function assumes BG in mg / dL
    constant = 32.9170208165394
    risk = constant * (np.log(bg) - np.log(target)) ** 2

    return risk



#!/usr/bin/env python3
import click
import dill
import requests
import warnings
import os
import wandb
import ast
import importlib
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from datetime import timedelta, datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import matplotlib.pyplot as plt

# Modules from this repository
from .parsers.base_parser import BaseParser
from .plots.base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.helpers.model_config_manager import ModelConfigurationManager, generate_model_configuration
import glupredkit.helpers.cli as helpers
import glupredkit.helpers.generate_report as generate_report
import glupredkit.api as gpk


# TODO: Fix so that all default values are defined upstream (=here in the CLI), and removed from downstream


@click.command()
def setup_directories():
    """Set up necessary directories for GluPredKit."""
    cwd = os.getcwd()
    print("Creating directories...")

    folder_path = 'data'
    folder_names = ['raw', 'configurations', 'trained_models', 'tested_models', 'figures', 'reports']

    for folder_name in folder_names:
        path = os.path.join(cwd, folder_path, folder_name)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory {path}.")

    print("Directories created for usage of GluPredKit.")


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'tidepool_dataset', 'nightscout', 'apple_health',
                                             'ohio_t1dm', 'open_aps', 't1dexi']),
              help='Choose a parser', required=True)
@click.option('--username', type=str, required=False)
@click.option('--password', type=str, required=False)
@click.option('--start-date', type=str,
              help='Start date for data retrieval. Default is two weeks ago. Format "dd-mm-yyyy"')
@click.option('--file-path', type=str, required=False)
@click.option('--end-date', type=str, help='End date for data retrieval. Default is now. Format "dd-mm-yyyy"')
@click.option('--output-file-name', type=str, help='The file name for the output.')
@click.option('--test-size', type=float, default=0.25)
def parse(parser, username, password, start_date, file_path, end_date, output_file_name, test_size):
    """Parse data and store it as CSV in data/raw using a selected parser"""

    # Load the chosen parser dynamically based on user input
    parser_module = importlib.import_module(f'glupredkit.parsers.{parser}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(parser_module.Parser, BaseParser):
        raise click.ClickException(f"The selected parser '{parser}' must inherit from BaseParser.")

    # Create an instance of the chosen parser
    chosen_parser = parser_module.Parser()

    click.echo(f"Parsing data using {parser}...")

    date_format = "%d-%m-%Y"

    if not end_date:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, date_format)

    if not start_date:
        start_date = end_date - timedelta(days=14)
    else:
        start_date = datetime.strptime(start_date, date_format)

    def save_data(output_file_name, data):
        output_path = 'data/raw/'
        date_format = "%d-%m-%Y"

        if output_file_name:
            file_name = output_file_name + '.csv'
        else:
            file_name = (parser + '_' + start_date.strftime(date_format) + '_to_' + end_date.strftime(
                date_format) + '.csv')

        click.echo("Storing data as CSV...")
        helpers.store_data_as_csv(data, output_path, file_name)
        click.echo(f"Data stored as CSV at '{output_path}' as '{file_name}'")
        click.echo(f"Data has the shape: {data.shape}")

    # Ensure that the optional params match the parser
    if parser in ['tidepool', 'nightscout']:
        if username is None or password is None:
            raise ValueError(f"{parser} parser requires that you provide --username and --password")
        else:
            parsed_data = chosen_parser(start_date=start_date, end_date=end_date, username=username, password=password)
    elif parser in ['apple_health']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            parsed_data = chosen_parser(start_date=start_date, end_date=end_date,
                                        file_path=file_path)
    elif parser in ['ohio_t1dm']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            ids_2018 = ['559', '563', '570', '575', '588', '591']
            ids_2020 = ['540', '544', '552', '567', '584', '596']

            merged_df = pd.DataFrame()

            for subject_id in ids_2018:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2018')
                parsed_data['id'] = subject_id
                merged_df = pd.concat([parsed_data, merged_df], ignore_index=False)

            for subject_id in ids_2020:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2020')
                parsed_data['id'] = subject_id
                merged_df = pd.concat([parsed_data, merged_df], ignore_index=False)
            save_data(output_file_name="OhioT1DM", data=merged_df)

            return
    elif parser in ['t1dexi']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            folder_1 = os.path.join(file_path, 'T1DEXI.zip')
            parsed_data_1 = chosen_parser(file_path=folder_1)
            parsed_data_1 = helpers.add_is_test_column(parsed_data_1, test_size)
            parsed_data_1['source_file'] = 'T1DEXI'
            save_data(output_file_name="T1DEXI", data=parsed_data_1)

            folder_2 = os.path.join(file_path, 'T1DEXIP.zip')
            parsed_data_2 = chosen_parser(file_path=folder_2)
            parsed_data_2 = helpers.add_is_test_column(parsed_data_2, test_size)
            parsed_data_2['source_file'] = 'T1DEXIP'
            save_data(output_file_name="T1DEXIP", data=parsed_data_2)
            return
    elif parser in ['open_aps']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            parsed_data = chosen_parser(file_path=file_path)
            output_file_name = "OpenAPS"
            parsed_data['source_file'] = output_file_name
    elif parser in ['tidepool_dataset']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            for prefix in ['HCL150', 'SAP100', 'PA50']:
                parsed_data = chosen_parser(prefix, file_path=file_path)
                output_file_name = f"Tidepool-JDRF-{prefix}"
                parsed_data['source_file'] = output_file_name
                save_data(output_file_name=output_file_name, data=parsed_data)

            # Already split into train and test data
            return
    else:
        raise ValueError(f"unrecognized parser: '{parser}'")

    # Train and test split
    parsed_data = helpers.add_is_test_column(parsed_data, test_size)
    save_data(output_file_name=output_file_name, data=parsed_data)


@click.command()
@click.option('--file-name', help='Give a file name to the configuration file (e.g., "my_config").',
              callback=helpers.validate_config_file_name, required=True)
@click.option('--data', help='Name of the input data file (e.g., "data.csv"). '
                             'This file should be located in data/raw/.',
              callback=helpers.check_if_data_file_exists, required=True)
@click.option('--subject-ids', help='The ids you want to include in the model training and testing. '
                                    ' will include all of the subjects.', callback=helpers.validate_subject_ids,
              required=False, default='')
@click.option('--preprocessor', required=False, default='basic', type=click.Choice([
    'basic',
    'standardscaler'
]))
@click.option('--prediction-horizon', help='Integer for prediction horizon in minutes.',
              callback=helpers.validate_prediction_horizon, required=True)
@click.option('--num-lagged-features', help='Number of lagged features.',
              callback=helpers.validate_num_lagged_features, required=True)
@click.option('--num-features', help='Comma-separated list of numerical features.',
              callback=helpers.validate_feature_list, required=True, default='CGM')
@click.option('--cat-features', help='Comma-separated list of categorical features.',
              callback=helpers.validate_feature_list, required=False, default='')
@click.option('--what-if-features', help='Comma-separated list of categorical features.',
              callback=helpers.validate_feature_list, required=False, default='')
def generate_config(file_name, data, subject_ids, preprocessor, prediction_horizon, num_lagged_features, num_features,
                    cat_features, what_if_features):

    if data == 'synthetic_data':
        cwd = os.getcwd()
        url = 'https://raw.githubusercontent.com/miriamkw/GluPredKit/main/example_data/synthetic_data.csv'
        save_folder = 'data/raw/'
        save_path = os.path.join(cwd, save_folder, 'synthetic_data.csv')

        if not os.path.exists(save_path):
            response = requests.get(url)
            with open(save_path, 'wb') as file:
                file.write(response.content)

            click.echo(f"Synthetic data saved to {save_path}")

    generate_model_configuration(file_name, data, subject_ids, preprocessor, int(prediction_horizon),
                                 int(num_lagged_features), num_features, cat_features, what_if_features)
    click.echo(f"Storing configuration file to data/configurations/{file_name}...")
    click.echo(f"Note that it might take a minute before the file appears in the folder.")


@click.command()
@click.argument('config-file-name', type=str)
@click.option('--model', type=click.Choice([
    'double_lstm',
    'loop',
    'loop_v2',
    'lstm',
    'mtl',
    'naive_linear_regressor',
    'random_forest',
    'ridge',
    'weighted_ridge',
    'pytorch_ridge',
    'stacked_plsr',
    'stl',
    'svr',
    'tcn',
    'uva_padova',
    'zero_order'
]), help="Choose a pre-defined model")
@click.option('--model-path', type=click.Path(exists=True), help="Provide the path to your custom model file")
@click.option('--epochs', type=int, required=False)
@click.option('--n-cross-val-samples', type=int, required=False)
@click.option('--n-steps', type=int, required=False)
@click.option('--training-samples-per-subject', type=int, required=False)
@click.option('--model-name', type=str, required=False)
@click.option('--max-samples', type=int, required=False)
def train_model(config_file_name, model, model_name, model_path, epochs, n_cross_val_samples, n_steps,
                training_samples_per_subject, max_samples):
    """
    This method does the following:
    1) Process data using the given configurations
    2) Train the given models for the given prediction horizons in the configuration
    """
    if not model and not model_path:
        raise click.UsageError("You must specify either --model or --model-path.")
    if model and model_path:
        raise click.UsageError("You can specify only one: either --model or --model-path.")

    if model:
        click.echo(f"Using pre-defined model: {model}")
        model_module = helpers.get_model_module(model=model)
        if not model_name:
            model_name = model
    else:
        click.echo(f"Using custom model from: {model_path}")
        model_module = helpers.get_model_module(model_path=model_path)
        if not model_name:
            model_name = model_path.split('/')[-1].split('.')[0]

    # Filtering out UserWarning because we are using an old Keras file format on purpose
    warnings.filterwarnings(
        "ignore",
        message=".*You are saving your model as an HDF5 file.*"
    )

    click.echo(f"Starting pipeline to train model {model} with configurations in {config_file_name}...")
    model_config_manager = ModelConfigurationManager(config_file_name)
    prediction_horizon = model_config_manager.get_prediction_horizon()

    # PREPROCESSING
    # Perform data preprocessing using your preprocessor
    input_file_name = model_config_manager.get_data()
    data = helpers.read_data_from_csv("data/raw/", input_file_name)
    data = data[~data['is_test']]

    if max_samples:
        data = data.tail(max_samples + model_config_manager.get_num_lagged_features() + (prediction_horizon // 5))

    train_data, _ = helpers.get_preprocessed_data(data, prediction_horizon, model_config_manager)
    click.echo(f"Training data finished preprocessing...")

    # MODEL TRAINING
    # Create an instance of the chosen model
    chosen_model = model_module.Model(prediction_horizon)

    processed_data = chosen_model.process_data(train_data, model_config_manager, real_time=False)
    target_columns = [column for column in processed_data.columns if column.startswith('target')]
    x_train = processed_data.drop(target_columns, axis=1)
    y_train = processed_data[target_columns]

    click.echo(f"Training model...")

    # Initialize and train the model
    # Ensure that the optional params match the parser
    if model in ['double_lstm', 'lstm', 'mtl', 'stl', 'tcn'] and epochs:
        model_instance = chosen_model.fit(x_train, y_train, epochs)
    elif model in ['loop'] and n_cross_val_samples:
        model_instance = chosen_model.fit(x_train, y_train, n_cross_val_samples)
    elif model in ['uva_padova'] and n_steps or training_samples_per_subject:
        model_instance = chosen_model.fit(x_train, y_train, n_steps, training_samples_per_subject)
    else:
        model_instance = chosen_model.fit(x_train, y_train)

    click.echo(f"Model {model} with prediction horizon {prediction_horizon} minutes trained successfully!")

    # Assuming model_instance is your class instance
    output_dir = Path("data") / "trained_models"
    output_file_name = f'{model_name}__{config_file_name}__{prediction_horizon}.pkl'
    output_path = output_dir / output_file_name

    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the model to a file
        with open(output_path, 'wb') as f:
            click.echo(f"Saving model {model} to {output_path}...")
            dill.dump(model_instance, f)
    except Exception as e:
        click.echo(f"Error saving model {model}: {e}")

    # Optionally print model hyperparameters if they exist
    if hasattr(model_instance, 'best_params'):
        click.echo(f"Model hyperparameters: {model_instance.best_params()}")


@click.command()
@click.argument('model_file', type=str)
@click.option('--max-samples', type=int, required=False)
def evaluate_model(model_file, max_samples):
    tested_models_path = "data/tested_models"
    model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                        int(model_file.split('__')[2].split('.')[0]))

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(model_file)

    input_file_name = model_config_manager.get_data()
    data = helpers.read_data_from_csv("data/raw/", input_file_name)
    test_data = data[data['is_test']]
    training_data = data[~data['is_test']]
    _, test_data = helpers.get_preprocessed_data(test_data, prediction_horizon, model_config_manager)

    test_data = model_instance.process_data(test_data, model_config_manager, real_time=False)
    target_cols = [col for col in test_data if col.startswith('target')]

    if max_samples:
        test_data = test_data[-max_samples:]
        x_test = test_data.drop(target_cols, axis=1)
    else:
        x_test = test_data.drop(target_cols, axis=1)
    y_pred = model_instance.predict(x_test)

    results_df = gpk.get_results_df(model_name, training_data, test_data, y_pred, prediction_horizon,
                                    num_lagged_features=model_config_manager.get_num_lagged_features(),
                                    num_features=model_config_manager.get_num_features(),
                                    cat_features=model_config_manager.get_cat_features(),
                                    what_if_features=model_config_manager.get_what_if_features())

    # Define the path to store the dataframe
    model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                        int(model_file.split('__')[2].split('.')[0]))
    output_file = f"{tested_models_path}/{model_name}__{config_file_name}__{prediction_horizon}.csv"

    # Store the dataframe in a file
    results_df.to_csv(output_file, index=False)
    click.echo(f"Model {model_name} is finished testing. Results are stored in {tested_models_path}")


@click.command()
@click.option('--results-files', help='The name of the tested model results to evaluate, with ".csv". If '
                                      'None, all models will be tested.')
@click.option('--plots', help='List of plots to be computed, separated by comma. '
                              'By default a scatter plot will be drawn. ', default='scatter_plot')
@click.option('--prediction-horizons', help='Integer for prediction horizons in minutes. Comma-separated'
                                            'without space.', default=None)
@click.option('--type', type=click.Choice(['parkes', 'clarke']), default='parkes',
              help='The type of error grid evaluation method to use.')
@click.option('--show-plots', type=bool, default=True)
@click.option('--metric', type=click.Choice(['mean_error', 'rmse']), default='mean_error',
              help='The type of metric to use in results across regions.')
@click.option('--wandb-project', help='Optional logging of the plot to a wandb project.', default=None)
def draw_plots(results_files, plots, prediction_horizons, type, show_plots, metric, wandb_project):
    """
    This command draws the given plots and store them in data/figures/.
    """
    # Prepare a list of plots
    plots = helpers.split_string(plots)

    if results_files is None:
        results_files = helpers.list_files_in_directory('data/tested_models/')
    else:
        results_files = helpers.split_string(results_files)

    dfs = []
    for results_file in results_files:
        df = generate_report.get_df_from_results_file(results_file)
        dfs += [df]

    # Set global params
    plt.rcParams.update({
        "font.size": 18,  # Default font size for all text
        "axes.titlesize": 18,  # Title size
        "axes.labelsize": 16,  # Axis label size
        "xtick.labelsize": 16,  # X-tick label size
        "ytick.labelsize": 16,  # Y-tick label size
        "legend.fontsize": 14,  # Legend font size
        "figure.titlesize": 20  # Figure title size
    })

    click.echo(f"Drawing plots...")

    if prediction_horizons is None:
        prediction_horizons = '30'
    prediction_horizons = helpers.split_string(prediction_horizons)

    figures = []
    names = []
    for plot in plots:
        plot_module = importlib.import_module(f'glupredkit.plots.{plot}')
        if not issubclass(plot_module.Plot, BasePlot):
            raise click.ClickException(f"The selected plot '{plot}' must inherit from BasePlot.")
        chosen_plot = plot_module.Plot()

        if plot in ['all_metrics_table', 'cgpm_table', 'confusion_matrix',
                    'pareto_frontier', 'scatter_plot', 'single_prediction_horizon',
                    'weighted_loss', 'cgpm_components_single_ph']:
            for prediction_horizon in prediction_horizons:
                plts, plot_names = chosen_plot(dfs, show_plot=show_plots, prediction_horizon=prediction_horizon)
                figures += plts
                names += plot_names

        elif plot in ['error_grid_plot', 'error_grid_table']:
            for prediction_horizon in prediction_horizons:
                plts, plot_names = chosen_plot(dfs, show_plot=show_plots, prediction_horizon=prediction_horizon, type=type)
                figures += plts
                names += plot_names

        elif plot in ['results_across_regions']:
            for prediction_horizon in prediction_horizons:
                plts, plot_names = chosen_plot(dfs, show_plot=show_plots, prediction_horizon=prediction_horizon, metric=metric)
                figures += plts
                names += plot_names

        elif plot in ['trajectories', 'trajectories_with_events']:
            plts, plot_names = chosen_plot(dfs, show_plot=show_plots)
            figures += plts
            names += plot_names

        else:
            raise ValueError(f"Plot {plot} does not exist. Please look in the documentation for the existing plots.")

    gpk.save_figures(figures, names)

    if wandb_project:
        gpk.log_figures_in_wandb(wandb_project, figures, names)


@click.command()
@click.option('--results-file', help='The name of the tested model results to evaluate, with ".csv".',
              required=True)
def generate_evaluation_pdf(results_file):
    """
    This command stores a standardized pdf report of the given model in data/reports/.
    """
    click.echo(f"Generating evaluation report...")
    df = generate_report.get_df_from_results_file(results_file)

    # Convert results to DataFrame and save as CSV
    results_file_path = 'data/reports/'
    os.makedirs(results_file_path, exist_ok=True)
    results_file_name = f'{results_file.split(".")[0]}.pdf'

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # FRONT PAGE
    c = generate_report.set_title(c, f'Model Evaluation for {df["Model Name"][0]}')
    c = generate_report.generate_single_model_front_page(c, df)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # MODEL ACCURACY
    c = generate_report.set_title(c, f'Model Accuracy')
    c = generate_report.draw_model_accuracy_table(c, df)
    plot_height = 1.3  # TODO: Dynamically scale when table is larger
    c = generate_report.plot_across_prediction_horizons(c, df, f'RMSE [{unit_config_manager.get_unit()}]',
                                                        ['rmse'], y_placement=420, height=plot_height)
    c = generate_report.plot_across_prediction_horizons(c, df, f'ME [{unit_config_manager.get_unit()}]',
                                                        ['me'], y_placement=260, height=plot_height)
    c = generate_report.plot_across_prediction_horizons(c, df, f'MARE [%]',
                                                        ['mare'], y_placement=100, height=plot_height)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    def round_to_nearest_5(number, divisor):
        if number <= 5:
            return 5
        result = number / divisor
        rounded_result = 5 * round(result / 5)
        return rounded_result

    prediction_horizon = generate_report.get_ph(df)
    if prediction_horizon < 20:
        ph_subparts_list = list(range(5, prediction_horizon + 5, 5))
    else:
        ph_quarter = round_to_nearest_5(prediction_horizon, 4)
        ph_subparts_list = [ph_quarter, ph_quarter * 2, ph_quarter * 3, prediction_horizon]

    c = generate_report.set_title(c, f'Error Grid Analysis')
    c = generate_report.draw_error_grid_table(c, df)

    c = generate_report.draw_scatter_plot(c, df, ph_subparts_list[0], 100, 360)
    c = generate_report.draw_scatter_plot(c, df, ph_subparts_list[1], 380, 360)

    if len(ph_subparts_list) > 2:
        c = generate_report.draw_scatter_plot(c, df, ph_subparts_list[2], 100, 120)
    if len(ph_subparts_list) > 3:
        c = generate_report.draw_scatter_plot(c, df, prediction_horizon, 380, 120)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION
    c = generate_report.set_title(c, f'Glycemia Detection - Matthews Correlation Coefficient (MCC)')
    c = generate_report.draw_mcc_table(c, df)
    c = generate_report.plot_across_prediction_horizons(c, df, f'Matthews Correlation Coefficient (MCC)',
                                                        ['mcc_hypo', 'mcc_hyper'], y_placement=150,
                                                        height=4, y_label='MCC',
                                                        y_labels=['MCC Hypoglycemia', 'MCC Hyperglycemia'])
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION PAGE 2
    c = generate_report.set_title(c, f'Glycemia Detection - Confusion Matrices')

    # Plot confusion matrixes
    classes = ['Hypo', 'Target', 'Hyper']

    c = generate_report.plot_confusion_matrix(c, df, classes, ph_subparts_list[0], 50, 450)
    c = generate_report.plot_confusion_matrix(c, df, classes, ph_subparts_list[1], 320, 450)

    if len(ph_subparts_list) > 2:
        c = generate_report.plot_confusion_matrix(c, df, classes, ph_subparts_list[2], 50, 150)
    if len(ph_subparts_list) > 3:
        c = generate_report.plot_confusion_matrix(c, df, classes, prediction_horizon, 320, 150)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PHYSIOLOGICAL ALIGNMENT PAGE
    c = generate_report.set_title(c, f'Physiological Alignment')
    c = generate_report.set_subtitle(c, f'Physiological Alignment for insulin', 720)
    c = generate_report.draw_physiological_alignment_single_dimension_table(c, df, 'bolus', 670)
    c = generate_report.plot_partial_dependencies_across_prediction_horizons(c, df, 'bolus', y_placement=440)

    c = generate_report.set_subtitle(c, f'Physiological Alignment for carbohydrates', 360)
    c = generate_report.draw_physiological_alignment_single_dimension_table(c, df, 'carbs', 310)
    c = generate_report.plot_partial_dependencies_across_prediction_horizons(c, df, 'carbs', y_placement=100)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PREDICTION DISTRIBUTION PAGE
    c = generate_report.set_title(c, f'Distribution of Predictions')
    c = generate_report.plot_predicted_distribution(c, df, 100, 400)

    # Show the page
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(f"An evaluation report for {results_file} is stored in '{results_file_path}' as '{results_file_name}'")


@click.command()
@click.option('--results-files', help='The name of the tested model results to evaluate, with ".csv". If '
                                      'None, all models will be tested.')
def generate_comparison_pdf(results_files):
    """
    This command stores a standardized pdf comparison report of the given models in data/reports/.
    """
    click.echo(f"Generating comparison report...")

    if results_files is None:
        results_files = helpers.list_files_in_directory('data/tested_models/')
    else:
        results_files = helpers.split_string(results_files)

    results_file_path = "data/reports/"

    timestamp = datetime.now().isoformat()
    safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
    safe_timestamp = safe_timestamp.replace('.', '_')

    results_file_name = f'comparison_report_{safe_timestamp}.pdf'

    dfs = []
    for results_file in results_files:
        df = generate_report.get_df_from_results_file(results_file)
        dfs += [df]

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # FRONT PAGE
    c = generate_report.set_title(c, f'Model Comparison Report')
    # TODO: Add front page summary of all parts and all the configurations
    c = generate_report.draw_overall_ranking_table(c, dfs, y_placement=700 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # MODEL ACCURACY
    c = generate_report.set_title(c, f'Model Accuracy')
    c = generate_report.draw_model_comparison_accuracy_table(c, dfs, 'rmse', 700 - 20 * len(dfs))
    #c = generate_report.draw_model_comparison_accuracy_table(c, dfs, 'me', 550 - 20 * len(dfs))
    c = generate_report.plot_rmse_across_prediction_horizons(c, dfs, y_placement=200)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    c = generate_report.set_title(c, f'Error Grid Analysis')
    c = generate_report.draw_model_comparison_error_grid_table(c, dfs, 700 - 20 * len(dfs))
    c = generate_report.plot_error_grid_across_prediction_horizons(c, dfs, y_placement=400 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION
    c = generate_report.set_title(c, f'Glycemia Detection')
    # TODO: Draw both in the table, and add a total score. Plot the total score for each model.
    c = generate_report.draw_model_comparison_glycemia_detection_table(c, dfs, 700 - 20 * len(dfs))
    c = generate_report.plot_mcc_across_prediction_horizons(c, dfs, y_placement=400 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PREDICTED DISTRIBUTION
    c = generate_report.set_title(c, f'Predicted Distribution')
    # TODO: Draw both in the table, and add a total score. Plot the total score for each model.
    c = generate_report.draw_model_comparison_predicted_distribution_table(c, dfs, 700 - 20 * len(dfs))
    c = generate_report.plot_predicted_dristribution_across_prediction_horizons(c, dfs, y_placement=400 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(f"An evaluation report for {results_files} is stored in '{results_file_path}' as '{results_file_name}'")


@click.command()
@click.option('--use-mgdl', type=bool, help='Set whether to use mg/dL or mmol/L', default=None)
def set_unit(use_mgdl):
    # Update the config
    unit_config_manager.use_mgdl = use_mgdl
    click.echo(f'Set unit to {"mg/dL" if use_mgdl else "mmol/L"}.')


# Create a Click group and add the commands to it
cli = click.Group(commands={
    'setup_directories': setup_directories,
    'parse': parse,
    'generate_config': generate_config,
    'train_model': train_model,
    'evaluate_model': evaluate_model,
    'draw_plots': draw_plots,
    'generate_evaluation_pdf': generate_evaluation_pdf,
    'generate_comparison_pdf': generate_comparison_pdf,
    'set_unit': set_unit,
})

if __name__ == "__main__":
    cli()

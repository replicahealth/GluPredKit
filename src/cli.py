import click
import dill
import os
import importlib
import pandas as pd
from parsers.base_parser import BaseParser
from preprocessors.base_preprocessor import BasePreprocessor
from models.base_model import BaseModel
from metrics.base_metric import BaseMetric
from datetime import timedelta, datetime


def read_data_from_csv(input_path, file_name):
    file_path = input_path + file_name
    return pd.read_csv(file_path, index_col="date", parse_dates=True)


def store_data_as_csv(df, output_path, file_name):
    file_path = output_path + file_name
    df.to_csv(file_path)


def split_string(input_string):
    return [] if not input_string else input_string.split(',')


# TODO: Fix so that all default values are defined upstream (=here in the CLI), and removed from downstream


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'nightscout']), help='Choose a parser')
@click.argument('username', type=str)
@click.argument('password', type=str)
@click.option('--file-name', type=str, help='Optional file name of output')
@click.option('--start-date', type=str,
              help='Start date for data retrieval. Default is two weeks ago. Format "dd-mm-yyyy"')
@click.option('--end-date', type=str,
              help='End date for data retrieval. Default is now. Format "dd-mm-yyyy"')
def parse(parser, username, password, file_name, start_date, end_date):
    """Parse data and store it as CSV in data/raw using a selected parser"""

    # Load the chosen parser dynamically based on user input
    parser_module = importlib.import_module(f'parsers.{parser}')

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

    # Perform parsing using the chosen parser
    parsed_data = chosen_parser(start_date, end_date, username, password)

    output_path = '../data/raw/'
    date_format = "%d-%m-%Y"

    # Add default file name if input is not provided
    if file_name is not None:
        file_name = file_name
    else:
        file_name = (parser + '_' + start_date.strftime(date_format) + '_to_' + end_date.strftime(date_format)
                     + '.csv')

    click.echo("Storing data as CSV...")
    store_data_as_csv(parsed_data, output_path, file_name)
    click.echo(f"Data stored as CSV at '{output_path}' as '{file_name}'")
    click.echo(f"Data has the shape: {parsed_data.shape}")


@click.command()
@click.option('--preprocessor', type=click.Choice(['scikit_learn']), default='scikit_learn',
              help='Choose a preprocessor (default: scikit_learn)')
# TODO: Make the list of parsers dynamic to the files in the parsers folder
@click.argument('input-file-name', type=str)
@click.option('--prediction-horizon', type=int, default=60)
@click.option('--num-lagged-features', type=int, default=12,
              help='The number of samples of time-lagged features (default: 12).')
@click.option('--include-hour', type=bool, default=True,
              help='Include hour of day as an input feature (default: True).')
@click.option('--test-size', type=float, default=0.2,
              help='Fraction of data to reserve for testing (default: 0.2).')
def preprocess(preprocessor, input_file_name, prediction_horizon, num_lagged_features, include_hour, test_size):
    """
    Preprocess data from an input CSV file and store train and test data into CSV files.

    Args:
        preprocessor (str): Type of preprocessor from the preprocessor module.
        input_file_name (str): Input CSV file containing the data.
        prediction_horizon (int): The prediction horizon for the target value in minutes.
        num_lagged_features (int): The number of samples of time-lagged features.
        include_hour (bool): Whether to include hour of day as an input feature.
        test_size (float): Fraction of data to reserve for testing.
    """
    if prediction_horizon % 5 != 0:
        raise click.BadParameter('Prediction horizon must be divisible by 5.')

    # Load the chosen parser dynamically based on user input
    preprocessor_module = importlib.import_module(f'preprocessors.{preprocessor}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(preprocessor_module.Preprocessor, BasePreprocessor):
        raise click.ClickException(f"The selected preprocessor '{preprocessor}' must inherit from BasePreprocessor.")

    # Create an instance of the chosen parser
    chosen_preprocessor = preprocessor_module.Preprocessor()

    input_path = "../data/raw/"
    click.echo(f"Preprocessing data using {preprocessor} from file {input_path}{input_file_name}...")

    # Load the input CSV file into a DataFrame
    data = read_data_from_csv(input_path, input_file_name)

    # Perform data preprocessing using your preprocessor
    train_data, test_data = chosen_preprocessor(data, prediction_horizon, num_lagged_features, include_hour, test_size)

    # Define output file names
    output_path = "../data/processed/"
    train_output_file = f"train-data_{preprocessor}_ph-{prediction_horizon}_lag-{num_lagged_features}.csv"
    test_output_file = f"test-data_{preprocessor}_ph-{prediction_horizon}_lag-{num_lagged_features}.csv"

    # Store train and test data as CSV files
    store_data_as_csv(train_data, output_path, train_output_file)
    store_data_as_csv(test_data, output_path, test_output_file)

    click.echo(f"Train data saved as '{train_output_file}', with shape {train_data.shape}")
    click.echo(f"Test data saved as '{test_output_file}', with shape {test_data.shape}")


@click.command()
@click.option('--model', prompt='Model name', help='Name of the model file (without .py) to be trained.')
@click.argument('input-file-name', type=str)
@click.option('--prediction-horizon', type=int, default=60)
@click.option('--num_features', default='CGM,insulin,carbs', help='List of numerical features, separated by comma.')
@click.option('--cat_features', default='', help='List of categorical features, separated by comma.')
def train_model(model, input_file_name, prediction_horizon, num_features, cat_features):
    # Convert comma-separated string of features to list
    num_features = split_string(num_features)
    cat_features = split_string(cat_features)

    # Load the chosen parser dynamically based on user input
    model_module = importlib.import_module(f'models.{model}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(model_module.Model, BaseModel):
        raise click.ClickException(f"The selected model '{model}' must inherit from BaseModel.")

    # Create an instance of the chosen parser
    chosen_model = model_module.Model(prediction_horizon, num_features, cat_features)

    input_path = "../data/processed/"
    click.echo(f"Training model {model} with training data from {input_path}{input_file_name}...")

    # Load the input CSV file into a DataFrame
    train_data = read_data_from_csv(input_path, input_file_name)
    x_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    # Initialize and train the model
    model_instance = chosen_model.fit(x_train, y_train)

    # Assuming model_instance is your class instance
    output_path = "../data/models/"
    output_file_name = f'{model}_ph-{prediction_horizon}.pkl'
    with open(f'{output_path}{output_file_name}', 'wb') as f:
        dill.dump(model_instance, f)

    click.echo(f"Model {model} trained successfully!")

    if hasattr(model_instance, 'best_params'):
        click.echo(f"Model hyperparameters: {model_instance.best_params()}")


# TODO: Handle different units (REMEMBER TO TEST)
# TODO: add example plots (SEG, trajectories)
# TODO: add support for plot here
# TODO: add possibility list of models (default all???)
# TODO: add possibility list of metrics (and store in one report)  (default all???)
# TODO: add possibility list of plots (and store in figures)
# TODO: Add documentation in readme
@click.command()
@click.option('--model-file', prompt='Model file name',
              help='Name of the file (with .pkl) containing a specific trained model. '
                   'By default all models will be included. ')
@click.option('--metric', prompt='Metric name', help='Name of a specific metric to be computed. '
                                                     'By default all metrics will be computed. ')
@click.argument('test-file-name', type=str)
def evaluate_model(model_file, metric, test_file_name):
    model_path = "../data/models/"

    # Ensure the model file exists
    if not os.path.exists(model_path + model_file):
        raise click.ClickException(f"Model file {model_file} does not exist.")

    # Load the model
    with open(model_path + model_file, 'rb') as f:
        model_instance = dill.load(f)

    # Load the test data
    test_file_path = "../data/processed/"
    test_data = read_data_from_csv(test_file_path, test_file_name)  # Adjust as per your actual logic to load data
    x_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    # Get predictions
    y_pred = model_instance.predict(x_test)

    # Load the chosen metric dynamically based on user input
    metric_module = importlib.import_module(f'metrics.{metric}')

    # Ensure the chosen metric inherits from BaseMetric
    if not issubclass(metric_module.Metric, BaseMetric):
        raise click.ClickException(f"The selected metric '{metric}' must inherit from BaseMetric.")

    chosen_metric = metric_module.Metric()
    score = chosen_metric(y_test, y_pred)

    # TODO: Store metrics in reports
    click.echo(f"{metric} for model {model_file}: {score}")


if __name__ == "__main__":
    # Create a Click group and add the commands to it
    cli = click.Group(commands={
        'parse': parse,
        'preprocess': preprocess,
        'train_model': train_model,
        'evaluate_model': evaluate_model,
    })

    cli()

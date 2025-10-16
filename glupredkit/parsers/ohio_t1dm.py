"""
The Ohio T1DM parser is processing the raw .xml data from the Ohio T1DM datasets and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import datetime


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, subject_id: str, year: str, *args):
        """
        file_path -- the file path to the OhioT1DM dataset root folder.
        subject_id -- the id of the subject.
        year -- the version year for the dataset.
        """
        self.validate_year(year)

        training_tree = ET.parse(os.path.join(file_path, 'OhioT1DM', year, 'train', f'{subject_id}-ws-training.xml'))
        testing_tree = ET.parse(os.path.join(file_path, 'OhioT1DM', year, 'test', f'{subject_id}-ws-testing.xml'))

        df_training = self.resample_data(training_tree, is_test=False)
        df_testing = self.resample_data(testing_tree, is_test=True)

        # Use "combine first" to handle when df training and df testing is overlapping If df_test has a NaN for a
        # particular index/column, it takes the value from df_train. If df_test has a valid value, it keeps it.
        merged_df = df_testing.combine_first(df_training)
        merged_df = merged_df.sort_index()

        # Add gender and insulin delivery device, gotten from the paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/
        merged_df['gender'] = get_gender(subject_id)
        merged_df['insulin_delivery_device'] = get_insulin_delivery_device(subject_id)

        # Ensure 5-min intervals after merging train and test
        merged_df.sort_index(inplace=True)
        merged_df = merged_df.resample('5min').asfreq()
        time_diffs = merged_df.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=5)
        valid_intervals = (time_diffs[1:] == expected_interval).all()
        if not valid_intervals:
            invalid_intervals = time_diffs[time_diffs != expected_interval]
            print(f"invalid time intervals found:", invalid_intervals)

        return merged_df

    def resample_data(self, tree, is_test):
        root = tree.getroot()

        dataframes = {}

        for child in root:
            tag_name = child.tag
            events = []
            for event in child.findall('event'):
                events.append(event.attrib)
            dataframes[tag_name] = pd.DataFrame(events)

        # Resampling all datatypes into the same time-grid
        df = dataframes['glucose_level'].copy()

        df['ts'] = pd.to_datetime(df['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.rename(columns={'value': 'CGM', 'ts': 'date'}, inplace=True)
        df.set_index('date', inplace=True)
        df = df.resample('5min', label='right').mean()

        # Carbohydrates
        df_carbs = dataframes['meal'].copy()
        if not df_carbs.empty:
            df_carbs['ts'] = pd.to_datetime(df_carbs['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
            df_carbs['carbs'] = pd.to_numeric(df_carbs['carbs'], errors='coerce')
            df_carbs.rename(columns={'ts': 'date'}, inplace=True)
            df_carbs = df_carbs[['date', 'carbs']]
            df_carbs.set_index('date', inplace=True)
            df_carbs = df_carbs.resample('5min', label='right').sum()
            df = pd.merge(df, df_carbs, on="date", how='outer')
        else:
            df['carbs'] = np.nan

        # Bolus doses
        df_bolus = dataframes['bolus'].copy()
        df_bolus['ts_begin'] = pd.to_datetime(df_bolus['ts_begin'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_bolus['dose'] = pd.to_numeric(df_bolus['dose'], errors='coerce')
        df_bolus.rename(columns={'ts_begin': 'date', 'dose': 'bolus'}, inplace=True)
        df_bolus = df_bolus[['date', 'bolus']]
        df_bolus.set_index('date', inplace=True)
        df_bolus = df_bolus.resample('5min', label='right').sum()
        df = pd.merge(df, df_bolus, on="date", how='outer')
        df['bolus'] = df['bolus']

        # Basal rates
        df_basal = dataframes['basal'].copy()
        df_basal['ts'] = pd.to_datetime(df_basal['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_basal['value'] = pd.to_numeric(df_basal['value'])
        df_basal.rename(columns={'ts': 'date', 'value': 'basal'}, inplace=True)
        df_basal = df_basal[['date', 'basal']]
        df_basal.set_index('date', inplace=True)
        df_basal = df_basal.resample('5min', label='right').last().ffill()

        # Temp basal rates
        df_temp_basal = dataframes['temp_basal'].copy()
        if not df_temp_basal.empty:
            df_temp_basal['ts_begin'] = pd.to_datetime(df_temp_basal['ts_begin'], format='%d-%m-%Y %H:%M:%S',
                                                       errors='coerce')
            df_temp_basal['ts_end'] = pd.to_datetime(df_temp_basal['ts_end'], format='%d-%m-%Y %H:%M:%S',
                                                     errors='coerce')
            # Override the basal rates with the temp basal rate data
            for index, row in df_temp_basal.iterrows():
                def round_down_date_time(dt):
                    delta_min = dt.minute % 5
                    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute - delta_min)

                # Convert dates to nearest five minutes
                start_date = round_down_date_time(row['ts_begin'])
                end_date = round_down_date_time(row['ts_end'])
                value = row['value']
                df_basal.loc[start_date:end_date] = float(value)

        # Merge basal into dataframe
        df_basal['basal'] = pd.to_numeric(df_basal['basal'], errors='coerce')
        df = pd.merge(df, df_basal, on="date", how='outer')
        df['basal'] = df['basal'].ffill()

        df = merge_data_type_into_dataframe(df, dataframes, 'basis_heart_rate', 'heartrate')
        df = merge_data_type_into_dataframe(df, dataframes, 'basis_gsr', 'galvanic_skin_response')
        df = merge_data_type_into_dataframe(df, dataframes, 'basis_skin_temperature', 'skin_temp')
        if 'basis_air_temperature' in dataframes:
            df = merge_data_type_into_dataframe(df, dataframes, 'basis_air_temperature', 'air_temp')
        if 'basis_steps' in dataframes:
            df = merge_data_type_into_dataframe(df, dataframes, 'basis_steps', 'steps', use_mean=False)
        if 'acceleration' in dataframes:
            df = merge_data_type_into_dataframe(df, dataframes, 'acceleration', 'acceleration')

        # Exercise
        df_exercise = dataframes['exercise'].copy()
        if not df_exercise.empty:
            df_exercise['ts'] = pd.to_datetime(df_exercise['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
            df_exercise['intensity'] = pd.to_numeric(df_exercise['intensity'], errors='coerce')
            df_exercise['duration'] = pd.to_numeric(df_exercise['duration'], errors='coerce')
            df_exercise.rename(columns={'ts': 'date', 'intensity': 'workout_intensity', 'duration': 'workout_duration'}, inplace=True)
            df_exercise.set_index('date', inplace=True)
            df_exercise = df_exercise[['workout_intensity', 'workout_duration']]
            df_exercise = df_exercise.resample('5min', label='right').mean()
            df = pd.merge(df, df_exercise, on="date", how='outer')

        df['basal'] = df['basal'] / 12  # From U/hr to U
        df['insulin'] = df['basal'] + df['bolus'].fillna(0)

        df['is_test'] = is_test
        df['insulin_type_bolus'] = root.get('insulin_type').split(' ')[0].lower()
        df['insulin_type_basal'] = root.get('insulin_type').split(' ')[0].lower()

        df['cgm_device'] = 'Medtronic Enlite'
        df['insulin_delivery_modality'] = 'SAP'
        df['source_file'] = 'OhioT1DM'

        return df.sort_index()

    @staticmethod
    def validate_year(year):
        if year not in ['2018', '2020']:
            raise ValueError('The input year must be either 2018 or 2020.')

    @staticmethod
    def parse_xml_file(base_path, dataset_type, subject_id, year):
        file_name = f"{subject_id}-ws-{dataset_type}.xml"
        file_path = os.path.join(base_path, 'OhioT1DM', year, dataset_type, file_name)
        return ET.parse(file_path)


def merge_data_type_into_dataframe(df, data, type_name, value_name, use_mean=True):
    df_data_type = data[type_name].copy()
    if not df_data_type.empty:
        df_data_type['ts'] = pd.to_datetime(df_data_type['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_data_type['value'] = pd.to_numeric(df_data_type['value'], errors='coerce')
        df_data_type.rename(columns={'ts': 'date', 'value': value_name}, inplace=True)
        df_data_type.set_index('date', inplace=True)
        if use_mean:
            df_data_type = df_data_type.resample('5min', label='right').mean()
        else:
            df_data_type = df_data_type.resample('5min', label='right').sum()
        return pd.merge(df, df_data_type, on="date", how='outer')
    else:
        return df


def get_gender(subject_id):
    gender_map = {
        '540': 'Male',
        '544': 'Male',
        '552': 'Male',
        '567': 'Female',
        '584': 'Male',
        '596': 'Male',
        '559': 'Female',
        '563': 'Male',
        '570': 'Male',
        '575': 'Female',
        '588': 'Female',
        '591': 'Female',
    }
    return gender_map[subject_id]


def get_insulin_delivery_device(subject_id):
    map = {
        '540': 'MiniMed 630G',
        '544': 'MiniMed 530G',
        '552': 'MiniMed 630G',
        '567': 'MiniMed 630G',
        '584': 'MiniMed 530G',
        '596': 'MiniMed 530G',
        '559': 'MiniMed 530G',
        '563': 'MiniMed 530G',
        '570': 'MiniMed 530G',
        '575': 'MiniMed 530G',
        '588': 'MiniMed 530G',
        '591': 'MiniMed 530G',
    }
    return map[subject_id]


"""
The T1DEXI parser is processing the .xpt data from the Ohio T1DM datasets and returning the data merged into
the same time grid in a dataframe.

To do:
- correct the dates that are now in the future
- workouts and similar should have a separate column for duration instead
-
"""
from .base_parser import BaseParser
import pandas as pd
import isodate
import xport
import numpy as np
import datetime
import zipfile_deflate64


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.subject_ids = []

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the T1DEXI dataset root folder.
        """
        print(f"Processing data from {file_path}")
        self.subject_ids = get_valid_subject_ids(file_path)
        df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, steps_or_cal_burn_dict = self.get_dataframes(file_path)
        df_resampled = self.resample_data(df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, steps_or_cal_burn_dict)
        return df_resampled

    def resample_data(self, df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, steps_or_cal_burn_dict):
        # There are 88 subjects on MDI in the dataset --> 502 - 88 = 414. In the youth version: 261 - 37 = 224.
        # Total: 763 with, 638 without MDI
        # We use only the subjects not on MDI in this parser
        processed_dfs = []
        count = 1
        for subject_id in self.subject_ids:
            df_subject = df_glucose[df_glucose['id'] == subject_id].copy()
            df_subject['id'] = df_subject['id'].astype('int')
            df_subject = df_subject.resample('5min', label='right').mean()
            df_subject['id'] = df_subject['id'].astype('object')
            df_subject.sort_index(inplace=True)

            df_subject_meals = df_meals[df_meals['id'] == subject_id].copy()
            if not df_subject_meals.empty:
                df_subject_meal_name = df_subject_meals[df_subject_meals['meal_label'].notna()][['meal_label']].resample('5min', label='right').agg(
                    lambda x: ', '.join(x))
                df_subject_carbs = df_subject_meals[df_subject_meals['carbs'].notna()][['carbs']].resample('5min', label='right').sum()

                df_subject = pd.merge(df_subject, df_subject_meal_name, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_carbs, on="date", how='outer')

                # Fill NaN where numbers were turned to 0 or strings were turned to ''
                df_subject['meal_label'] = df_subject['meal_label'].replace('', np.nan)
                df_subject['carbs'] = df_subject['carbs'].infer_objects(copy=False).replace(0, np.nan)
            else:
                df_subject['meal_label'] = np.nan
                df_subject['carbs'] = np.nan

            df_subject_bolus = df_bolus[df_bolus['id'] == subject_id].copy()
            if not df_subject_bolus.empty:
                df_subject_bolus = df_subject_bolus[['bolus']].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_bolus, on="date", how='outer')
                df_subject['bolus'] = df_subject['bolus'].replace(0, np.nan)
            else:
                df_subject['bolus'] = np.nan

            df_subject_basal = df_basal[df_basal['id'] == subject_id].copy()
            if not df_subject_basal.empty:
                df_subject_basal = df_subject_basal[['basal']].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_basal, on="date", how='outer')
            else:
                df_subject['basal'] = np.nan

            df_subject_exercise = df_exercise[df_exercise['id'] == subject_id].copy()
            if not df_subject_exercise.empty:
                df_subject_exercise_workout = df_subject_exercise[['workout_label']].resample('5min', label='right').agg(
                    lambda x: ', '.join(sorted(set(x))))
                df_subject_exercise_workout_intensity = df_subject_exercise[['workout_intensity']].resample('5min',
                                                                                                            label='right').mean()
                df_subject_exercise_workout_duration = df_subject_exercise[['workout_duration']].resample('5min',
                                                                                                          label='right').sum()

                df_subject = pd.merge(df_subject, df_subject_exercise_workout, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_exercise_workout_intensity, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_exercise_workout_duration, on="date", how='outer')

                # Fill NaN for empty strings
                df_subject['workout_label'] = df_subject['workout_label'].replace('', np.nan)
            else:
                df_subject['workout_label'] = np.nan
                df_subject['workout_intensity'] = np.nan
                df_subject['workout_duration'] = np.nan

            if subject_id in heartrate_dict and not heartrate_dict[subject_id].empty:
                df_subject_heartrate = heartrate_dict[subject_id]
                df_subject_heartrate = df_subject_heartrate[['heartrate']].resample('5min', label='right').mean()
                df_subject = pd.merge(df_subject, df_subject_heartrate, on="date", how='outer')
            else:
                print(f"No heartrate data for subject {subject_id}")
                df_subject['heartrate'] = np.nan

            if subject_id in steps_or_cal_burn_dict and not steps_or_cal_burn_dict[subject_id].empty:
                df_subject_steps_or_cal_burn = steps_or_cal_burn_dict[subject_id]
                if 'steps' in df_subject_steps_or_cal_burn.columns:
                    col_name = 'steps'
                    df_subject['calories_burned'] = np.nan
                else:
                    col_name = 'calories_burned'
                    df_subject['steps'] = np.nan
                df_subject_steps_or_cal_burn = df_subject_steps_or_cal_burn[[col_name]].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_steps_or_cal_burn, on="date", how='outer')
            else:
                print(f"No steps or calories burned data for subject {subject_id}")
                df_subject['steps'] = np.nan
                df_subject['calories_burned'] = np.nan

            # Some rows might have gotten added nan values for the subject id after resampling
            df_subject['id'] = subject_id
            df_subject.sort_index(inplace=True)

            # Ensuring homogenous time intervals
            df_subject = df_subject.resample('5min').asfreq()

            processed_dfs.append(df_subject)
            print(f"{count}/{len(self.subject_ids)} are prepared")
            count += 1

        df_final = pd.concat(processed_dfs)
        return df_final

    def get_dataframes(self, file_path):
        """
        Extract the data that we are interested in from the dataset, and process them into our format and into separate
        dataframes.
        """
        df_glucose = get_df_glucose(file_path, self.subject_ids)
        print("Glucose data processed", df_glucose)
        df_meals = get_df_meals(file_path, self.subject_ids)
        print("Meal data processed", df_meals)
        df_insulin = get_df_insulin(file_path, self.subject_ids)
        print("Insulin data processed", df_insulin)
        df_bolus = get_df_bolus(df_insulin)
        print("Bolus data processed", df_bolus)
        df_basal = get_df_basal(df_insulin)
        print("Basal data processed", df_basal)
        df_exercise = get_df_exercise(file_path, self.subject_ids)
        print("Exercise data processed", df_exercise)
        heartrate_dict = get_heartrate_dict(file_path, self.subject_ids)
        print("Heartrate dict processed", heartrate_dict[self.subject_ids[0]])
        steps_or_cal_burn_dict = get_steps_or_cal_burn_dict(file_path, self.subject_ids)
        print("Steps or calories burned dict processed", steps_or_cal_burn_dict[self.subject_ids[0]])

        return df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, steps_or_cal_burn_dict


def get_valid_subject_ids(file_path):
    df_device = get_df_from_zip_deflate_64(file_path, 'DX.xpt')
    subject_ids_not_on_mdi = df_device[df_device['DXTRT'] != 'MULTIPLE DAILY INJECTIONS']['USUBJID'].unique()
    return subject_ids_not_on_mdi


def get_heartrate_dict(file_path, subject_ids):
    # Heartrate data is processed by chunks because the heartrate file is so big and creates memory problems
    row_count = 0
    chunksize = 1000000
    file_name = 'VS.xpt'
    heartrate_dict = {}

    with zipfile_deflate64.ZipFile(file_path, 'r') as zip_file:
        matched_files = [f for f in zip_file.namelist() if f.endswith(file_name)]
        matched_file = matched_files[0]
        with zip_file.open(matched_file) as xpt_file:
            for chunk in pd.read_sas(xpt_file, format='xport', chunksize=chunksize):
                row_count += chunksize
                df_heartrate = chunk.map(lambda x: x.decode() if isinstance(x, bytes) else x)

                # For some reason, the heart rate mean gives more data in T1DEXI version, while not in T1DEXIP
                # The heart rate from both devices in the study
                df_heartrate = df_heartrate[(df_heartrate['VSTEST'] == 'Heart Rate') | (df_heartrate['VSTEST'] == 'Heart Rate, Mean')]

                # Filter the DataFrame for subject_ids before looping over unique values
                df_heartrate = df_heartrate[df_heartrate['USUBJID'].isin(subject_ids)]

                for subject_id in df_heartrate['USUBJID'].unique():
                    filtered_rows = df_heartrate[df_heartrate['USUBJID'] == subject_id].copy()
                    filtered_rows['date'] = create_sas_date_for_column(filtered_rows['VSDTC'])
                    filtered_rows.loc[:, 'VSSTRESN'] = pd.to_numeric(filtered_rows['VSSTRESN'])
                    filtered_rows.rename(columns={'VSSTRESN': 'heartrate'}, inplace=True)
                    filtered_rows = filtered_rows[['heartrate', 'date']]
                    filtered_rows.set_index('date', inplace=True)
                    filtered_rows = filtered_rows.resample('5min', label='right').mean()

                    # If the USUBJID is already in the dictionary, append the new data
                    if subject_id in heartrate_dict:
                        heartrate_dict[subject_id] = pd.concat([heartrate_dict[subject_id], filtered_rows])
                    else:
                        # Otherwise, create a new DataFrame for this USUBJID
                        heartrate_dict[subject_id] = filtered_rows
            return heartrate_dict


def get_steps_or_cal_burn_dict(file_path, subject_ids):
    # Steps data is processed by chunks because the steps file is so big and creates memory problems
    row_count = 0
    chunksize = 1000000
    file_name = '/FA.xpt'
    results_dict = {}

    with zipfile_deflate64.ZipFile(file_path, 'r') as zip_file:
        matched_files = [f for f in zip_file.namelist() if f.endswith(file_name)]
        matched_file = matched_files[0]
        with zip_file.open(matched_file) as xpt_file:
            for chunk in pd.read_sas(xpt_file, format='xport', chunksize=chunksize):
                row_count += chunksize
                df_fa = chunk.map(lambda x: x.decode() if isinstance(x, bytes) else x)
                # Filter the DataFrame for subject_ids before looping over unique values
                df_fa = df_fa[df_fa['USUBJID'].isin(subject_ids)]
                df_fa['date'] = create_sas_date_for_column(df_fa['FADTC'])

                for subject_id in df_fa['USUBJID'].unique():
                    filtered_rows = df_fa[df_fa['USUBJID'] == subject_id].copy()

                    if 'T1DEXIP' in file_path:
                        filtered_rows = filtered_rows[filtered_rows['FATESTCD'] == 'CALBURN']
                        filtered_rows.loc[:, 'FASTRESN'] = pd.to_numeric(filtered_rows['FASTRESN'])
                        filtered_rows.rename(columns={'FASTRESN': 'calories_burned'}, inplace=True)
                        filtered_rows = filtered_rows[['calories_burned', 'date']]
                    else:
                        filtered_rows = filtered_rows[filtered_rows['FAOBJ'] == '10-SECOND INTERVAL STEP COUNT']
                        filtered_rows.loc[:, 'FASTRESN'] = pd.to_numeric(filtered_rows['FASTRESN'])
                        filtered_rows.rename(columns={'FASTRESN': 'steps'}, inplace=True)
                        filtered_rows = filtered_rows[['steps', 'date']]

                    filtered_rows.set_index('date', inplace=True)
                    filtered_rows = filtered_rows.resample('5min', label='right').sum()

                    # If the USUBJID is already in the dictionary, append the new data
                    if subject_id in results_dict:
                        results_dict[subject_id] = pd.concat([results_dict[subject_id], filtered_rows])
                    else:
                        # Otherwise, create a new DataFrame for this USUBJID
                        results_dict[subject_id] = filtered_rows
            return results_dict

def get_df_glucose(file_path, subject_ids):
    df_glucose = get_df_from_zip_deflate_64(file_path, 'LB.xpt', subject_ids=subject_ids)
    df_glucose['date'] = create_sas_date_for_column(df_glucose['LBDTC'])
    df_glucose.loc[:, 'LBSTRESN'] = pd.to_numeric(df_glucose['LBSTRESN'], errors='coerce')
    df_glucose.rename(columns={'LBSTRESN': 'CGM', 'USUBJID': 'id'}, inplace=True)
    df_glucose = df_glucose[df_glucose['LBTESTCD'] == 'GLUC']  # Filtering out glucose (from hba1c)
    df_glucose = df_glucose[['id', 'CGM', 'date']]
    df_glucose.set_index('date', inplace=True)
    return df_glucose


def get_df_meals(file_path, subject_ids):
    df_meals = get_df_from_zip_deflate_64(file_path, 'FAMLPM.xpt', subject_ids=subject_ids)
    df_meals = df_meals[df_meals['FACAT'] == 'CONSUMED']  # Consumed is taken - returned
    df_meals = df_meals[df_meals['FATESTCD'] == 'DCARBT']  # Extracting dietary carbohydrates
    df_meals['date'] = create_sas_date_for_column(df_meals['FADTC'])
    df_meals.loc[:, 'FAORRES'] = pd.to_numeric(df_meals['FAORRES'], errors='coerce')
    df_meals.rename(columns={'FAORRES': 'carbs', 'PTMLDESC': 'meal_label', 'USUBJID': 'id'}, inplace=True)
    df_meals = df_meals[['carbs', 'meal_label', 'id', 'date']]
    df_meals.set_index('date', inplace=True)
    return df_meals


def get_df_insulin(file_path, subject_ids):
    df_insulin = get_df_from_zip_deflate_64(file_path, 'FACM.xpt', subject_ids=subject_ids)
    df_insulin['date'] = create_sas_date_for_column(df_insulin['FADTC'])
    df_insulin.loc[:, 'FASTRESN'] = pd.to_numeric(df_insulin['FASTRESN'], errors='coerce')
    df_insulin.rename(columns={'FASTRESN': 'dose', 'FAORRESU': 'unit', 'USUBJID': 'id', 'INSNMBOL':
        'delivered bolus', 'INSEXBOL': 'extended bolus', 'FADUR': 'duration', 'FATEST': 'type'}, inplace=True)
    df_insulin['duration'] = df_insulin['duration'].apply(lambda x: parse_duration(x))
    df_insulin = df_insulin.drop_duplicates()
    return df_insulin[['id', 'unit', 'dose', 'date', 'duration', 'delivered bolus', 'extended bolus', 'INSSTYPE',
                       'type']]


def get_df_bolus(df_insulin):
    df_bolus = df_insulin[df_insulin['type'] == 'BOLUS INSULIN'].copy()
    # Let values in delivered bolus override dose for extended boluses
    df_bolus.loc[df_bolus['delivered bolus'].notna(), 'dose'] = df_bolus[df_bolus['delivered bolus'].notna()]['delivered bolus']
    df_bolus.drop(columns=['delivered bolus'], inplace=True)
    # For the square boluses, "delivered bolus" is nan, but dose should be 0.0 because all of the bolus dose is extended
    df_bolus.loc[df_bolus['INSSTYPE'] == 'square', 'dose'] = 0.0

    # Get extended boluses as a dataframe with doses spread across 5-minute intervals
    df_bolus['extended bolus'] = df_bolus['extended bolus'].replace(0.0, np.nan)
    extended_boluses = df_bolus[df_bolus['extended bolus'].notna()].copy()
    new_rows = []
    for _, row in extended_boluses.iterrows():
        new_rows.extend(split_duration(row, 'extended bolus'))
    extended_boluses = pd.DataFrame(new_rows)
    extended_boluses.drop(columns=['duration'], inplace=True)

    columns = ['date', 'dose', 'id']
    extended_boluses.rename(columns={'extended bolus': 'dose'}, inplace=True)
    merged_bolus_df = pd.concat([df_bolus[columns], extended_boluses[columns]], ignore_index=True)
    merged_bolus_df.rename(columns={'dose': 'bolus'}, inplace=True)
    merged_bolus_df.set_index('date', inplace=True)
    return merged_bolus_df


def get_df_basal(df_insulin):
    df_basal = df_insulin[df_insulin['type'] != 'BOLUS INSULIN'][['INSSTYPE', 'dose', 'unit', 'date', 'id', 'duration']]
    df_basal = df_basal[df_basal['unit'] == 'U']
    df_basal['duration'] = df_basal['duration'].fillna(pd.Timedelta(minutes=0))
    df_basal = df_basal.drop_duplicates(subset=['id', 'date'])  # Drop sample when start dates are the same

    # manipulate duration to match with the next sample if the id is the same
    df_basal['next_date'] = df_basal['date'].shift(-1)  # Get next row's date
    df_basal['next_id'] = df_basal['id'].shift(-1)  # Get next row's id
    df_basal['duration'] = df_basal.apply(
        lambda row: (row['next_date'] - row['date']) if row['id'] == row['next_id'] else row['duration'],
        axis=1
    )
    # Split basal insulin into time interval bins so that it is distributed in the resampling
    new_rows = []
    for _, row in df_basal.iterrows():
        new_rows.extend(split_duration(row, 'dose'))
    df_basal = pd.DataFrame(new_rows)
    df_basal.drop(columns=['duration'], inplace=True)
    df_basal['dose'] = df_basal['dose'] * 12  # Convert to U/hr
    df_basal.set_index('date', inplace=True)
    df_basal.rename(columns={'dose': 'basal'}, inplace=True)
    return df_basal


def get_df_exercise(file_path, subject_ids):
    df_exercise = get_df_from_zip_deflate_64(file_path, '/PR.xpt', subject_ids=subject_ids)
    df_exercise['date'] = pd.to_datetime(create_sas_date_for_column(df_exercise['PRSTDTC']))

    if 'T1DEXIP' in file_path:
        exercise_map = {
            '': 0,
            'Low: Pretty easy': 1.7,
            'Mild: Working a bit': 3.3,
            'Feeling the burn - Mild': 5.0,
            'Moderate: Working to keep up': 6.7,
            'Heavy: Hard to keep going but did it': 8.3,
            'Exhaustive: Too tough/Had to stop': 10
        }
        df_exercise.loc[:, 'EXCINTSY'] = df_exercise['EXCINTSY'].map(exercise_map)
    else:
        # Original values are 0, 1 and 2, but we add 1 and multiply with 3.3 so the scale is from 0-10
        df_exercise.loc[:, 'EXCINTSY'] = pd.to_numeric(df_exercise['EXCINTSY'])
        df_exercise.loc[:, 'EXCINTSY'] = df_exercise['EXCINTSY'] + 1
        df_exercise['EXCINTSY'] *= 3.3
    df_exercise.rename(
        columns={'USUBJID': 'id', 'PRCAT': 'workout_label', 'PLNEXDUR': 'workout_duration',
                 'EXCINTSY': 'workout_intensity'}, inplace=True)
    df_exercise = df_exercise[
        ['workout_label', 'workout_duration', 'workout_intensity', 'id', 'date']]
    df_exercise.set_index('date', inplace=True)
    return df_exercise


def split_duration(row, value_column):
    """
    For bolus doses or basal rates with a duration, we split the dose injection across 5-minute intervals by adding
    new rows for every 5-minute window in duration, and equally split the original dose across those rows.
    """
    rounded_duration = round(row['duration'] / pd.Timedelta(minutes=5)) * pd.Timedelta(minutes=5)
    num_intervals = rounded_duration // pd.Timedelta(minutes=5)
    if num_intervals < 1:
        num_intervals = 1
    value_per_interval = row[value_column] / num_intervals
    new_rows = []
    for i in range(int(num_intervals)):
        new_row = {
            'date': row['date'] + pd.Timedelta(minutes=5 * i),
            value_column: value_per_interval,
            'duration': pd.Timedelta(minutes=5),
            'id': row['id'],
        }
        new_rows.append(new_row)
    return new_rows


def parse_duration(duration_str):
    try:
        return pd.to_timedelta(isodate.parse_duration(duration_str)) if duration_str else np.nan
    except isodate.ISO8601Error:
        return np.nan


def create_sas_date_for_column(series):
    sas_epoch = datetime.datetime(1960, 1, 1)
    series = pd.to_datetime(series, unit='s', origin=sas_epoch)
    return series


def get_df_from_zip_deflate_64(zip_path, file_name, subject_ids=None):
    with zipfile_deflate64.ZipFile(zip_path, 'r') as zip_file:
        # Find the file in the archive that ends with the specified file_name
        matched_files = [f for f in zip_file.namelist() if f.endswith(file_name)]

        if not matched_files:
            raise FileNotFoundError(f"No file ending with '{file_name}' found in the zip archive.")

        # Use the first match (if multiple matches, refine criteria as needed)
        matched_file = matched_files[0]

        # Read the .xpt file directly from the zip
        with zip_file.open(matched_file) as xpt_file:
            df = xport.to_dataframe(xpt_file)  # Load the .xpt file into a DataFrame

        if subject_ids is not None:
            df = df[df['USUBJID'].isin(subject_ids)]
        return df



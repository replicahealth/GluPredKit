import pandas as pd
import os
import zipfile
import re
import numpy as np
import psutil
import json
from datetime import datetime
from .base_parser import BaseParser


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the folder that contains all the .zip files of the OpenAPS data.
        """
        merged_df = pd.DataFrame()

        # List all files in the folder
        files = [el for el in os.listdir(file_path) if el.endswith('.zip')]  # 142 subjects

        # Process the files one by one
        for file in files:
            if file == 'AndroidAPS Uploader.zip':
                with zipfile.ZipFile(file_path + 'AndroidAPS Uploader.zip', 'r') as zip_ref:
                    # Find unique ids
                    all_ids = np.unique([file.split('/')[0] for file in zip_ref.namelist() if not file.split('/')[0] == ''])

                    for subject_id in all_ids:
                        print(f'Processing {subject_id}...')

                        def get_relevant_files(name):
                            relevant_files = []

                            for file_name in zip_ref.namelist():
                                if file_name.startswith(subject_id) and file_name.endswith(f'{name}.json'):
                                    relevant_files.append(file_name)
                                    # print("FILE NAME", file_name)

                            # Check whether the file is found
                            if not relevant_files:
                                print(f"No files found  containing '{name}' in the name!")

                            return relevant_files

                        # Blood glucose
                        entries_files = get_relevant_files('BgReadings')

                        # Carbohydrates and insulin
                        treatments_files = get_relevant_files('Treatments')

                        # Basal rates
                        basal_files = get_relevant_files('APSData')

                        # Temporary basal rates
                        temp_basal_files = get_relevant_files('TemporaryBasals')

                        # Skip to next iteration if entries_files is empty
                        if not entries_files or not treatments_files or not basal_files:
                            print("Skipping to next subject...")
                            continue

                        all_entries_dfs = []
                        for entries_file in entries_files:
                            with zip_ref.open(entries_file) as f:
                                entries_df = pd.read_json(f, convert_dates=False)

                            entries_df['date'] = pd.to_datetime(entries_df['date'], unit='ms')
                            entries_df['value'] = pd.to_numeric(entries_df['value'])
                            entries_df = entries_df[['date', 'value']]
                            entries_df.rename(columns={'value': 'CGM'}, inplace=True)
                            entries_df.set_index('date', inplace=True)
                            entries_df.sort_index(inplace=True)
                            all_entries_dfs.append(entries_df)

                        df = pd.concat(all_entries_dfs)
                        df = df.resample('5min').mean()

                        carbs_dfs = []
                        bolus_dfs = []
                        for treatments_file in treatments_files:
                            with zip_ref.open(treatments_file) as f:
                                treatments_df = pd.read_json(f, convert_dates=False)

                            carbs_df = treatments_df.copy()[['date', 'carbs']]
                            carbs_df['date'] = pd.to_datetime(carbs_df['date'], unit='ms')
                            carbs_df['carbs'] = pd.to_numeric(carbs_df['carbs'])
                            carbs_df.set_index('date', inplace=True)
                            carbs_df.sort_index(inplace=True)
                            carbs_df = carbs_df[carbs_df['carbs'].notna() & (carbs_df['carbs'] != 0)]
                            carbs_dfs.append(carbs_df)

                            bolus_df = treatments_df.copy()[['date', 'insulin']]
                            bolus_df['date'] = pd.to_datetime(bolus_df['date'], unit='ms')
                            bolus_df['insulin'] = pd.to_numeric(bolus_df['insulin'])
                            bolus_df.rename(columns={'insulin': 'bolus'}, inplace=True)
                            bolus_df.set_index('date', inplace=True)
                            bolus_df.sort_index(inplace=True)
                            bolus_df = bolus_df[bolus_df['bolus'].notna() & (bolus_df['bolus'] != 0)]
                            bolus_dfs.append(bolus_df)

                        df_carbs = pd.concat(carbs_dfs)
                        df_carbs = drop_duplicates(df_carbs, 'carbs')
                        df_carbs = df_carbs.resample('5min').sum().fillna(value=0)
                        df = pd.merge(df, df_carbs, on="date", how='outer')

                        df_bolus = pd.concat(bolus_dfs)
                        df_bolus = drop_duplicates(df_bolus, 'bolus')
                        df_bolus = df_bolus.resample('5min').sum().fillna(value=0)
                        df = pd.merge(df, df_bolus, on="date", how='outer')

                        all_basal_dfs = []
                        for basal_file in basal_files:
                            with zip_ref.open(basal_file) as f:
                                basal_df = pd.read_json(f, convert_dates=False)

                            basal_df = basal_df.copy()[['queuedOn', 'profile']]
                            basal_df['queuedOn'] = pd.to_datetime(basal_df['queuedOn'], unit='ms')
                            basal_df['profile'] = pd.to_numeric(basal_df['profile'].apply(lambda x: x['current_basal']))
                            basal_df.rename(columns={'queuedOn': 'date', 'profile': 'basal'}, inplace=True)
                            basal_df.set_index('date', inplace=True)
                            basal_df.sort_index(inplace=True)
                            all_basal_dfs.append(basal_df)

                        df_basal = pd.concat(all_basal_dfs)
                        df_basal = df_basal.resample('5min').last()
                        df = pd.merge(df, df_basal, on="date", how='outer')

                        # Override basal rates with temporary basal rates
                        if len(temp_basal_files) > 0:
                            all_temp_basal_dfs = []
                            for temp_basal_file in temp_basal_files:
                                with zip_ref.open(temp_basal_file) as f:
                                    temp_basal_df = pd.read_json(f, convert_dates=False)
                                temp_basal_df = temp_basal_df.copy()[
                                    ['date', 'durationInMinutes', 'isAbsolute', 'percentRate', 'absoluteRate']]
                                temp_basal_df['date'] = pd.to_datetime(temp_basal_df['date'], unit='ms')
                                temp_basal_df.set_index('date', inplace=True)
                                temp_basal_df.sort_index(inplace=True)
                                temp_basal_df['durationInMinutes'] = pd.to_numeric(temp_basal_df['durationInMinutes'])
                                temp_basal_df = temp_basal_df[temp_basal_df['durationInMinutes'] > 0]
                                all_temp_basal_dfs.append(temp_basal_df)

                            df_temp_basal = pd.concat(all_temp_basal_dfs)
                            df_temp_basal = df_temp_basal.resample('5min').last()
                            df_temp_basal['isAbsolute'] = df_temp_basal['isAbsolute'].astype('boolean')
                            df = pd.merge(df, df_temp_basal, on="date", how='outer')

                            # Forward fill temp_basal up to the number in the duration column
                            for idx, row in df.iterrows():
                                if not pd.isna(row['percentRate']) and not pd.isna(
                                        row['durationInMinutes']) and not pd.isna(row['isAbsolute']):
                                    fill_limit = int(
                                        row['durationInMinutes'])  # duration in minutes, index freq is 5 minutes
                                    timedeltas = range(5, int(row['durationInMinutes']), 5)

                                    for timedelta in timedeltas:
                                        fill_index = idx + pd.Timedelta(minutes=timedelta)
                                        if fill_index in df.index:
                                            df.loc[fill_index, 'percentRate'] = row['percentRate']
                                            df.loc[fill_index, 'absoluteRate'] = row['absoluteRate']
                                            df.loc[fill_index, 'isAbsolute'] = row['isAbsolute']

                            df.loc[df['isAbsolute'] == False, 'absoluteRate'] = np.nan
                            df['merged_basal'] = df['absoluteRate'].combine_first(df['basal'])

                            # Check if temp basal is "Percentage". If yes, calculate from basal rate. Print those columns
                            df.loc[df['isAbsolute'] == False, 'merged_basal'] = df['percentRate'] * df['basal'] / 100
                            df.drop(columns=['durationInMinutes', 'isAbsolute', 'percentRate', 'absoluteRate', 'basal'],
                                    inplace=True)
                            df.rename(columns={'merged_basal': 'basal'}, inplace=True)

                        # Try to extract reservoir data from APSData if available
                        df['reservoir'] = np.nan
                        for basal_file in basal_files:
                            with zip_ref.open(basal_file) as f:
                                aps_df = pd.read_json(f, convert_dates=False)
                                if 'pump' in aps_df.columns:
                                    reservoir_df = pd.DataFrame()
                                    reservoir_df['queuedOn'] = pd.to_datetime(aps_df['queuedOn'], unit='ms')
                                    reservoir_df['reservoir'] = aps_df['pump'].apply(
                                        lambda x: x.get('reservoir') if isinstance(x, dict) else np.nan
                                    )
                                    reservoir_df.rename(columns={'queuedOn': 'date'}, inplace=True)
                                    reservoir_df.set_index('date', inplace=True)
                                    reservoir_df.sort_index(inplace=True)
                                    reservoir_df = reservoir_df[reservoir_df['reservoir'].notna()]
                                    if not reservoir_df.empty:
                                        reservoir_df = reservoir_df.resample('5min').last()
                                        df = pd.merge(df, reservoir_df, on="date", how='outer')
                                        df['reservoir'] = df['reservoir'].ffill()
                                        break
                        merged_df = finalize_and_merge_subject_into_df(merged_df, df, subject_id)
            else:
                subject_id = file.split('.')[0]

                print(f'Processing {file}...')
                zip_file_path = file_path + file
                sub_folder_path = 'direct-sharing-31/'

                # Open the zip file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    def get_relevant_files(name):
                        relevant_files = []

                        for file_name in zip_ref.namelist():
                            if len(file_name.split('/')) == 2:
                                if name in file_name.lower():
                                    if file_name.endswith('.json'):
                                        relevant_files.append(file_name)

                        # Check whether the file is found
                        if not relevant_files:
                            print(f"No files found  containing '{name}' in the name!")

                        return relevant_files

                    # Blood glucose
                    entries_files = get_relevant_files('entries')
                    all_entries_dfs = []
                    for entries_file in entries_files:
                        with zip_ref.open(entries_file) as f:
                            if entries_file.startswith(sub_folder_path):
                                try:
                                    entries_df = pd.read_json(f, convert_dates=False)
                                except ValueError as e:
                                    # Reset the file handle to the start
                                    f.seek(0)
                                    entries_df = pd.DataFrame()
                                    print(f"Error parsing JSON directly: {e}")
                                    for line in f:
                                        try:
                                            data = json.loads(line)
                                            entries_df = entries_df._append(data, ignore_index=True)
                                        except json.JSONDecodeError as json_err:
                                            print(f"Skipping line due to error: {json_err}")
                            else:
                                entries_df = pd.read_json(f, convert_dates=False, lines=True)
                            if entries_df.empty:
                                continue
                            entries_df['dateString'] = entries_df['dateString'].apply(parse_datetime_without_timezone)
                            entries_df['sgv'] = pd.to_numeric(entries_df['sgv'])
                            entries_df = entries_df[['dateString', 'sgv']]
                            entries_df.rename(columns={'sgv': 'CGM', 'dateString': 'date'}, inplace=True)
                            entries_df.set_index('date', inplace=True)
                            entries_df.sort_index(inplace=True)
                            all_entries_dfs.append(entries_df)

                    if len(all_entries_dfs) == 0:
                        print(f'No glucose entries for {file}. Skipping to next subject.')
                        continue
                    df = pd.concat(all_entries_dfs)
                    df = df.resample('5min').mean()

                    # Carbohydrates
                    treatments_files = get_relevant_files('treatments')
                    carbs_dfs = []
                    bolus_dfs = []
                    temp_basal_dfs = []
                    for treatments_file in treatments_files:
                        with zip_ref.open(treatments_file) as f:
                            if treatments_file.startswith(sub_folder_path):
                                treatments_df = pd.read_json(f, convert_dates=False)
                            else:
                                treatments_df = pd.read_json(f, convert_dates=False, lines=True)

                            if treatments_df.empty:
                                continue

                            carbs_df = treatments_df.copy()[['created_at', 'carbs']]
                            carbs_df['created_at'] = carbs_df['created_at'].apply(parse_datetime_without_timezone)
                            carbs_df['carbs'] = pd.to_numeric(carbs_df['carbs'])
                            carbs_df.rename(columns={'created_at': 'date'}, inplace=True)
                            carbs_df.set_index('date', inplace=True)
                            carbs_df.sort_index(inplace=True)
                            carbs_df = carbs_df[carbs_df['carbs'].notna() & (carbs_df['carbs'] != 0)]
                            carbs_dfs.append(carbs_df)

                            bolus_df = treatments_df.copy()[['created_at', 'insulin']]
                            bolus_df['created_at'] = bolus_df['created_at'].apply(parse_datetime_without_timezone)
                            bolus_df['insulin'] = pd.to_numeric(bolus_df['insulin'])
                            bolus_df.rename(columns={'created_at': 'date', 'insulin': 'bolus'}, inplace=True)
                            bolus_df.set_index('date', inplace=True)
                            bolus_df.sort_index(inplace=True)
                            bolus_df = bolus_df[bolus_df['bolus'].notna() & (bolus_df['bolus'] != 0)]
                            bolus_dfs.append(bolus_df)

                            if not 'rate' in treatments_df.columns:
                                if ('percent' in treatments_df.columns) and ('duration' in treatments_df.columns):
                                    temp_basal_df = treatments_df.copy()[['created_at', 'percent', 'duration']]
                                    temp_basal_df['temp'] = 'percentage'
                                    temp_basal_df['created_at'] = temp_basal_df['created_at'].apply(
                                        parse_datetime_without_timezone)
                                    temp_basal_df['percent'] = pd.to_numeric(temp_basal_df['percent'],
                                                                             errors='coerce') + 100
                                    temp_basal_df.rename(columns={'created_at': 'date', 'percent': 'temp_basal'},
                                                         inplace=True)
                                elif ('absolute' in treatments_df.columns) and ('duration' in treatments_df.columns):
                                    temp_basal_df = treatments_df.copy()[['created_at', 'absolute', 'duration']]
                                    temp_basal_df['temp'] = np.nan
                                    temp_basal_df['created_at'] = temp_basal_df['created_at'].apply(
                                        parse_datetime_without_timezone)
                                    temp_basal_df['absolute'] = pd.to_numeric(temp_basal_df['absolute'], errors='coerce')
                                    temp_basal_df.rename(columns={'created_at': 'date', 'absolute': 'temp_basal'},
                                                         inplace=True)
                                else:
                                    print("No columns for temporary basal is found! ")
                                    print(f'Data columns are: {treatments_df.columns}')
                                    print(f'EventTypes are: {treatments_df.eventType.unique()}')
                                    temp_basal_df = pd.DataFrame()
                            else:
                                if 'temp' in treatments_df.columns:
                                    temp_basal_df = treatments_df.copy()[['created_at', 'rate', 'duration', 'temp']]
                                else:
                                    temp_basal_df = treatments_df.copy()[['created_at', 'rate', 'duration']]
                                    temp_basal_df['temp'] = None
                                temp_basal_df['created_at'] = temp_basal_df['created_at'].apply(
                                    parse_datetime_without_timezone)
                                temp_basal_df['rate'] = pd.to_numeric(temp_basal_df['rate'], errors='coerce')
                                temp_basal_df.rename(columns={'created_at': 'date', 'rate': 'temp_basal'}, inplace=True)
                            if not temp_basal_df.empty:
                                temp_basal_df.set_index('date', inplace=True)
                                temp_basal_df.sort_index(inplace=True)
                                temp_basal_df = temp_basal_df[temp_basal_df['temp_basal'].notna()]
                                temp_basal_dfs.append(temp_basal_df)

                    df_carbs = pd.concat(carbs_dfs)
                    df_carbs = drop_duplicates(df_carbs, 'carbs')
                    df_carbs = df_carbs.resample('5min').sum().fillna(value=0)
                    df = pd.merge(df, df_carbs, on="date", how='outer')
                    df['carbs'] = df['carbs'].fillna(value=0.0)

                    df_bolus = pd.concat(bolus_dfs)
                    df_bolus = drop_duplicates(df_bolus, 'bolus')
                    df_bolus = df_bolus.resample('5min').sum().fillna(value=0)
                    df = pd.merge(df, df_bolus, on="date", how='outer')
                    df['bolus'] = df['bolus'].fillna(value=0.0)

                    if temp_basal_df.empty:
                        df['temp_basal'] = np.nan
                        df['duration'] = np.nan
                        df['temp'] = np.nan
                    else:
                        df_temp_basal = pd.concat(temp_basal_dfs)
                        df_temp_basal = df_temp_basal.resample('5min').last()
                        df = pd.merge(df, df_temp_basal, on="date", how='outer')

                    # Forward fill temp_basal up to the number in the duration column
                    for idx, row in df.iterrows():
                        if not pd.isna(row['temp_basal']) and not pd.isna(row['duration']):
                            fill_limit = int(row['duration'])  # duration in minutes, index freq is 5 minutes
                            timedeltas = range(5, int(row['duration']), 5)

                            for timedelta in timedeltas:
                                fill_index = idx + pd.Timedelta(minutes=timedelta)
                                if fill_index in df.index:
                                    if pd.isna(df.loc[fill_index, 'temp_basal']):
                                        df.loc[fill_index, 'temp_basal'] = row['temp_basal']
                                        df.loc[fill_index, 'temp'] = row['temp']
                                    else:
                                        continue

                    # Drop the duration column
                    df.drop(columns='duration', inplace=True)

                    # Reservoir data from devicestatus
                    devicestatus_files = get_relevant_files('devicestatus')
                    reservoir_dfs = []
                    for devicestatus_file in devicestatus_files:
                        with zip_ref.open(devicestatus_file) as f:
                            if devicestatus_file.startswith(sub_folder_path):
                                try:
                                    devicestatus_df = pd.read_json(f, convert_dates=False)
                                except ValueError as e:
                                    # Reset the file handle to the start
                                    f.seek(0)
                                    devicestatus_df = pd.DataFrame()
                                    print(f"Error parsing devicestatus JSON directly: {e}")
                                    for line in f:
                                        try:
                                            data = json.loads(line)
                                            devicestatus_df = devicestatus_df._append(data, ignore_index=True)
                                        except json.JSONDecodeError as json_err:
                                            print(f"Skipping devicestatus line due to error: {json_err}")
                            else:
                                print(f"Skipping devicestatus line for subject {subject_id}")
                                continue

                            if devicestatus_df is None:
                                print(f"Devicestatus is None. Skipping devicestatus line for subject {subject_id}")
                                continue

                            if devicestatus_df.empty:
                                print(f"Devicestatus is empty. Skipping devicestatus line for subject {subject_id}")
                                continue
                            
                            # Extract reservoir data if available
                            if 'pump' in devicestatus_df.columns:
                                reservoir_df = pd.DataFrame()
                                reservoir_df['created_at'] = devicestatus_df['created_at'].apply(parse_datetime_without_timezone)
                                # Extract reservoir value from nested pump object
                                reservoir_df['reservoir'] = devicestatus_df['pump'].apply(
                                    lambda x: x.get('reservoir') if isinstance(x, dict) else np.nan
                                )
                                reservoir_df = reservoir_df[['created_at', 'reservoir']]
                                reservoir_df.rename(columns={'created_at': 'date'}, inplace=True)
                                reservoir_df.set_index('date', inplace=True)
                                reservoir_df.sort_index(inplace=True)
                                reservoir_df = reservoir_df[reservoir_df['reservoir'].notna()]
                                if not reservoir_df.empty:
                                    reservoir_dfs.append(reservoir_df)
                    
                    # Merge reservoir data if available
                    if reservoir_dfs:
                        df_reservoir = pd.concat(reservoir_dfs)
                        df_reservoir = df_reservoir.resample('5min').last()  # Use last value in each 5-min window
                        df = pd.merge(df, df_reservoir, on="date", how='outer')
                        # Forward fill reservoir values since they don't change frequently
                        df['reservoir'] = df['reservoir'].ffill()
                    else:
                        df['reservoir'] = np.nan

                    # Basal rates
                    profile_files = get_relevant_files('profile')
                    for profile_file in profile_files:
                        with zip_ref.open(profile_file) as f:
                            if profile_file.startswith(sub_folder_path):
                                basal_df = pd.read_json(f, convert_dates=False)
                            else:
                                basal_df = pd.read_json(f, convert_dates=False, lines=True)

                            if 'store' in basal_df.columns:
                                basal_df = basal_df[['store', 'startDate', 'defaultProfile']]
                                basal_df['startDate'] = basal_df['startDate'].apply(parse_datetime_without_timezone)
                                basal_df.set_index('startDate', inplace=True)

                                # Drop duplicates based on the date part of the DatetimeIndex
                                basal_df = basal_df[~basal_df.index.normalize().duplicated(keep='first')]
                                basal_df.sort_index(inplace=True)

                                df['basal'] = np.nan
                                for idx, row in basal_df.iterrows():
                                    if pd.isna(row['store']):
                                        continue
                                    basal_rates = row['store'][row['defaultProfile']]['basal']
                                    for basal in basal_rates:
                                        basal_time = datetime.strptime(basal['time'], "%H:%M").time()
                                        # Create filter mask for main_df based on time and date
                                        mask = (df.index >= idx) & (df.index.time >= basal_time)
                                        df.loc[mask, 'basal'] = float(basal['value'])
                            elif 'basal' in basal_df.columns:
                                basal_df = basal_df[['basal', 'startDate']]
                                basal_df['startDate'] = basal_df['startDate'].apply(parse_datetime_without_timezone)
                                basal_df.set_index('startDate', inplace=True)

                                # Drop duplicates based on the date part of the DatetimeIndex
                                basal_df = basal_df[~basal_df.index.normalize().duplicated(keep='first')]
                                basal_df.sort_index(inplace=True)

                                df['basal'] = np.nan
                                for idx, row in basal_df.iterrows():
                                    basal_rates = row['basal']
                                    for basal in basal_rates:
                                        basal_time = datetime.strptime(basal['time'], "%H:%M").time()
                                        # Create filter mask for main_df based on time and date
                                        mask = (df.index >= idx) & (df.index.time >= basal_time)
                                        df.loc[mask, 'basal'] = float(basal['value'])
                            else:
                                print(f"GET BASAL ERROR FOR {file}")

                    df['merged_basal'] = df['temp_basal'].combine_first(df['basal'])

                    # Check if temp basal is "Percentage". If yes, calculate from basal rate. Print those columns
                    df.loc[df['temp'] == 'percentage', 'merged_basal'] = df['temp_basal'] * df['basal'] / 100
                    df.drop(columns=['temp', 'temp_basal', 'basal'], inplace=True)
                    df.rename(columns={'merged_basal': 'basal'}, inplace=True)

                merged_df = finalize_and_merge_subject_into_df(merged_df, df, subject_id)

        """
        # TODO: This should be removed to the CLI and check for all of the datasets
        # Function to validate the time intervals
        def validate_intervals(group):
            # Calculate the time difference between consecutive dates
            time_diff = group.index.to_series().diff().dt.total_seconds().dropna()
            # Check if all time differences are exactly 300 seconds (5 minutes)
            valid = (time_diff == 300).all()
            if not valid:
                print(f"ID {group['id'].iloc[0]} has invalid intervals.")
            return valid

        # Group by 'id' and apply the validation function
        valid_intervals = merged_df.groupby('id').apply(validate_intervals)

        if valid_intervals.all():
            print("All IDs have valid 5-minute intervals with no bigger breaks than 5 minutes.")
        else:
            print("There are IDs with invalid intervals.")
        """
        merged_df['basal'] = merged_df['basal'] / 12  # From U/hr to U
        merged_df['insulin'] = merged_df['bolus'].fillna(0) + merged_df['basal']
        merged_df['source_file'] = 'OpenAPS'
        merged_df = add_demographics_to_df(file_path, merged_df)
        return merged_df

def finalize_and_merge_subject_into_df(merged_df, subject_df, subject_id):
    subject_df['id'] = subject_id

    # Remove current and following 8 hrs of insulin if outlier
    for dose_col in ['insulin', 'bolus']:
        if dose_col in subject_df.columns:
            bad_idx = subject_df.index[(subject_df[dose_col] < 0) | (subject_df[dose_col] > 50)]
            if len(bad_idx) > 0:
                print(f"Warning: Subject {subject_id} has {len(bad_idx)} outlier {dose_col} values. "
                      "We set the value and the following eight hours of data to nan.")
                rows_to_nan = []
                for idx in bad_idx:
                    loc = subject_df.index.get_loc(idx)  # safe unless duplicates exist
                    rows_to_nan.extend(range(loc, loc + 96))
                rows_to_nan = [i for i in rows_to_nan if i < len(subject_df)]
                insulin_col = subject_df.columns.get_loc(dose_col)
                subject_df.iloc[rows_to_nan, insulin_col] = np.nan

    merged_df = pd.concat([subject_df, merged_df], ignore_index=False)
    print(f"Current memory usage: {get_memory_usage()} MB")
    return merged_df


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

# TODO: We need to also look for possible duplicates across unique subject ids
def drop_duplicates(df_with_duplications, col):
    # TODO: We need to inspect further whether the duplicated can have larger differentials in the datetime (1s here)
    df_with_duplications['rounded_time'] = df_with_duplications.index.round('s')  # 's' for seconds

    # Identify duplicates while keeping the first occurrence
    duplicates_mask = df_with_duplications.duplicated(subset=['rounded_time', col], keep='first')

    # Invert the mask to keep only the first occurrences
    return df_with_duplications[~duplicates_mask][[col]]

# List of time zone abbreviations
tz_abbreviations = [
    'MDT', 'MST', 'GMT', 'CEST', 'CET', 'CDT', 'PST', 'PDT', 'EST', 'EDT', 'AST', 'vorm.', 'nachm.'
]
def parse_datetime_without_timezone(dt_str):
    dt_str = str(dt_str)

    # Remove timezone abbreviations
    for tz in tz_abbreviations:
        dt_str = dt_str.replace(tz, '')

    # Remove any extra spaces that may be left after removal
    dt_str = re.sub(' +', ' ', dt_str).strip()

    # Try to parse the datetime
    try:
        # Handle different formats
        if '/' in dt_str and ('AM' in dt_str or 'PM' in dt_str):
            # Check if it uses 24-hour format incorrectly labeled with PM
            match = re.search(r'(\d{1,2}):\d{2}:\d{2}', dt_str)
            if match:
                hour = int(match.group(1))
                if hour > 12 and 'PM' in dt_str:
                    dt_str = dt_str.replace(' PM', '')  # Remove incorrect PM
                    dt = pd.to_datetime(dt_str, format="%m/%d/%Y %H:%M:%S")
                else:
                    if ' 00:' in dt_str and 'AM' in dt_str:
                        dt_str = dt_str.replace(' 00:', ' 12:')
                    dt = pd.to_datetime(dt_str, format="%m/%d/%Y %I:%M:%S %p")
        else:
            dt = pd.to_datetime(dt_str).tz_localize(None)
    except ValueError:
        print("ValueError:", dt_str)
        dt = np.nan  # Handle cases with unrecognized formats

    return dt


def load_demographics_data(file_path):
    """Load demographics data from OpenAPS Data Commons Excel file."""
    demographics_file = f"{file_path}OpenAPS Data Commons_demographics-n-231.xlsx"

    if not os.path.exists(demographics_file):
        print("Warning: OpenAPS demographics file not found")
        return pd.DataFrame()

    try:
        df_demo = pd.read_excel(demographics_file)
        # Rename the ID column for easier access
        id_col = 'Your OpenHumans OpenAPS Data Commons "project member ID"'
        df_demo = df_demo.rename(columns={id_col: 'id'})

        # Ensure 'id' column is string
        df_demo['id'] = df_demo['id'].astype(str)

        # Ensure dat columns are datetime
        for col in ['Timestamp', 'When were you diagnosed with diabetes?', 'When were you born?']:
            df_demo[col] = pd.to_datetime(df_demo[col], errors='coerce')

        return df_demo
    except Exception as e:
        print(f"Error loading demographics file: {e}")
        return pd.DataFrame()

def add_demographics_to_df(file_path, df_merged):
    # Load demographics data for additional columns
    demographics_data = load_demographics_data(file_path)
    df_merged = df_merged.reset_index().sort_values(['id', 'date'])
    print(df_merged)

    # Fill in demographics data per subject if available
    if not demographics_data.empty:
        for subject_id in df_merged['id'].unique():
            demo_row = demographics_data[demographics_data['id'].astype(str) == str(subject_id)]
            if not demo_row.empty:
                row = demo_row.iloc[0]

                # Parse height (various formats possible)
                height = parse_height(row.get('How tall are you?'))

                # Parse weight (various formats possible)
                weight = parse_weight(row.get('How much do you weigh?'), height)

                # Get birth year for dynamic age calculation
                birth_year = extract_year(row.get('When were you born?'))
                diagnosis_year = extract_year(row.get('When were you diagnosed with diabetes?'))

                # There is this weird thing were some birth years are in the future / very close to the registration
                if birth_year is None:
                    birth_year = np.nan
                else:
                    if birth_year >= extract_year(row.get('Timestamp')):
                        birth_year = np.nan
                        print(f"Subject {subject_id} has invalid birth date, set to NaN instead.")

                if diagnosis_year is None:
                    diagnosis_year = np.nan
                elif diagnosis_year < birth_year:
                    diagnosis_year = np.nan
                    print(f"WARNING: Subject {subject_id} has invalid date of diagnosis")

                age_of_diagnosis = diagnosis_year - birth_year

                # Get and clean algorithm information
                algorithm_raw = row.get(
                    'What type of DIY close loop technology do you use? Select all that you actively use:')
                algorithm = clean_algorithm(algorithm_raw)

                gender = row.get('Gender')
                ethnicity_raw = row.get('Ethnicity origin:')
                ethnicity = standardize_ethnicity(ethnicity_raw)

                # Set demographics for all records of this subject
                df_merged.loc[df_merged['id'] == subject_id, 'weight'] = weight
                df_merged.loc[df_merged['id'] == subject_id, 'height'] = height
                df_merged.loc[df_merged['id'] == subject_id, 'insulin_delivery_algorithm'] = algorithm
                df_merged.loc[df_merged['id'] == subject_id, 'gender'] = gender
                df_merged.loc[df_merged['id'] == subject_id, 'ethnicity'] = ethnicity

                # Calculate dynamic age for each row based on time series date
                if birth_year:
                    subject_mask = df_merged['id'] == subject_id
                    subject_dates = df_merged.loc[subject_mask, 'date']
                    ages = subject_dates.dt.year - birth_year
                    df_merged.loc[subject_mask, 'age'] = ages
                    df_merged.loc[subject_mask, 'age_of_diagnosis'] = age_of_diagnosis

                    # Age of diagnosis cannot exceed age
                    invalid_mask = subject_mask & (df_merged['age_of_diagnosis'] > df_merged['age'])
                    df_merged.loc[invalid_mask, 'age_of_diagnosis'] = np.nan

    return df_merged

def parse_weight(weight_str, height):
    """Parse weight from various string formats to pounds."""
    if pd.isna(weight_str):
        return np.nan

    weight_str = str(weight_str).lower().strip()

    # Extract numbers
    import re
    numbers = re.findall(r'\d+\.?\d*', weight_str)
    if not numbers:
        return np.nan

    weight = float(numbers[0])

    # Convert kg to lbs
    if weight_str == '242 (110 kg)':
        weight = 242
    elif 'kg' in weight_str:
        weight = weight * 2.20462
    # Convert detected outliers
    elif (weight <= 80) and (height > 5.9):
        weight = weight * 2.20462

    # Otherwise assume lbs
    return weight

def parse_height(height_str):
    """Parse height from various string formats to feet."""
    if pd.isna(height_str):
        return np.nan

    height_str = str(height_str).lower().strip()

    # Extract numbers
    import re
    numbers = re.findall(r'\d+\.?\d*', height_str)
    if not numbers:
        return np.nan

    height = float(numbers[0])

    # Handle feet and inches (e.g., "5'10", "5 feet 10 inches")
    height_str = height_str.replace('’', "'").replace('‘', "'").replace('´', "'").replace('”', '"').replace('“', '"')
    height_str = height_str.replace(' ', '')  # remove spaces
    if "'" in height_str or 'feet' in height_str or '"' in height_str or '″' in height_str:
        feet_match = re.search(r'(\d+)', height_str)
        inches_match = re.search(r'(\d+)(?:\s*inch|")', height_str)

        if feet_match:
            feet = float(feet_match.group(1))
            inches = float(inches_match.group(1)) if inches_match else 0

        # Manually handle wrongly formatted input-values
        if height_str == "5'11.5\"":
            feet = 5
            inches = 11.5
        elif height_str == "5'11":
            feet = 5
            inches = 11
        elif height_str == "3'7''":
            feet = 3
            inches = 7
        elif height_str == "5'8.5''":
            feet = 5
            inches = 8.5
        elif height_str == '5"8"':
            feet = 5
            inches = 8
        elif height_str == "5'8":
            feet = 5
            inches = 8
        elif height_str == "5'10":
            feet = 5
            inches = 10
        elif height_str == "6″0":
            feet = 6
            inches = 0
        elif height_str == "5'6'":
            feet = 5
            inches = 6
        elif height_str == "6'0''":
            feet = 6
            inches = 0
        elif height_str == "5'3''":
            feet = 5
            inches = 3
        elif height_str == "6'2''":
            feet = 6
            inches = 2
        elif height_str == "5'8''":
            feet = 5
            inches = 8
        elif height_str == "5'7''":
            feet = 5
            inches = 7
        elif height_str == "5'2":
            feet = 5
            inches = 2
        elif height_str == "5'8.6\"":
            feet = 5
            inches = 8.6
        elif height_str == "5'3":
            feet = 5
            inches = 3
        elif height_str == '6"0"':
            feet = 6
            inches = 0
        elif height_str == "5'7":
            feet = 5
            inches = 7
        elif height_str == "5,9":
            feet = 5
            inches = 9
        elif height_str == "6'13\"(187cm)":
            return round(187 * 0.0328084, 2)
        elif height_str == "6'4''":
            feet = 6
            inches = 4
        elif height_str == "6'":
            feet = 6
            inches = 0
        elif height_str == "5'77\"":
            return np.nan
        elif height_str == "5'11'":
            feet = 5
            inches = 11
        elif height_str == "5'11'":
            feet = 5
            inches = 11
        elif height_str == "5,7":
            feet = 5
            inches = 7
        elif height_str == "66'54\"":
            return np.nan
        elif height_str == "5'55\"":
            return np.nan

        height = feet + inches / 12.0

    # Handle cm
    elif 'cm' in height_str:
        cm = float(numbers[0])
        height = cm * 0.0328084

    # Handle just inches
    elif 'inch' in height_str:
        inches = float(numbers[0])
        height = inches / 12.0

    # Cm determined based on unlikely feet numbers
    elif height > 9:
        # There is one person that likely has entered in inches (based on age and weight)
        if height == 70:
            height = 70 / 12
        else:
            cm = float(numbers[0])
            height = cm * 0.0328084

    height = round(height, 2)

    # Default assume it's already in feet or a decimal feet value
    return height

def calculate_age(birth_info):
    """Calculate age from birth year information."""
    if pd.isna(birth_info):
        return np.nan

    import re
    birth_str = str(birth_info).strip()

    # Extract year
    year_match = re.search(r'(19|20)\d{2}', birth_str)
    if year_match:
        birth_year = int(year_match.group())
        # Calculate age as of 2020 (approximate median year of OpenAPS data)
        return 2020 - birth_year

    return np.nan

def clean_algorithm(algorithm_raw):
    """Clean and shorten algorithm descriptions."""
    if pd.isna(algorithm_raw):
        return np.nan

    algorithm_str = str(algorithm_raw).strip()

    # Define mapping for algorithm cleaning
    algorithm_map = {
        'A "traditional" OpenAPS rig using the oref0 algorithm (i.e. using a Raspberry Pi/Carelink; or an Edison/Explorer Board; etc.)': 'oref0',
        'OpenAPS using the oref1 algorithm and hard-wired "on" SMB/UAM': 'oref1',
        'Using UMA but not SMB from oref1': 'oref1',
        'OpenAPS Oref1': 'oref1',
        'Loopkit/Loop': 'LoopAlgorithm',
        'AndroidAPS': 'AndroidAPS',
        'Haps': 'Haps'
    }

    # Split by comma if multiple algorithms
    algorithms = [alg.strip() for alg in algorithm_str.split(',')]
    cleaned_algorithms = []

    for algo in algorithms:
        # Find the best match in our mapping
        cleaned = algorithm_map.get(algo, algo)
        if cleaned not in cleaned_algorithms:  # Avoid duplicates
            cleaned_algorithms.append(cleaned)

    result = ', '.join(cleaned_algorithms) if cleaned_algorithms else np.nan
    if result == 'oref0, AndroidAPS':
        # remove redundancy
        result = 'oref0'
    print(f"BEFORE: {algorithm_raw}, AFTER: {result}")

    return ', '.join(cleaned_algorithms) if cleaned_algorithms else np.nan


def standardize_ethnicity(ethnicity_raw):
    """Standardize ethnicity values according to specified mappings."""
    if pd.isna(ethnicity_raw):
        return np.nan

    ethnicity_str = str(ethnicity_raw).strip()

    # Define ethnicity standardization mapping
    ethnicity_map = {
        'Ashkenaz Jewish': 'White',
        'I prefer not to answer': np.nan,
        "halfbreed' white/latino": 'White, Hispanic/Latino',
        '1/2 Spanish, 1/4 German, 1/4 Prussian': 'White',
        # Keep existing values as they are
        'White': 'White',
        'Asian/Pacific Islander': 'Asian/Pacific Islander',
        'Two or more races': 'Two or more races',
        'Mixed Race': 'Mixed Race'
    }

    return ethnicity_map.get(ethnicity_str, ethnicity_str)


def parse_age_of_diagnosis(self, diagnosis_info, birth_info):
    """Parse age of diagnosis from various formats, using birth year and diagnosis year when possible."""
    import re

    # First try to calculate from birth year and diagnosis year
    if pd.notna(diagnosis_info) and pd.notna(birth_info):
        # Extract years from both fields
        diagnosis_year = self.extract_year(diagnosis_info)
        birth_year = self.extract_year(birth_info)

        if diagnosis_year and birth_year:
            age_at_diagnosis = diagnosis_year - birth_year
            if 0 <= age_at_diagnosis <= 100:  # Reasonable age range
                return float(age_at_diagnosis)

    # Fallback to parsing age directly from diagnosis info
    if pd.isna(diagnosis_info):
        return np.nan

    diagnosis_str = str(diagnosis_info).lower()

    # Look for age patterns (e.g., "when I was 25 years old")
    age_match = re.search(r'(\d+)\s*(?:years?\s*old|yrs?)', diagnosis_str)
    if age_match:
        return float(age_match.group(1))

    # Look for just numbers if it's a simple age
    number_match = re.search(r'^\d+$', diagnosis_str.strip())
    if number_match:
        age = int(number_match.group())
        if 0 <= age <= 100:  # Reasonable age range
            return float(age)

    return np.nan

def extract_year(date_info):
    """Extract year from various date formats."""
    if pd.isna(date_info):
        return None

    import re
    date_str = str(date_info)

    # Look for 4-digit years (1900-2099)
    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        return int(year_match.group())

    return None


def main():
    input_path = 'data/raw/OpenAPS/'
    parser = Parser()

    # Process data
    try:
        df = parser(input_path)

        if df.empty:
            print("No data was processed.")
            return

        # Delete meals below 0g of carbs
        num_outlier_carbs = len(df[df['carbs'] < 0])
        num_carbs = len(df[df['carbs'] != 0])
        print(f'Found {num_outlier_carbs}/{num_carbs} = {round(num_outlier_carbs / num_carbs * 100, 1)}%')
        df.loc[df['carbs'] < 0, 'carbs'] = np.nan

        # Save processed data
        df.to_csv("OpenAPS.csv")

    except Exception as e:
        print(f"Error processing OpenAPS data: {e}")
        return None

    #merged_df = pd.read_csv('data/raw/OpenAPS.csv')
    #merged_df['date'] = pd.to_datetime(merged_df['date'])
    #merged_df = add_demographics_to_df(input_path, merged_df)
    #merged_df.to_csv('OpenAPS.csv')


if __name__ == "__main__":
    main()


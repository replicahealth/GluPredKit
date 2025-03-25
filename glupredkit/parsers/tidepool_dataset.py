"""
The Tidepool Dataset parser is processing the data from the Researcher datasets and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import pandas as pd
import os
import numpy as np
from datetime import timedelta


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the tidepool dataset root folder.
        """
        # TODO: Save PA, SAP, and HCL separately?
        # TODO: Insulin types?
        file_paths = {
            #'HCL150': ['Tidepool-JDRF-HCL150-train', 'Tidepool-JDRF-HCL150-test'],
            #'SAP100': ['Tidepool-JDRF-SAP100-train', 'Tidepool-JDRF-SAP100-test'],
            'PA50': ['Tidepool-JDRF-PA50-train', 'Tidepool-JDRF-PA50-test']
        }
        all_dfs, all_ids, is_test_bools = [], [], []
        for prefix, folders in file_paths.items():
            for folder in folders:
                # TODO: Each subject should be taken train/test chronologically so that we can "safe merge" them!
                current_file_path = os.path.join(file_path, folder, 'train-data' if 'train' in folder else 'test-data')
                is_test = True if 'test' in folder else False
                all_dfs, all_ids, is_test_bools = get_dfs_and_ids(current_file_path, all_dfs, all_ids, is_test_bools, is_test, id_prefix=f'{prefix}-')

        processed_dfs = []
        for index, df in enumerate(all_dfs):
            df_glucose, df_bolus, df_basal, df_carbs, df_workouts = self.get_dataframes(df)
            df_resampled = self.resample_data(df_glucose, df_bolus, df_basal, df_carbs, df_workouts)
            df_resampled['id'] = all_ids[index]
            df_resampled['is_test'] = is_test_bools[index]
            processed_dfs.append(df_resampled)

        # TODO: add some validation here, for 5-min intervals, sorting, merging of test-train like in ohio
        # TODO: sort by subject, and date

        df_final = pd.concat(processed_dfs)
        return df_final

    def resample_data(self, df_glucose, df_bolus, df_basal, df_carbs, df_workouts):
        # Ensure the index is sorted and is correctly set as date without nan
        for df in [df_glucose, df_bolus, df_basal, df_carbs, df_workouts]:
            if not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    print(f"Index is not correctly set as date")
                if df.index.isna().any():
                    print(f"Index contains invalid datetime entries (NaT)")
                if not df.index.is_monotonic_increasing:
                    df.sort_index(inplace=True)

        df = df_glucose.copy()
        df = df['CGM'].resample('5min', label='right').mean()

        if not df_carbs.empty:
            df_carbs = df_carbs.resample('5min', label='right').sum()
            df = pd.merge(df, df_carbs, on="date", how='outer')
        else:
            print("Subject with no carbohydrates")

        # TODO: Extended boluses?
        if not df_bolus.empty:
            df_bolus = df_bolus.resample('5min', label='right').sum()
            df = pd.merge(df, df_bolus, on="date", how='outer')
        else:
            print("Subject with no boluses")

        if not df_basal.empty:
            df_basal = df_basal.resample('5min', label='right').sum()
            df = pd.merge(df, df_basal, on="date", how='outer')
        else:
            print("Subject with no basals")

        if not df_workouts.empty:
            df_workout_labels = df_workouts['workout_label'].resample('5min', label='right').last()
            df_workout_labels = df_workouts['workout_duration'].resample('5min', label='right').sum()
            df_calories_burned = df_workouts['calories_burned'].resample('5min', label='right').sum()
            df = pd.merge(df, df_workout_labels, on="date", how='outer')
            df = pd.merge(df, df_calories_burned, on="date", how='outer')

        df['insulin'] = df['bolus'] + df['basal'] / 12

        # Ensuring homogenous time intervals
        df.sort_index(inplace=True)
        df = df.resample('5min').asfreq()

        time_diffs = df.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=5)
        valid_intervals = (time_diffs[1:] == expected_interval).all()
        if not valid_intervals:
            invalid_intervals = time_diffs[time_diffs != expected_interval]
            print(f"invalid time intervals found:", invalid_intervals)

        return df

    def get_dataframes(self, df):
        # We find the subject time zone offset by finding the offset most often present, if any. If not, use UTC time
        # Future work should look at whether the local times are safe to use, validate that offsets are consistent across datatypes
        # If yes, we could just use local time when available, and default back to UTC when unavailable
        grouped = df.groupby('est.timezoneOffset').size().reset_index(name='count')
        if grouped.empty:
            ts_offset = 0.0
        else:
            ts_offset = grouped.loc[grouped['count'].idxmax(), 'est.timezoneOffset']
        print("USER OFFSET", ts_offset)

        df['time'] = pd.to_datetime(df['time'], format='ISO8601', utc=True)
        df['time'] = df['time'].dt.tz_localize(None)
        df['date'] = df['time']
        df['date'] = df.apply(
            lambda row: row['date'] + pd.to_timedelta(ts_offset, unit='m'),
            axis=1
        )

        # Dataframe blood glucose
        # cbg = continuous blood glucose, smbg = self-monitoring of blood glucose
        #df_glucose = df[df['type'] == 'cbg'][['time', 'units', 'value']]
        df_glucose = df[df['type'].isin(['cbg', 'smbg'])][['date', 'units', 'value']]
        df_glucose['value'] = df_glucose.apply(
            lambda row: row['value'] * 18.0182 if row['units'] == 'mmol/L' else row['value'], axis=1)
        df_glucose.rename(columns={"value": "CGM"}, inplace=True)
        df_glucose.drop(columns=['units'], inplace=True)
        df_glucose.sort_values(by='date', inplace=True, ascending=True)
        df_glucose.set_index('date', inplace=True)

        # Dataframe bolus doses
        df_bolus = df[df['type'] == 'bolus'][['date', 'normal']]
        df_bolus.rename(columns={"normal": "bolus"}, inplace=True)
        df_bolus.sort_values(by='date', inplace=True, ascending=True)
        df_bolus.set_index('date', inplace=True)

        # Dataframe basal rates
        df_basal = df[df['type'] == 'basal'][['date', 'duration', 'rate', 'units', 'deliveryType']].copy()
        df_basal['duration'] = df_basal['duration'] / 1000  # convert to seconds
        df_basal['duration'] = df_basal['duration'].fillna(0)
        df_basal.rename(columns={"rate": "basal"}, inplace=True)
        df_basal.sort_values(by='date', inplace=True, ascending=True)
        df_basal.loc[df_basal['deliveryType'] == 'suspend', 'basal'] = 0.0  # Set suspend values to 0.0
        # Remove duplicate start dates, we choose the first as we cannot know exactly what to do
        df_basal = df_basal.drop_duplicates(subset=['date'])
        # Manipulate duration to match with the next sample if the id is the same, if overlapping
        df_basal['end_date'] = df_basal['date'] + pd.to_timedelta(df_basal['duration'], unit='s')  # Get end date
        df_basal['next_date'] = df_basal['date'].shift(-1)  # Get next row's date
        df_basal['duration'] = df_basal.apply(
            lambda row: (row['next_date'] - row['date']).total_seconds() if pd.notna(row['next_date']) and row['end_date'] >= row['next_date']
            else row['duration'],
            axis=1
        )
        # We need to split each sample into five minute intervals, and create new rows for each five minutes
        expanded_rows = []
        for _, row in df_basal.iterrows():
            intervals = split_basal_into_intervals(row)
            expanded_rows.extend(intervals)
        df_basal = pd.DataFrame(expanded_rows)
        df_basal.set_index('date', inplace=True)

        # Dataframe carbohydrates
        df_carbs = pd.DataFrame()
        carb_dfs = []
        for carb_col in ['nutrition.carbohydrate.net', 'carbInput']:
            if carb_col in df.columns:
                df_carbs = df[df['type'] == 'food'][['date', carb_col]]
                df_carbs.rename(columns={"nutrition.carbohydrate.net": "carbs"}, inplace=True)
                carb_dfs += [df_carbs]

        if len(carb_dfs) == 0:
            assert ValueError("No carbohydrate data detected for subject! Inspect data.")
        else:
            df_carbs = pd.concat(carb_dfs)
        df_carbs.sort_values(by='date', inplace=True, ascending=True)
        df_carbs.set_index('date', inplace=True)
        print("CARBS DF", df_carbs)

        # Dataframe workouts
        df_workouts = pd.DataFrame
        if 'activityName' in df.columns:
            df_workouts = df.copy()[df['activityName'].notna()]
            if not df_workouts.empty:
                if 'activityDuration.value' in df_workouts.columns:
                    df_workouts.rename(columns={"activityName": "workout_label",
                                                "activityDuration.value": "workout_duration"}, inplace=True)
                    df_workouts.sort_values(by='date', inplace=True, ascending=True)

                    if 'energy.value' in df_workouts.columns:
                        df_workouts.rename(columns={"energy.value": "calories_burned"}, inplace=True)
                    else:
                        df_workouts['calories_burned'] = np.nan
                    df_workouts.set_index('date', inplace=True)
                    df_workouts = df_workouts[['workout_label', 'calories_burned', 'workout_duration']]
                else:
                    print("No duration registered for physical activity!")

        return df_glucose, df_bolus, df_basal, df_carbs, df_workouts


def split_basal_into_intervals(row):
    start = row.date
    end = start + timedelta(seconds=row['duration'])
    total_insulin_delivered = row['basal'] * row['duration'] / 60
    interval = timedelta(minutes=5)

    # Track rows for the new DataFrame
    rows = []
    current = start

    while current < end:
        next_interval = min(current + interval, end)
        # Calculate the proportion of time in the 5-minute bin
        proportion = (next_interval - current) / timedelta(minutes=row['duration'])
        bin_value = total_insulin_delivered * proportion

        # Add the row to the list
        rows.append({
            'date': current,
            'basal': bin_value * 12  # Convert back to U/hr when we are done
        })
        current = next_interval

    return rows


def get_dfs_and_ids(file_path, all_dfs, all_ids, is_test_bools, is_test, id_prefix):
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.csv'):
                subject_file_path = os.path.join(root, file)
                df = pd.read_csv(subject_file_path, low_memory=False)
                all_dfs.append(df)

                subject_id = id_prefix + file.split("_")[1].split(".")[0]
                all_ids.append(subject_id)

                is_test_bools.append(is_test)
    return all_dfs, all_ids, is_test_bools

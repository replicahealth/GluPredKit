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

    def __call__(self, prefix: str, file_path: str, *args):
        """
        file_path -- the file path to the tidepool dataset root folder.
        """
        folders = [f'Tidepool-JDRF-{prefix}-train', f'Tidepool-JDRF-{prefix}-test']
        processed_dfs = []

        def get_file_path(folder):
            is_test = 'test' in folder
            return os.path.join(file_path, folder, 'test-data' if is_test else 'train-data')

        train_dfs_dict = get_dfs_and_ids(get_file_path(folders[0]),  id_prefix=f'{prefix}-')
        test_dfs_dict = get_dfs_and_ids(get_file_path(folders[1]),  id_prefix=f'{prefix}-')

        # Validating that the train / test subjects are the same
        if set(train_dfs_dict.keys()) == set(test_dfs_dict.keys()):
            print(f"✅ The dictionaries in {prefix} have the same keys. Total keys: {len(set(train_dfs_dict.keys()))}")
        else:
            print(f"❌ The dictionaries have different keys.")

        count = 0
        for subject_id in train_dfs_dict.keys():
            subject_train_df = train_dfs_dict[subject_id]
            subject_test_df = test_dfs_dict[subject_id]

            # add is_test column
            subject_train_df['is_test'] = False
            subject_test_df['is_test'] = True

            def get_resampled_df(df):
                df_glucose, df_bolus, df_basal, df_carbs, df_workouts = self.get_dataframes(df)
                df_resampled = self.resample_data(df_glucose, df_bolus, df_basal, df_carbs, df_workouts)
                return df_resampled

            df_training = get_resampled_df(subject_train_df)
            df_testing = get_resampled_df(subject_test_df)

            merged_df = df_testing.combine_first(df_training)
            merged_df = merged_df.sort_index()

            # Ensure 5-min intervals after merging train and test
            merged_df = merged_df.resample('5min').asfreq()
            time_diffs = merged_df.index.to_series().diff()
            expected_interval = pd.Timedelta(minutes=5)
            valid_intervals = (time_diffs[1:] == expected_interval).all()
            if not valid_intervals:
                invalid_intervals = time_diffs[time_diffs != expected_interval]
                print(f"invalid time intervals found:", invalid_intervals)

            age, gender, age_of_diagnosis = get_age_and_diagnosis(file_path, prefix, subject_id)
            merged_df['age'] = age
            merged_df['gender'] = gender
            merged_df['age_of_diagnosis'] = age_of_diagnosis
            merged_df['id'] = subject_id

            #print(merged_df)
            processed_dfs.append(merged_df)

            count += 1
            print(f"{count}: Finished processing subject {subject_id}")

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
            df_workout_label = df_workouts['workout_label'].resample('5min', label='right').last()
            df_workout_duration = df_workouts['workout_duration'].resample('5min', label='right').sum()
            df_calories_burned = df_workouts['calories_burned'].resample('5min', label='right').sum()
            df = pd.merge(df, df_workout_label, on="date", how='outer')
            df = pd.merge(df, df_workout_duration, on="date", how='outer')
            df = pd.merge(df, df_calories_burned, on="date", how='outer')

        df['basal'] = df['basal'] / 12
        df['insulin'] = df['bolus'].fillna(0) + df['basal']

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
        # The most strict (but maybe necessary) alternative would be to skip all samples without local time (around 1/3)
        grouped = df.groupby('est.timezoneOffset').size().reset_index(name='count')
        if grouped.empty:
            ts_offset = 0.0
        else:
            ts_offset = grouped.loc[grouped['count'].idxmax(), 'est.timezoneOffset']

        df['time'] = pd.to_datetime(df['time'], utc=True)
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
        df_bolus.set_index('date', inplace=True)

        if 'extended' in df.columns:
            expanded_rows = []
            for _, row in df[df['extended'].notna()].iterrows():
                intervals = split_extended_bolus_into_intervals(row)
                expanded_rows.extend(intervals)

            if not len(expanded_rows) == 0:
                df_extended_boluses = pd.DataFrame(expanded_rows)
                df_extended_boluses.set_index('date', inplace=True)
                df_bolus = pd.concat([df_bolus, df_extended_boluses])
        df_bolus.sort_values(by='date', inplace=True, ascending=True)

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
        possible_carb_cols = [col for col in df.columns if col in ['nutrition.carbohydrate.net', 'carbInput']]
        main_carb_col = possible_carb_cols[0]
        if len(possible_carb_cols) > 1:
            # If there are available carbs in two columns, choose the column with more samples
            # The reason we do not merge them is to avoid duplicate samples
            n_samples_prev = 0
            for col in possible_carb_cols:
                df_carbs = df[['date', col]]
                df_carbs = df_carbs.dropna(subset=[col])
                print(f"{col}: {len(df_carbs)}")
                if len(df_carbs) > n_samples_prev:
                    main_carb_col = col
                    n_samples_prev = len(df_carbs)
        df_carbs = df[['date', main_carb_col]]
        df_carbs = df_carbs.dropna(subset=[main_carb_col])

        n_carbs_with_duplicates = len(df_carbs)
        df_carbs = df_carbs.drop_duplicates(subset=['date', main_carb_col], keep='first')
        if n_carbs_with_duplicates > len(df_carbs):
            print(f"Warning: Removed {n_carbs_with_duplicates - len(df_carbs)} when dropping duplicate carb samples")

        df_carbs.rename(columns={main_carb_col: "carbs"}, inplace=True)
        df_carbs.sort_values(by='date', inplace=True, ascending=True)
        df_carbs.set_index('date', inplace=True)

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
    duration_minutes = row['duration'] / 60
    end = start + timedelta(minutes=duration_minutes)
    total_insulin_delivered = row['basal'] / 60 * duration_minutes  # U /hr / 60 -> U/min, *min --> U
    interval = timedelta(minutes=5)

    # Track rows for the new DataFrame
    rows = []
    current = start

    while current < end:
        next_interval = min(current + interval, end)
        # Calculate the proportion of time in the 5-minute bin
        proportion = (next_interval - current) / timedelta(minutes=duration_minutes)
        bin_value = total_insulin_delivered * proportion

        # Add the row to the list
        rows.append({
            'date': current,
            'basal': bin_value * 12  # From U to U/hr
        })
        current = next_interval

    return rows


def split_extended_bolus_into_intervals(row):
    start = row.date
    duration = row['duration'] / 1000
    end = start + timedelta(seconds=duration)
    interval = timedelta(minutes=5)

    # Track rows for the new DataFrame
    rows = []
    current = start

    while current < end:
        next_interval = min(current + interval, end)
        # Calculate the proportion of time in the 5-minute bin
        proportion = (next_interval - current) / timedelta(seconds=duration)
        bin_value = row['extended'] * proportion

        # Add the row to the list
        rows.append({
            'date': current,
            'bolus': bin_value
        })
        current = next_interval

    return rows


def get_dfs_and_ids(file_path, id_prefix):
    dfs_dict = {}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.csv'):
                subject_file_path = os.path.join(root, file)
                df = pd.read_csv(subject_file_path, low_memory=False)
                subject_id = id_prefix + file.split("_")[1].split(".")[0]
                dfs_dict[subject_id] = df
    return dfs_dict


def get_age_and_diagnosis(file_path, prefix, subject_id):
    folder = f'Tidepool-JDRF-{prefix}-test'
    file_name = f'{prefix}-test-metadata-summary.csv'
    full_subject_id = f'test_{subject_id.split("-")[1]}.csv'
    file_path = os.path.join(file_path, folder, file_name)
    metadata_df = pd.read_csv(file_path)
    subject_data = metadata_df[metadata_df['file_name'] == full_subject_id]
    
    # Extract age and gender
    age = subject_data['ageStart'].iloc[0]
    gender = subject_data['biologicalSex'].iloc[0]
    gender_map = {
        'female': 'Female',
        'male': 'Male'
    }
    if pd.notna(gender):
        gender = gender_map[gender.lower()]
    
    # Calculate age_of_diagnosis from ageStart - yearsLivingWithDiabetesStart
    age_of_diagnosis = None
    if 'yearsLivingWithDiabetesStart' in subject_data.columns:
        years_living_with_diabetes = subject_data['yearsLivingWithDiabetesStart'].iloc[0]
        if pd.notna(age) and pd.notna(years_living_with_diabetes):
            age_of_diagnosis = age - years_living_with_diabetes
            age_of_diagnosis = max(0, age_of_diagnosis)  # Ensure non-negative
    
    return age, gender, age_of_diagnosis


def main():
    input_path = 'data/raw/Tidepool/'
    parser = Parser()

    # Process data
    for prefix in ['HCL150', 'PA50', 'SAP100']:
        df = parser(prefix, input_path)

        # Save processed data
        df.to_csv(f"Tidepool-JDRF-{prefix}.csv")


if __name__ == "__main__":
    main()




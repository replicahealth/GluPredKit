"""
DiaTrend parser for processing DiaTrend dataset.

The DiaTrend parser processes DiaTrend data and returns merged data on the same time grid.
"""

from .base_parser import BaseParser
import pandas as pd
import numpy as np
import re
import os


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the DiaTrend dataset root folder.
        """
        print(f"Processing BrisT1D data from {file_path}")

        demographics = self.get_demographics(file_path)

        all_resampled_dfs = []
        for i in range(24):
            if i+1 < 10:
                subject_id = f'P0{i+1}'
            else:
                subject_id = f'P{i+1}'
            subject_file_path = f'{file_path}/device_data/processed_state/{subject_id}.csv'

            print("Processing subject: ", subject_id)

            if not os.path.exists(subject_file_path):
                # skip this iteration
                continue

            if subject_id == 'P17':
                # Skip subject due to missing insulin data
                continue

            df = pd.read_csv(subject_file_path)
            resampled_df = self.resample_df(df, subject_id=subject_id)

            subject_demographics = demographics[demographics['id'] == (i+1)].iloc[0]
            demographics_cols = ['age', 'age_of_diagnosis', 'gender', 'ethnicity']
            for col in demographics_cols:
                resampled_df[col] = subject_demographics[col]

            all_resampled_dfs.append(resampled_df)

        merged_df = pd.concat(all_resampled_dfs, ignore_index=True)
        merged_df = self.get_insulin_delivery_modality_from_insulin_delivery_device(merged_df)
        merged_df = self.get_insulin_delivery_algorithm_from_insulin_delivery_device(merged_df)
        merged_df['source_file'] = 'BrisT1D'
        return merged_df

    def get_insulin_delivery_modality_from_insulin_delivery_device(self, df):
        mapping = {
            'Medtronic MiniMed 780G': 'AID',
            'Medtronic MiniMed 640G': 'SAP',
            # Tandem t:slim X2 could be using either SAP or AID
            'Omnipod Eros': 'SAP',
            'Omnipod Dash': 'SAP',
            'Omnipod 5': 'AID'
        }
        df['insulin_delivery_modality'] = df['insulin_delivery_device'].map(mapping)
        return df

    def get_insulin_delivery_algorithm_from_insulin_delivery_device(self, df):
        mapping = {
            'MiniMed 780G': '780G Advanced HCL',
            'MiniMed 640G': '640G',
            # Tandem t:slim X2 could be using either Control-IQ or Basal-IQ
            'Omnipod Eros': 'Basal-Bolus',
            'Omnipod Dash': 'Basal-Bolus',
            'Omnipod 5': 'Omnipod 5'
        }
        df['insulin_delivery_algorithm'] = df['insulin_delivery_device'].map(mapping)
        return df

    def get_demographics(self, file_path):
        df_demographics = pd.read_csv(f'{file_path}/demographic_data.csv', encoding='latin1')


        df_demographics['id'] = pd.to_numeric(df_demographics['Participant Number'])
        df_demographics['age'] = pd.to_numeric(df_demographics['How old are you?'])

        gender_map = {
            'Male': 'Male',
            'Male ': 'Male',
            'Female': 'Female',
            'Female ': 'Female',
            'Cis Female ': 'Female',
            'non binary': 'Non Binary',
            'Non-binary': 'Non Binary',
        }
        df_demographics['gender'] = df_demographics['What would you describe your gender as?'].map(gender_map)

        ethnicity_map = {
            'White British': 'White',
            'White British ': 'White',
            'white british': 'White',
            'White irish': 'White',
            'White ': 'White',
            'White': 'White',
            'White Scottish': 'White',
            'British': 'White',
            'White - British': 'White',
            'Indian': 'Asian',
        }
        df_demographics['ethnicity'] = df_demographics['What would you describe your ethnicity as?'].map(ethnicity_map)

        def parse_years(text):
            if pd.isna(text):
                return None

            # Remove "Just under" and strip whitespace
            text = text.replace('Just under', '').strip()

            # Handle "½" or "1/2"
            text = text.replace(' ½', '.5')

            # Extract the first number (can be float)
            match = re.search(r'(\d+(\.\d+)?)', text)
            if match:
                return float(match.group(1))  # convert to number
            return None

        df_demographics['years_since_diagnosis'] = df_demographics['How long have you had type 1 diabetes?'].apply(parse_years)
        df_demographics['age_of_diagnosis'] = df_demographics['age'] - df_demographics['years_since_diagnosis']

        df_demographics = df_demographics[['id', 'age', 'age_of_diagnosis', 'gender', 'ethnicity']]
        return df_demographics

    def resample_df(self, df, subject_id):
        """
        This function takes a subjects dataframe from the processed states, converts the units, resamples into 5-min
        intervals, and renames the columns to fit into our standardized format.
        """
        df['device'] = df['device'].replace({
            'Guardian 4': 'Medtronic Guardian 4',
            ' FreeStyle Libre 2': 'FreeStyle Libre 2'})
        unique_devices = df['device'].dropna().unique()
        cgm_devices = self.get_cgm_devices_from_devices(unique_devices)
        insulin_delivery_devices = self.get_insulin_delivery_devices_from_devices(unique_devices)

        # keep only if in allowed, else None
        df['cgm_device'] = df['device'].where(df['device'].isin(cgm_devices))
        df['insulin_delivery_device'] = df['device'].where(df['device'].isin(insulin_delivery_devices))
        df['insulin_delivery_device'] = df['insulin_delivery_device'].replace('Tandem t:slim X2', 't:slim X2')
        df['insulin_delivery_device'] = df['insulin_delivery_device'].replace('Medtronic MiniMed 640G', 'MiniMed 640G')
        df['insulin_delivery_device'] = df['insulin_delivery_device'].replace('Medtronic MiniMed 780G', 'MiniMed 780G')

        # Convert units
        df['bg'] = df['bg'] * 18.018

        # Resample blood glucose
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        resampled_df = (
            df.resample('5min')
            .agg({
                'bg': 'mean',
                'insulin': lambda x: x.sum(min_count=1),
                'carbs': lambda x: x.sum(min_count=1),
                'steps': lambda x: x.sum(min_count=1),
                'cals': lambda x: x.sum(min_count=1),
                'hr': 'mean',
                'activity': 'last',
                'cgm_device': 'last',
                'insulin_delivery_device': 'last',
            })
        )
        cols_to_fill = ['cgm_device', 'insulin_delivery_device']
        resampled_df[cols_to_fill] = resampled_df[cols_to_fill].ffill().bfill()

        # Checking that all values are preserved
        preserved_activities = (df['activity'].value_counts() == resampled_df['activity'].value_counts()).all()
        preserved_carbs = round(df['carbs'].sum()) == round(resampled_df['carbs'].sum())
        preserved_insulin = round(df['insulin'].sum()) == round(resampled_df['insulin'].sum())
        preserved_bg = round(df['bg'].mean()) == round(resampled_df['bg'].mean())
        preserved_steps = round(df['steps'].sum()) == round(resampled_df['steps'].sum())
        preserved_cals = round(df['cals'].sum()) == round(resampled_df['cals'].sum())
        preserved_hr = round(df['hr'].mean()) == round(resampled_df['hr'].mean())

        checks = {
            'activities': preserved_activities,
            'carbs': preserved_carbs,
            'insulin': preserved_insulin,
            'bg': preserved_bg,
            'steps': preserved_steps,
            'cals': preserved_cals,
            'hr': preserved_hr
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Preservation check failed for: {', '.join(failed)}")

        resampled_df.reset_index(inplace=True)
        resampled_df = resampled_df.rename(columns={
            'timestamp': 'date',
            'bg': 'CGM',
            'cals': 'calories_burned',
            'hr': 'heartrate',
            'activity': 'workout_label'
        })

        # Set negative insulin values to nan (there are two of them in the entire dataset)
        # We also set the following 8 hours of data after the negative dose to nan
        bad_idx = resampled_df.index[resampled_df['insulin'] < 0]
        if len(bad_idx) > 0:
            print(f"Warning: Subject {subject_id} has {len(bad_idx)} negative insulin values. "
                  "We set the value and the following eight hours of data to nan.")
            rows_to_nan = []
            for idx in bad_idx:
                loc = resampled_df.index.get_loc(idx)  # safe unless duplicates exist
                rows_to_nan.extend(range(loc, loc + 96))
            rows_to_nan = [i for i in rows_to_nan if i < len(resampled_df)]
            insulin_col = resampled_df.columns.get_loc('insulin')
            resampled_df.iloc[rows_to_nan, insulin_col] = np.nan

        resampled_df['id'] = subject_id

        return resampled_df

    def get_cgm_devices_from_devices(self, unique_devices):
        possible_values = ['FreeStyle Libre 2', 'Dexcom G6', 'Medtronic Guardian 4', 'Dexcom One']
        return self.get_devices(possible_values, unique_devices)

    def get_insulin_delivery_devices_from_devices(self, unique_devices):
        possible_values = ['Medtronic MiniMed 780G', 'Medtronic MiniMed 640G', 'Tandem t:slim X2', 'Omnipod Eros',
                           'Omnipod 5', 'Omnipod Dash']
        return self.get_devices(possible_values, unique_devices)

    def get_devices(self, possible_values, unique_devices):
        matched_devices = []
        clean_unique_devices = [val.strip() for val in unique_devices]
        for val in possible_values:
            if val in clean_unique_devices:
                matched_devices += [val]

        if len(matched_devices) >= 1:
            return matched_devices
        else:
            raise ValueError(f'No devices available among alternatives: {possible_values}')


def main():
    """
    Main function to run DiaTrend parser as a standalone script.
    """
    print("=== BrisT1D Dataset Parser ===")

    # File paths
    input_path = "data/raw/BrisT1D"

    # Initialize parser
    parser = Parser()

    # Process data
    try:
        df = parser(input_path)

        if df.empty:
            print("No data was processed.")
            return

        # Save processed data
        output_file = os.path.join(input_path, "BrisT1D.csv")
        df.to_csv(output_file, index=True)

        print(f"✓ Saved BrisT1D dataframe to: {output_file}")
        print(f"Dataset shape: {df.shape}")

    except Exception as e:
        print(f"Error processing BrisT1D data: {e}")
        return None

    return df


if __name__ == "__main__":
    df = main()

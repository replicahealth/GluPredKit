"""
DiaTrend parser for processing DiaTrend dataset.

The DiaTrend parser processes DiaTrend data and returns merged data on the same time grid.
"""

from .base_parser import BaseParser
import pandas as pd
import numpy as np
import re
import os


CGM_FOLDER = 'Glucose Data'
ACTIVITY_DATA_FOLDER = 'Activity Data'
MEALS_DATA_FOLDER = 'Nutrition Data'
BOLUS_FOLDER = 'Insulin Data/Bolus Data'
BASAL_FOLDER = 'Insulin Data/Basal Data'
DEMOGRAPHICS_FOLDER = 'Demographics'

class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the DiaTrend dataset root folder.
        """
        print(f"Processing T1D-UOM data from {file_path}")

        subject_ids = [f[-8:-4] for f in os.listdir(f'{file_path}/{CGM_FOLDER}/') if f.endswith(".csv")
                       if not f[-8:-4] in ['2303', '2320', '2404']]
        df_demographics = self.get_demographics_df(file_path)
        all_resampled_dfs = []
        for subject_id in subject_ids:
            print("Processing subject: ", subject_id)
            resampled_df = self.get_resampled_df_for_subject(file_path, subject_id, df_demographics)
            all_resampled_dfs.append(resampled_df)

        merged_df = pd.concat(all_resampled_dfs, ignore_index=True)
        merged_df['source_file'] = 'T1D-UOM'

        # Set the one 68 U bolus dose as basal instead
        merged_df.loc[merged_df['bolus'] > 60, 'basal'] = 68
        merged_df.loc[merged_df['bolus'] > 60, 'bolus'] = np.nan

        return merged_df

    def get_resampled_df_for_subject(self, file_path, subject_id, demographics_df):
        glucose_df = self.get_glucose_df(file_path, subject_id)
        meals_df = self.get_meals_df(file_path, subject_id)
        bolus_df = self.get_bolus_df(file_path, subject_id)
        basal_df = self.get_basal_df(file_path, subject_id)
        activity_df = self.get_activity_df(file_path, subject_id)

        if meals_df.empty:
            merged_df = glucose_df.copy()
            merged_df['carbs'] = np.nan
            merged_df['meal_label'] = np.nan
        else:
            merged_df = self.merge_df(glucose_df, meals_df)

        merged_df = self.merge_df(merged_df, bolus_df)
        merged_df = self.merge_df(merged_df, basal_df)
        merged_df = self.merge_df(merged_df, activity_df)

        merged_df['insulin'] = merged_df['bolus'].fillna(0) + merged_df['basal']

        # Add demographics
        subject_demographics = demographics_df[demographics_df['participant_id'] == subject_id].iloc[0]
        demographics_cols = ['weight', 'height']
        for col in demographics_cols:
            merged_df[col] = subject_demographics[col]
        subject_demographics = self.create_participant_data_dataframe(subject_id)
        demographics_cols = ['gender', 'age', 'insulin_delivery_device']
        for col in demographics_cols:
            merged_df[col] = subject_demographics[col]

        # Insulin type is the same for the entire subject dataset
        merged_df['insulin_type_basal'] = merged_df['insulin_type_basal'].ffill().bfill()

        # The paper states that: "This dataset includes participants using both multiple daily injections (MDI) and
        # insulin pumps operating in open-loop mode"
        merged_df['insulin_delivery_algorithm'] = merged_df['insulin_type_basal'].map({
            'Rapid-Acting': 'Basal-Bolus',
            'Long-Acting': 'MDI'
        })
        merged_df['insulin_delivery_modality'] = merged_df['insulin_type_basal'].map({
            'Rapid-Acting': 'SAP',
            'Long-Acting': 'MDI'
        })
        merged_df['id'] = subject_id
        merged_df.reset_index(inplace=True)
        return merged_df

    def merge_df(self, base_df, new_df):
        merged_df = base_df.merge(new_df, left_index=True, right_index=True,
                                  how='outer')  # keeps all datetimes from both
        return merged_df

    def get_glucose_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{CGM_FOLDER}/UoMGlucose{subject_id}.csv')
        df['bg_ts'] = pd.to_datetime(df['bg_ts'].str.strip(), format="%d/%m/%Y %H:%M")
        df['value'] = df['value'] * 18.018
        df.rename(columns={'bg_ts': 'date', 'value': 'CGM'}, inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({'CGM': 'last'})
        return resampled_df

    def get_meals_df(self, file_path, subject_id):
        file = f"{file_path}/{MEALS_DATA_FOLDER}/UoMNutrition{subject_id}.csv"

        if os.path.exists(file):
            df = pd.read_csv(file)
        else:
            print(f"File not found: {file}")
            return pd.DataFrame()
        df['meal_ts'] = pd.to_datetime(df['meal_ts'].str.strip(), format='mixed', dayfirst=True)
        df['meal_tag'] = df['meal_tag'].replace('Not Reported', None)
        df.rename(columns={'meal_ts': 'date', 'carbs_g': 'carbs', 'meal_tag': 'meal_label'}, inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({
            'carbs': lambda x: x.sum(min_count=1),
            'meal_label': 'last',
        })
        return resampled_df

    def get_bolus_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{BOLUS_FOLDER}/UoMBolus{subject_id}.csv')
        df['bolus_ts'] = pd.to_datetime(df['bolus_ts'].str.strip(), format="%d/%m/%Y %H:%M")
        df.rename(columns={'bolus_ts': 'date', 'bolus_dose': 'bolus'}, inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({
            'bolus': lambda x: x.sum(min_count=1),
        })
        return resampled_df

    def get_basal_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{BASAL_FOLDER}/UoMBasal{subject_id}.csv')
        df['basal_ts'] = pd.to_datetime(df['basal_ts'].str.strip(), format="%d/%m/%Y %H:%M")

        # If it is a pump-user, the unit is reported in U/hr, and we need to convert
        df.loc[df['insulin_kind'] == 'R', 'basal_dose'] /= 12

        df['insulin_kind'] = df['insulin_kind'].map({'R': 'Rapid-Acting', 'L': 'Long-Acting'})
        df.rename(columns={'basal_ts': 'date', 'basal_dose': 'basal', 'insulin_kind': 'insulin_type_basal'},
                  inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({
            'basal': lambda x: x.sum(min_count=1),
            'insulin_type_basal': 'last',
        })
        return resampled_df

    def get_activity_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{ACTIVITY_DATA_FOLDER}/UoMActivity{subject_id}.csv')
        df['activity_ts'] = pd.to_datetime(df['activity_ts'].str.strip(), format="%d/%m/%Y %H:%M")
        df['duration_s'] = df['duration_s'] / 60

        # Set workout duration, workout label to nan
        activity_mask = df['activity_type'] == 'SEDENTARY'
        df.loc[activity_mask, 'duration_s'] = np.nan
        df.loc[activity_mask, 'activity_type'] = None

        # Map workout to not be caps-lock
        df['activity_type'] = df['activity_type'].map({
            'WALKING': 'Walking',
            'RUNNING': 'Running',
            'GENERIC': 'Generic'})

        # If there is a row with the same date, prioritize the row where workout_label is not 'Generic'
        def fix_generic(group):
            unique_labels = group['activity_type'].dropna().unique()
            # If more than 1 unique label and 'Generic' is one of them, set 'Generic' to NaN
            if len(unique_labels) > 1 and 'Generic' in unique_labels:
                group.loc[group['activity_type'] == 'Generic', 'duration_s'] = np.nan
                group['activity_type'] = group['activity_type'].replace('Generic', np.nan)
            return group
        df = df.groupby('activity_ts', group_keys=False).apply(fix_generic)

        # Rename columns
        df.rename(columns={'activity_ts': 'date', 'active_Kcal': 'calories_burned',
                           'step_count': 'steps', 'duration_s': 'workout_duration',
                           'activity_type': 'workout_label'}, inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({
            'calories_burned': lambda x: x.sum(min_count=1),
            'steps': lambda x: x.sum(min_count=1),
            'workout_label': 'last',
            'workout_duration': lambda x: x.sum(min_count=1),
        })
        return resampled_df

    def get_demographics_df(self, file_path):
        df = pd.read_csv(f'{file_path}/{DEMOGRAPHICS_FOLDER}/UoMBMI.csv')
        df['participant_id'] = df['participant_id'].str[4:]  # Replace N with the number of characters

        # Convert to Ibs and feet
        df['weight_kg'] = df['weight_kg'] * 2.20462
        df['height_m'] = df['height_m'] * 3.28084

        df.rename(columns={'weight_kg': 'weight', 'height_m': 'height'}, inplace=True)
        return df[['participant_id', 'weight', 'height']]

    def create_participant_data_dataframe(self, subject_id):
        """
        Creates a pandas DataFrame of participant diabetes management data,
        excluding the 'Start Date', 'End Date', and 'TIR (%)' columns.

        This dataframe is created from the values in Table 2 in the Nature paper
        """
        data = [
            {'Participant ID': '2301', 'gender': 'Female', 'age': 25, 'Sensor': 'CGM',
             'insulin_delivery_device': 't:slim X2'},
            {'Participant ID': '2302', 'gender': 'Female', 'age': 29, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2303', 'gender': 'Female', 'age': 29, 'Sensor': 'CGM',
             'insulin_delivery_device': 'N/A'},
            {'Participant ID': '2304', 'gender': 'Female', 'age': 29, 'Sensor': 'Flash',
             'insulin_delivery_device': 'MiniMed 780G'},
            {'Participant ID': '2305', 'gender': 'Female', 'age': 50, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2306', 'gender': 'Female', 'age': 50, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2307', 'gender': 'Female', 'age': 61, 'Sensor': 'CGM',
             'insulin_delivery_device': 't:slim X2'},
            {'Participant ID': '2308', 'gender': 'Male', 'age': 59, 'Sensor': 'CGM',
             'insulin_delivery_device': 'MiniMed 780G'},
            {'Participant ID': '2309', 'gender': 'Female', 'age': 59, 'Sensor': 'CGM',
             'insulin_delivery_device': 'MiniMed 780G'},
            {'Participant ID': '2310', 'gender': 'Male', 'age': 70, 'Sensor': 'CGM',
             'insulin_delivery_device': 'MiniMed 780G'},
            {'Participant ID': '2313', 'gender': 'Male', 'age': 39, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2314', 'gender': 'Male', 'age': 61, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2320', 'gender': 'Female', 'age': 46, 'Sensor': 'CGM',
             'insulin_delivery_device': 'Omnipod 5'},
            {'Participant ID': '2401', 'gender': 'Male', 'age': 46, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2403', 'gender': 'Male', 'age': 23, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2404', 'gender': 'Female', 'age': 37, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'},
            {'Participant ID': '2405', 'gender': 'Male', 'age': 52, 'Sensor': 'Flash',
             'insulin_delivery_device': 'Multiple Daily Injections'}
        ]
        df = pd.DataFrame(data)
        return df[df['Participant ID'] == subject_id].iloc[0]


def main():
    """
    Main function to run DiaTrend parser as a standalone script.
    """
    print("=== T1D-UOM Dataset Parser ===")
    input_path = "data/raw/T1D-UOM"
    parser = Parser()

    try:
        df = parser(input_path)

        if df.empty:
            print("No data was processed.")
            return

        # Save processed data
        output_file = os.path.join(input_path, "T1D-UOM.csv")
        df.to_csv(output_file, index=True)

        print("OUTLIER", df[df['bolus'] > 30])

        print(f"✓ Saved T1D-UOM dataframe to: {output_file}")
        print(f"Dataset shape: {df.shape}")

    except Exception as e:
        print(f"Error processing T1D-UOM data: {e}")
        return None

    return df


if __name__ == "__main__":
    df = main()

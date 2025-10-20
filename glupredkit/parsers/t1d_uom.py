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

        subject_ids = [f[-8:-4] for f in os.listdir(f'{file_path}/{CGM_FOLDER}/') if f.endswith(".csv")]
        print("all subjects", subject_ids)

        df_demographics = self.get_demographics_df(file_path)
        print("DEMOGRAPHICS", df_demographics)

        # TODO: Add demographics for each subject
        # TODO: Make for loop and concat
        # TODO: Skip subject 2303, 2320, and 2404, as they dont have basal data
        df = self.get_resampled_df_for_subject(file_path, subject_ids[0])

        # TODO: Add source file

        return df


    def get_resampled_df_for_subject(self, file_path, subject_id):
        glucose_df = self.get_glucose_df(file_path, subject_id)
        meals_df = self.get_meals_df(file_path, subject_id)
        bolus_df = self.get_bolus_df(file_path, subject_id)
        basal_df = self.get_basal_df(file_path, subject_id)

        # TODO: Merge!
        activity_df = self.get_activity_df(file_path, subject_id)
        print(activity_df)

        merged_df = self.merge_df(glucose_df, meals_df)
        merged_df = self.merge_df(merged_df, bolus_df)
        merged_df = self.merge_df(merged_df, basal_df)

        merged_df['id'] = subject_id

        merged_df['insulin_type_basal'] = merged_df['insulin_type_basal'].ffill().bfill()

        # TODO: Add insulin delivery algo and modality, knowing from the paper: "This dataset includes participants using both multiple daily injections (MDI) and insulin pumps operating in open-loop mode"

        print("MERGED!!!")
        print(merged_df)

        return merged_df

    def merge_df(self, base_df, new_df):
        merged_df = base_df.merge(new_df, left_index=True, right_index=True,
                                  how='outer')  # keeps all datetimes from both
        return merged_df

    def get_glucose_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{CGM_FOLDER}/UoMGlucose{subject_id}.csv')
        df['bg_ts'] = pd.to_datetime(df['bg_ts'], format="%d/%m/%Y %H:%M")
        df['value'] = df['value'] * 18.018
        df.rename(columns={'bg_ts': 'date', 'value': 'CGM'}, inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({'CGM': 'last'})
        return resampled_df

    def get_meals_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{MEALS_DATA_FOLDER}/UoMNutrition{subject_id}.csv')
        df['meal_ts'] = pd.to_datetime(df['meal_ts'], format="%d/%m/%Y %H:%M")
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
        df['bolus_ts'] = pd.to_datetime(df['bolus_ts'], format="%d/%m/%Y %H:%M")
        df.rename(columns={'bolus_ts': 'date', 'bolus_dose': 'bolus'}, inplace=True)
        df = df.set_index('date')
        resampled_df = df.resample('5min').agg({
            'bolus': lambda x: x.sum(min_count=1),
        })
        return resampled_df

    def get_basal_df(self, file_path, subject_id):
        df = pd.read_csv(f'{file_path}/{BASAL_FOLDER}/UoMBasal{subject_id}.csv')
        df['basal_ts'] = pd.to_datetime(df['basal_ts'], format="%d/%m/%Y %H:%M")

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
        df['activity_ts'] = pd.to_datetime(df['activity_ts'], format="%d/%m/%Y %H:%M")
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
        return df[['date', 'calories_burned', 'steps', 'workout_label', 'workout_duration']]

    def get_demographics_df(self, file_path):
        df = pd.read_csv(f'{file_path}/{DEMOGRAPHICS_FOLDER}/UoMBMI.csv')
        df['participant_id'] = df['participant_id'].str[4:]  # Replace N with the number of characters

        # Convert to Ibs and feet
        df['weight_kg'] = df['weight_kg'] * 2.20462
        df['height_m'] = df['height_m'] * 3.28084

        df.rename(columns={'weight_kg': 'weight', 'height_m': 'height'}, inplace=True)
        return df[['participant_id', 'weight', 'height']]


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

        print(f"✓ Saved T1D-UOM dataframe to: {output_file}")
        print(f"Dataset shape: {df.shape}")

    except Exception as e:
        print(f"Error processing T1D-UOM data: {e}")
        return None

    return df


if __name__ == "__main__":
    df = main()

#!/usr/bin/env python3
"""
AZT1D parser for processing diabetes dataset.
Processes CSV files from the AZT1D 2025 Dataset.
"""

from glupredkit.parsers.base_parser import BaseParser
import pandas as pd
import numpy as np
import os
import glob
import re


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.df_expanded = pd.DataFrame()
        self.df_demographics = pd.DataFrame()
        
    def __call__(self, file_path: str, *args):
        """
        Process AZT1D data from CSV files.
        
        file_path -- the file path to the AZT1D dataset CGM Records folder.
        """
        print(f"Processing AZT1D data from {file_path}")
        
        # Get all CSV files from the CGM Records folder
        csv_files = []
        for i in range(1, 26):  # Subjects 1-25
            subject_path = os.path.join(file_path, f"Subject {i}", f"Subject {i}.csv")
            if os.path.exists(subject_path):
                csv_files.append(subject_path)
        
        print(f"Found {len(csv_files)} CSV files")
        
        all_processed_data = []
        
        for file_idx, file_path_full in enumerate(csv_files):
            print(f"\rProcessing file {file_idx + 1}/{len(csv_files)}: {os.path.basename(file_path_full)}", end="", flush=True)
            
            try:
                # Extract subject ID from filename (e.g., Subject 1.csv -> 1)
                filename = os.path.basename(file_path_full)
                match = re.search(r'Subject (\d+)\.csv', filename)
                if match:
                    subject_id = int(match.group(1))
                else:
                    print(f"\nWarning: Could not extract subject ID from {filename}")
                    continue
                
                # Read the CSV file
                df_raw = pd.read_csv(file_path_full)
                
                if df_raw.empty:
                    print(f"\nWarning: Empty file {filename}")
                    continue
                
                # Process this subject's data
                df_subject = self.process_subject_file(df_raw, subject_id)
                if not df_subject.empty:
                    all_processed_data.append(df_subject)
                    
            except Exception as e:
                print(f"\nError processing {os.path.basename(file_path_full)}: {e}")
        
        print("\n\nCombining all processed data...")
        
        if all_processed_data:
            df_final = pd.concat(all_processed_data, ignore_index=True)
            df_final = df_final.sort_values(['id', 'date']).reset_index(drop=True)
            
            # Store in df_expanded
            self.df_expanded = df_final.copy()
            
            # Create demographics dataframe with one row per ID
            self.create_demographics_df()
            
            # Save demographics to CSV
            self.save_demographics()
            
            print(f"Stored {len(self.df_expanded)} records in df_expanded")
            print(f"Created demographics dataframe with {len(self.df_demographics)} unique subjects")
            
            return df_final
        else:
            print("Error: No data was processed successfully!")
            return pd.DataFrame()
    
    def round_up_to_5min(self, timestamp_series):
        """Round timestamps UP to the next 5-minute interval"""
        # Add 4 minutes 59 seconds, then floor to 5-minute intervals
        # This ensures rounding UP to the next 5-minute mark
        return (timestamp_series + pd.Timedelta(minutes=4, seconds=59)).dt.floor('5min')
    
    def process_subject_file(self, df_raw, subject_id):
        """Process a single subject's CSV file."""
        
        # Create a copy to work with
        df_subject = df_raw.copy()
        
        # Handle different CGM column names
        if 'Readings (CGM / BGM)' in df_subject.columns:
            df_subject = df_subject.rename(columns={'Readings (CGM / BGM)': 'CGM'})
        
        # Rename columns according to the specification
        column_mapping = {
            'EventDateTime': 'date',
            'CGM': 'CGM',
            'Basal': 'basal',
            'TotalBolusInsulinDelivered': 'bolus',
            'CarbSize': 'carbs'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df_subject.columns:
                df_subject = df_subject.rename(columns={old_name: new_name})
        
        # Convert date column to datetime
        df_subject['date'] = pd.to_datetime(df_subject['date'])
        
        # Round dates UP to next 5-minute interval (same as CTR3)
        df_subject['date'] = self.round_up_to_5min(df_subject['date'])
        
        # Add subject ID
        df_subject['id'] = subject_id
        
        # Convert numeric columns to proper data types
        numeric_columns = ['CGM', 'basal', 'bolus', 'carbs']
        for col in numeric_columns:
            if col in df_subject.columns:
                df_subject[col] = pd.to_numeric(df_subject[col], errors='coerce')
        
        # Replace 0.0 with NaN for relevant columns (no data vs actual zero)
        zero_to_nan_columns = ['bolus', 'carbs', 'basal']
        for col in zero_to_nan_columns:
            if col in df_subject.columns:
                df_subject[col] = df_subject[col].replace(0.0, np.nan)
        
        # Aggregate data within 5-minute windows (same logic as CTR3)
        if not df_subject.empty:
            agg_dict = {}
            
            # CGM: take last value in each window
            if 'CGM' in df_subject.columns:
                agg_dict['CGM'] = 'last'
            
            # Insulin and carbs: sum within each window  
            for col in ['basal', 'bolus', 'carbs']:
                if col in df_subject.columns:
                    agg_dict[col] = 'sum'
            
            # Context and device mode: combine unique values
            if 'DeviceMode' in df_subject.columns:
                agg_dict['DeviceMode'] = lambda x: ', '.join(x.dropna().unique()) if not x.dropna().empty else np.nan
            
            if agg_dict:
                df_subject = df_subject.groupby(['id', 'date']).agg(agg_dict).reset_index()
                
                # Replace 0.0 with NaN again after aggregation
                for col in zero_to_nan_columns:
                    if col in df_subject.columns:
                        df_subject[col] = df_subject[col].replace(0.0, np.nan)
        
        # Add missing columns with values from demographics where available
        demographics = get_subject_demographics()
        if subject_id in demographics:
            demo = demographics[subject_id]
            df_subject['age'] = demo['age']
        else:
            df_subject['age'] = np.nan
        
        df_subject['calories_burned'] = np.nan
        df_subject['heartrate'] = np.nan
        df_subject['steps'] = np.nan
        df_subject['weight'] = np.nan
        df_subject['height'] = np.nan
        df_subject['insulin_delivery_modality'] = 'AID'  # All subjects use Automated Insulin Delivery
        df_subject['insulin_delivery_device'] = 't:slim X2'  # From manuscript: Tandem t:slim X2 insulin pump
        df_subject['cgm_device'] = 'Dexcom G6'  # From manuscript: CGM devices (Dexcom G6 Pro)
        df_subject['source_file'] = 'AZT1D'
        
        # Calculate total insulin (basal + bolus)
        df_subject['insulin'] = df_subject['bolus'].fillna(0) + df_subject['basal'].fillna(0)
        df_subject['insulin'] = df_subject['insulin'].replace(0.0, np.nan)
        
        # Set insulin delivery algorithm
        df_subject['insulin_delivery_algorithm'] = 'Control-IQ'  # From manuscript: Control IQ system
        
        # Handle context_description_cache column based on device mode (sleep/exercise)
        if 'DeviceMode' in df_subject.columns:
            # Map device mode to context_description_cache with Control-IQ prefix for sleep and exercise
            def map_device_mode(mode):
                if mode == 'sleep':
                    return 'Control-IQ Sleep Activity'
                elif mode == 'exercise':
                    return 'Control-IQ Exercise Activity'
                else:
                    return np.nan
            
            df_subject['context_description_cache'] = df_subject['DeviceMode'].apply(map_device_mode)
        else:
            df_subject['context_description_cache'] = np.nan
        
        # Keep tag column as NaN (not used for device modes)
        df_subject['tag'] = np.nan
        
        # Add empty columns to match HUPA-UCM structure (context_description_cache already populated above)
        empty_columns = ['is_test', 'absorption_time', 'acceleration',
                         'air_temp', 'galvanic_skin_response', 'insulin_type_basal',
                         'insulin_type_bolus', 'is_pregnant', 'meal_label', 'scheduled_basal',
                         'skin_temp', 'workout_duration', 'workout_intensity', 'workout_label']

        for col in empty_columns:
            df_subject[col] = np.nan

        # Select and order the columns to match HUPA-UCM structure exactly
        final_columns = ['date', 'id', 'CGM', 'calories_burned', 'heartrate', 'steps', 'basal', 'bolus', 'carbs',
                        'age', 'weight', 'height', 'insulin_delivery_modality', 'insulin',
                         'insulin_delivery_device', 'cgm_device', 'source_file']
        
        # Add the populated special columns
        special_columns = ['is_test', 'context_description_cache', 'tag', 'absorption_time', 'acceleration',
                          'air_temp', 'galvanic_skin_response', 'insulin_delivery_algorithm', 'insulin_type_basal',
                          'insulin_type_bolus', 'is_pregnant', 'meal_label', 'scheduled_basal',
                          'skin_temp', 'workout_duration', 'workout_intensity', 'workout_label']
        
        df_subject = df_subject[final_columns + special_columns]
        
        return df_subject
    
    def create_demographics_df(self):
        """Create demographics dataframe with one row per unique ID."""
        
        # Get unique subject IDs from processed data
        unique_ids = self.df_expanded['id'].unique() if not self.df_expanded.empty else []
        
        # Get demographic data from Table 1 in manuscript
        demographics = get_subject_demographics()
        
        demographics_data = []
        for subject_id in unique_ids:
            if subject_id in demographics:
                demo = demographics[subject_id]
                demographics_data.append({
                    'id': subject_id,
                    'gender': demo['gender'],
                    'age_of_diagnosis': np.nan,  # Not provided in dataset
                    'TDD': np.nan,  # Not provided in dataset
                    'ethnicity': np.nan,  # Not provided in dataset - don't assume
                    'source_file': 'AZT1D'
                })
            else:
                demographics_data.append({
                    'id': subject_id,
                    'gender': np.nan,
                    'age_of_diagnosis': np.nan,
                    'TDD': np.nan,
                    'ethnicity': np.nan,
                    'source_file': 'AZT1D'
                })
        
        self.df_demographics = pd.DataFrame(demographics_data)
        self.df_demographics = self.df_demographics.sort_values('id').reset_index(drop=True)
    
    def save_demographics(self, output_path: str = None):
        """Save the demographics dataframe to CSV."""
        if output_path is None:
            output_path = "AZT1D_demographics.csv"
        
        if not self.df_demographics.empty:
            self.df_demographics.to_csv(output_path, index=False)
            print(f"Demographics data saved to {output_path}")
        else:
            print("No demographics data to save. Process data first.")


def get_dataset_info(file_path):
    """Get information about the AZT1D dataset."""
    csv_files = []
    for i in range(1, 26):
        subject_path = os.path.join(file_path, f"Subject {i}", f"Subject {i}.csv")
        if os.path.exists(subject_path):
            csv_files.append(subject_path)
    
    info = {
        'total_files': len(csv_files),
        'subjects': [],
        'file_sizes': {},
        'date_ranges': {}
    }
    
    for file_path_full in csv_files:
        filename = os.path.basename(file_path_full)
        
        # Extract subject ID
        match = re.search(r'Subject (\d+)\.csv', filename)
        if match:
            subject_id = int(match.group(1))
            info['subjects'].append(subject_id)
            
            try:
                df = pd.read_csv(file_path_full)
                info['file_sizes'][subject_id] = len(df)
                
                if 'EventDateTime' in df.columns:
                    dates = pd.to_datetime(df['EventDateTime'], errors='coerce').dropna()
                    if not dates.empty:
                        info['date_ranges'][subject_id] = {
                            'start': dates.min(),
                            'end': dates.max(),
                            'duration_days': (dates.max() - dates.min()).days
                        }
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    info['subjects'].sort()
    info['unique_subjects'] = len(set(info['subjects']))
    
    return info


def get_subject_demographics():
    """Get demographic information for AZT1D subjects from Table 1 of the manuscript."""
    demographics = {
        1: {'gender': 'Male', 'age': 65},
        2: {'gender': 'Female', 'age': 67},
        3: {'gender': 'Male', 'age': 65},
        4: {'gender': 'Female', 'age': 69},
        5: {'gender': 'Male', 'age': 80},
        6: {'gender': 'Female', 'age': 77},
        7: {'gender': 'Male', 'age': 36},
        8: {'gender': 'Female', 'age': 66},
        9: {'gender': 'Female', 'age': 54},
        10: {'gender': 'Female', 'age': 71},
        11: {'gender': 'Male', 'age': 59},
        12: {'gender': 'Male', 'age': 43},
        13: {'gender': 'Male', 'age': 80},
        14: {'gender': 'Female', 'age': 32},
        15: {'gender': 'Female', 'age': 52},
        16: {'gender': 'Male', 'age': 40},
        17: {'gender': 'Female', 'age': 66},
        18: {'gender': 'Male', 'age': 65},
        19: {'gender': 'Male', 'age': 27},
        20: {'gender': 'Female', 'age': 61},
        21: {'gender': 'Female', 'age': 46},
        22: {'gender': 'Female', 'age': 46},
        23: {'gender': 'Female', 'age': 67},
        24: {'gender': 'Male', 'age': 74},
        25: {'gender': 'Male', 'age': 72}
    }
    
    return demographics
#!/usr/bin/env python3
"""
HUPA-UCM parser for processing diabetes dataset.
Processes preprocessed CSV files from the HUPA-UCM Diabetes Dataset.
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
        Process HUPA-UCM data from preprocessed CSV files.
        
        file_path -- the file path to the HUPA-UCM dataset Preprocessed folder.
        """
        print(f"Processing HUPA-UCM data from {file_path}")
        
        # Get all CSV files from the Preprocessed folder
        csv_files = glob.glob(os.path.join(file_path, "*.csv"))
        csv_files = [f for f in csv_files if f.endswith('.csv')]
        
        print(f"Found {len(csv_files)} CSV files")
        
        all_processed_data = []
        
        for file_idx, file_path_full in enumerate(csv_files):
            print(f"\rProcessing file {file_idx + 1}/{len(csv_files)}: {os.path.basename(file_path_full)}", end="", flush=True)
            
            try:
                # Extract subject ID from filename (e.g., HUPA0001P.csv -> 1)
                filename = os.path.basename(file_path_full)
                match = re.search(r'HUPA(\d+)P\.csv', filename)
                if match:
                    subject_id = int(match.group(1))  # Convert to integer to remove leading zeros
                else:
                    print(f"\nWarning: Could not extract subject ID from {filename}")
                    continue
                
                # Read the CSV file
                df_raw = pd.read_csv(file_path_full, sep=';')
                
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
    
    def process_subject_file(self, df_raw, subject_id):
        """Process a single subject's CSV file."""
        
        # Create a copy to work with
        df_subject = df_raw.copy()
        
        # Rename columns according to the specification
        column_mapping = {
            'time': 'date',
            'glucose': 'CGM',
            'calories': 'calories_burned',
            'heart_rate': 'heartrate',
            'steps': 'steps',
            'basal_rate': 'basal',
            'bolus_volume_delivered': 'bolus',
            'carb_input': 'carbs'
        }
        
        # Rename columns
        df_subject = df_subject.rename(columns=column_mapping)
        
        # Convert date column to datetime
        df_subject['date'] = pd.to_datetime(df_subject['date'])
        
        # Add subject ID
        df_subject['id'] = subject_id
        
        # Convert numeric columns to proper data types
        numeric_columns = ['CGM', 'calories_burned', 'heartrate', 'steps', 'basal', 'bolus', 'carbs']
        for col in numeric_columns:
            if col in df_subject.columns:
                df_subject[col] = pd.to_numeric(df_subject[col], errors='coerce')
        
        # Replace 0.0 with NaN for relevant columns (no data vs actual zero)
        zero_to_nan_columns = ['bolus', 'carbs']
        for col in zero_to_nan_columns:
            if col in df_subject.columns:
                df_subject[col] = df_subject[col].replace(0.0, np.nan)
        
        # Multiply carb values by 10
        if 'carbs' in df_subject.columns:
            df_subject['carbs'] = df_subject['carbs'] * 10
        
        # Add demographic information (excluding gender and age_of_diagnosis for main df)
        demographics = get_subject_demographics()
        if subject_id in demographics:
            demo = demographics[subject_id]
            df_subject['age'] = demo['age']
            df_subject['weight'] = demo['weight']  # Already converted to lbs
            df_subject['height'] = demo['height']  # Already converted to feet
            df_subject['insulin_delivery_modality'] = demo['insulin_delivery']
            df_subject['insulin_delivery_device'] = demo['insulin_delivery_device']
        else:
            # Default values if demographic data is missing
            df_subject['age'] = np.nan
            df_subject['weight'] = np.nan
            df_subject['height'] = np.nan
            df_subject['insulin_delivery_modality'] = np.nan
            df_subject['insulin_delivery_device'] = np.nan

        df_subject['source_file'] = 'HUPA-UCM'
        df_subject['cgm_device'] = 'FreeStyle Libre 2'
        df_subject['insulin'] = df_subject['bolus'].fillna(0) + df_subject['basal']

        empty_columns = ['is_test', 'context_description_cache', 'tag', 'absorption_time', 'acceleration',
                         'air_temp', 'galvanic_skin_response', 'insulin_delivery_algorithm', 'insulin_type_basal',
                         'insulin_type_bolus', 'is_pregnant', 'meal_label', 'scheduled_basal',
                         'skin_temp', 'workout_duration', 'workout_intensity', 'workout_label']

        for col in empty_columns:
            df_subject[col] = np.nan

        # Select and order the columns we want (excluding gender and age_of_diagnosis)
        final_columns = ['date', 'id', 'CGM', 'calories_burned', 'heartrate', 'steps', 'basal', 'bolus', 'carbs',
                         'age', 'weight', 'height', 'insulin_delivery_modality', 'insulin',
                         'insulin_delivery_device', 'cgm_device', 'source_file']
        df_subject = df_subject[final_columns + empty_columns]
        
        return df_subject
    
    def create_demographics_df(self):
        """Create demographics dataframe with one row per unique ID."""
        demographics = get_subject_demographics()
        
        # Get unique subject IDs from processed data
        unique_ids = self.df_expanded['id'].unique() if not self.df_expanded.empty else []
        
        demographics_data = []
        for subject_id in unique_ids:
            if subject_id in demographics:
                demo = demographics[subject_id]
                demographics_data.append({
                    'id': subject_id,
                    'gender': demo['gender'],
                    'age_of_diagnosis': demo['age'] - demo['dx_time'],
                    'TDD': np.nan,
                    'ethnicity': 'White',
                    'source_file': 'HUPA-UCM'
                })
            else:
                demographics_data.append({
                    'id': subject_id,
                    'gender': np.nan,
                    'age_of_diagnosis': np.nan,
                    'TDD': np.nan,
                    'ethnicity': 'White',
                    'source_file': 'HUPA-UCM'
                })
        
        self.df_demographics = pd.DataFrame(demographics_data)
        self.df_demographics = self.df_demographics.sort_values('id').reset_index(drop=True)
    
    def save_demographics(self, output_path: str = None):
        """Save the demographics dataframe to CSV."""
        if output_path is None:
            output_path = "HUPA_UCM_demographics.csv"
        
        if not self.df_demographics.empty:
            self.df_demographics.to_csv(output_path, index=False)
            print(f"Demographics data saved to {output_path}")
        else:
            print("No demographics data to save. Process data first.")


def get_subject_demographics():
    """Get demographic information for HUPA-UCM subjects from Table 1 of the paper."""
    demographics = {
        1: {'gender': 'Female', 'age': 56.3, 'weight_kg': 59.0, 'height_cm': 161, 'dx_time': 15.5, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        2: {'gender': 'Male', 'age': 48.6, 'weight_kg': 82.4, 'height_cm': 186, 'dx_time': 36.5, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Roche Pump'},
        3: {'gender': 'Male', 'age': 43.4, 'weight_kg': 62.0, 'height_cm': 182, 'dx_time': 12.5, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        4: {'gender': 'Male', 'age': 41.2, 'weight_kg': 88.0, 'height_cm': 180, 'dx_time': 8.5, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Roche Pump'},
        5: {'gender': 'Female', 'age': 41.9, 'weight_kg': 58.5, 'height_cm': 161, 'dx_time': 39.5, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        6: {'gender': 'Male', 'age': 22.1, 'weight_kg': 71.0, 'height_cm': 170, 'dx_time': 13.5, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        7: {'gender': 'Male', 'age': 37.6, 'weight_kg': 102.6, 'height_cm': 183, 'dx_time': 10.1, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        9: {'gender': 'Female', 'age': 41.2, 'weight_kg': 64.0, 'height_cm': 165, 'dx_time': 30.7, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        10: {'gender': 'Female', 'age': 41.9, 'weight_kg': 51.0, 'height_cm': 164, 'dx_time': 15.2, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        11: {'gender': 'Female', 'age': 35.0, 'weight_kg': 56.0, 'height_cm': 153, 'dx_time': 27.3, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Roche Pump'},
        14: {'gender': 'Female', 'age': 50.0, 'weight_kg': 61.0, 'height_cm': 155, 'dx_time': 12.9, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        15: {'gender': 'Female', 'age': 43.1, 'weight_kg': 58.6, 'height_cm': 162, 'dx_time': 11.2, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        16: {'gender': 'Female', 'age': 29.9, 'weight_kg': 64.9, 'height_cm': 157, 'dx_time': 20.1, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Medtronic Pump'},
        17: {'gender': 'Female', 'age': 26.3, 'weight_kg': 61.8, 'height_cm': 167, 'dx_time': 24.2, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        18: {'gender': 'Female', 'age': 32.3, 'weight_kg': 57.2, 'height_cm': 167, 'dx_time': 25.6, 'insulin_delivery': 'SAP', 'insulin_delivery_device': np.nan},
        19: {'gender': 'Male', 'age': 18.0, 'weight_kg': 69.7, 'height_cm': 168, 'dx_time': 7.6, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        20: {'gender': 'Male', 'age': 45.7, 'weight_kg': 71.6, 'height_cm': 168, 'dx_time': 13.5, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        21: {'gender': 'Female', 'age': 48.6, 'weight_kg': 57.0, 'height_cm': 153, 'dx_time': 2.2, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        22: {'gender': 'Male', 'age': 59.6, 'weight_kg': 77.6, 'height_cm': 179, 'dx_time': 14.6, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Roche Pump'},
        23: {'gender': 'Male', 'age': 22.9, 'weight_kg': 55.5, 'height_cm': 173, 'dx_time': 0.8, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        24: {'gender': 'Male', 'age': 47.9, 'weight_kg': 80.5, 'height_cm': 174, 'dx_time': 35.9, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        25: {'gender': 'Male', 'age': 38.1, 'weight_kg': 104.8, 'height_cm': 188, 'dx_time': 20.3, 'insulin_delivery': 'SAP', 'insulin_delivery_device': 'Roche Pump'},
        26: {'gender': 'Female', 'age': 61.8, 'weight_kg': 80.0, 'height_cm': 165, 'dx_time': 21.5, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        27: {'gender': 'Male', 'age': 26.4, 'weight_kg': 76.0, 'height_cm': 185, 'dx_time': 23.7, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'},
        28: {'gender': 'Female', 'age': 21.2, 'weight_kg': 56.0, 'height_cm': 160, 'dx_time': 2.0, 'insulin_delivery': 'MDI', 'insulin_delivery_device': 'Insulin Pen'}
    }
    
    # Convert units: kg to lbs (1 kg = 2.20462 lbs), cm to feet (1 cm = 0.0328084 feet)
    for subject_id in demographics:
        demo = demographics[subject_id]
        demo['weight'] = demo['weight_kg'] * 2.20462  # Convert to lbs
        demo['height'] = demo['height_cm'] * 0.0328084  # Convert to feet
        # Keep original units as well for reference
    
    return demographics


def get_dataset_info(file_path):
    """Get information about the HUPA-UCM dataset."""
    csv_files = glob.glob(os.path.join(file_path, "*.csv"))
    
    info = {
        'total_files': len(csv_files),
        'subjects': [],
        'file_sizes': {},
        'date_ranges': {}
    }
    
    for file_path_full in csv_files:
        filename = os.path.basename(file_path_full)
        
        # Extract subject ID
        match = re.search(r'HUPA(\d+)P\.csv', filename)
        if match:
            subject_id = int(match.group(1))
            info['subjects'].append(subject_id)
            
            try:
                df = pd.read_csv(file_path_full, sep=';')
                info['file_sizes'][subject_id] = len(df)
                
                if 'time' in df.columns:
                    dates = pd.to_datetime(df['time'], errors='coerce').dropna()
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
#!/usr/bin/env python3
"""
AZT1D parser for processing diabetes dataset.
Processes CSV files from the AZT1D 2025 Dataset.
"""

from glupredkit.parsers.base_parser import BaseParser
import pandas as pd
import numpy as np
import os
import re


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame()

    def __call__(self, file_path: str, *args):
        """
        Process AZT1D data from CSV files.
        
        file_path -- the file path to the AZT1D dataset CGM Records folder.
        """
        print(f"Processing AZT1D data from {file_path}")
        
        # Get all CSV files from the CGM Records folder
        csv_files = []
        for i in range(1, 26):  # Subjects 1-25
            if not i in [10, 23]:  # Skipping subject 10 and 23 due to weird basal values ranging from 0 to above 1000
                subject_path = os.path.join(file_path, f"Subject {i}", f"Subject {i}.csv")
                if os.path.exists(subject_path):
                    csv_files.append(subject_path)
        
        print(f"Found {len(csv_files)} CSV files")
        
        all_processed_data = []
        
        for file_idx, file_path_full in enumerate(csv_files):
            print(f"\rProcessing file {file_idx + 1}/{len(csv_files)}: {os.path.basename(file_path_full)} ", end="", flush=True)
            
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
            df_final = pd.concat(all_processed_data)

            # Sort by 'id' column first, then by date
            df_final = df_final.sort_values(by=['id', 'date'])

            # Store in df
            self.df = df_final.copy()

            print(f"Stored {len(self.df)} records in df")

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

        # Correcting for weird basal rates, that systematically have the comma replaced three places
        df_subject['Basal'] = pd.to_numeric(df_subject['Basal'], errors='coerce')

        basal_threshold = 100
        df_subject.loc[df_subject['Basal'] > basal_threshold, 'Basal'] /= 1000
        if not df_raw.loc[df_raw['Basal'] > basal_threshold, 'Basal'].empty:
            print(f"Divided {df_raw.loc[df_raw['Basal'] > basal_threshold, 'Basal'].shape[0]} basal values by 1000")

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
        
        # Round dates UP to next 5-minute interval
        df_subject['date'] = self.round_up_to_5min(df_subject['date'])
        df_subject = df_subject.sort_values(by='date')

        # Convert numeric columns to proper data types
        numeric_columns = ['CGM', 'basal', 'bolus', 'carbs']
        for col in numeric_columns:
            if col in df_subject.columns:
                df_subject[col] = pd.to_numeric(df_subject[col], errors='coerce')
            else:
                df_subject[col] = np.nan

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

        df_subject = df_subject[['date', 'CGM', 'basal', 'bolus', 'carbs', 'context_description_cache']]
        df_subject = df_subject.drop_duplicates(subset=['date', 'CGM', 'carbs', 'bolus', 'basal']).sort_values('date')
        df_subject = df_subject.set_index('date')

        # Resample into 5-minute intervals
        agg_dict = {
            'CGM': 'mean',
            # There seems to be duplicate rows if several rows within a single 5-min interval, so we take mean value and not sum in this dataset
            'basal': 'mean',
            'bolus': 'mean',
            'carbs': 'mean',
            'context_description_cache': 'last',  # take last value in interval
        }

        # Resample to 5-minute intervals (data was already resampled, but with some missing / duplicate values)
        df_subject = df_subject.resample('5min').agg(agg_dict)

        # Check for 5-minute intervals
        valid_intervals = (df_subject.index.to_series().diff().dropna() == pd.Timedelta("5min")).all()
        if valid_intervals:
            print(f"Subject {subject_id} has perfect 5-minute intervals")
        else:
            print(f"Warning: Subject {subject_id} does not have valid 5-minute intervals!")

        # Add subject ID
        df_subject['id'] = subject_id

        # Add missing columns with values from demographics where available
        demographics = get_subject_demographics()
        if subject_id in demographics:
            demo = demographics[subject_id]
            df_subject['age'] = demo['age']
            df_subject['gender'] = demo['gender']
        else:
            df_subject['age'] = np.nan
            df_subject['gender'] = np.nan

        df_subject['insulin_delivery_modality'] = 'AID'  # All subjects use Automated Insulin Delivery
        df_subject['insulin_delivery_device'] = 't:slim X2'  # From manuscript: Tandem t:slim X2 insulin pump
        df_subject['cgm_device'] = 'Dexcom G6'  # From manuscript: CGM devices (Dexcom G6 Pro)
        df_subject['source_file'] = 'AZT1D'

        # Calculate total insulin (basal + bolus)
        df_subject['basal'] = df_subject['basal'] / 12  # From U/hr to U
        df_subject['insulin'] = df_subject['bolus'].fillna(0) + df_subject['basal']

        # Set insulin delivery algorithm
        df_subject['insulin_delivery_algorithm'] = 'Control-IQ'  # From manuscript: Control IQ system
        df_subject.reset_index(drop=False, inplace=True)

        # Select and order the columns
        final_columns = ['date', 'id', 'CGM', 'basal', 'bolus', 'carbs', 'age', 'insulin_delivery_modality', 'insulin',
                         'gender', 'insulin_delivery_device', 'cgm_device', 'source_file', 'context_description_cache',
                         'insulin_delivery_algorithm']

        df_subject = df_subject[final_columns]
        return df_subject


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


def main():
    """Main function to run the AZT1D parser with a hard-coded file path."""
    # Hard-coded file path - update this to your actual AZT1D dataset path
    file_path = "data/raw/AZT1D 2025/CGM Records"
    
    # Create parser instance
    parser = Parser()
    
    # Process the data
    result = parser(file_path)
    result.to_csv('data/raw/AZT1D.csv', index=False)

    print(result)

    if not result.empty:
        print(f"\nProcessing completed successfully!")
        print(f"Total records processed: {len(result)}")
        print(f"Unique subjects: {result['id'].nunique()}")
    else:
        print("No data was processed. Please check the file path and data format.")


if __name__ == "__main__":
    main()

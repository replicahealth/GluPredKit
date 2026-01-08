#!/usr/bin/env python3
"""
CTR3 parser for processing diabetes dataset.
Processes tab-separated files from the CTR3_Public_Dataset.

Understanding the insulin delivery data here is not so intuitive, but:
- MonitorTotalBolus: This file is a consolidation of all insulin delivered (bolus+basal), in units
- MonitorBasalBolus: This is all the basal insulin delivered, but is called "bolus" because it shows basal insulin in Units and not in U/hr
- MonitorMealBolus and MonitorCorrectionBolus: All actual boluses, either for a meal or a correction bolus
"""

from glupredkit.parsers.base_parser import BaseParser
import pandas as pd
import numpy as np
import os
import glob


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame()

    def __call__(self, file_path: str, *args):
        """
        Process CTR3 data from tab-separated files.
        
        file_path -- the file path to the CTR3_Public_Dataset Data Tables folder.
        """
        self.file_path = file_path  # Store for enrollment data loading
        print(f"Processing CTR3 data from {file_path}")
        
        # Read all relevant data files
        cgm_data = self.read_cgm_data(file_path)
        insulin_data = self.read_insulin_data(file_path)
        meal_data = self.read_meal_data(file_path)

        print(f"Loaded CGM: {len(cgm_data)} records")
        print(f"Loaded Insulin: {len(insulin_data)} records")
        print(f"Loaded Meal: {len(meal_data)} records")
        
        # Merge all data on DeidentID and timestamp
        df_merged = self.merge_all_data(cgm_data, insulin_data, meal_data)
        
        if df_merged.empty:
            print("Error: No data was merged successfully!")
            return pd.DataFrame()
        
        print(f"Merged data: {len(df_merged)} records")
        
        # Process and format the merged data
        df_final = self.process_merged_data(df_merged)

        # Remove outlier insulin data and following 8 hours of data
        df_final = self.remove_outliers_and_following(df_final, ['insulin'], extend_lim=8*12)

        # Store in df
        self.df = df_final.copy()

        print(f"Stored {len(self.df)} records in df_expanded")
        return df_final
    
    def read_cgm_data(self, file_path):
        """Read CGM data from CGM.txt"""
        cgm_file = os.path.join(file_path, "CGM.txt")
        if not os.path.exists(cgm_file):
            print(f"Warning: CGM file not found at {cgm_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(cgm_file, sep='|')
        df['timestamp'] = pd.to_datetime(df['InternalTime'])
        return df[['DeidentID', 'timestamp', 'CGM']].rename(columns={'DeidentID': 'id'})

    def read_insulin_data(self, file_path):
        """Read basal insulin data from MonitorTotalBolus.txt"""
        basal_file = os.path.join(file_path, "MonitorTotalBolus.txt")
        if not os.path.exists(basal_file):
            print(f"Warning: Basal file not found at {basal_file}")
            return pd.DataFrame()

        df = pd.read_csv(basal_file, sep='|')
        df['timestamp'] = pd.to_datetime(df['LocalDeliveredDtTm'])
        return df[['DeidentID', 'timestamp', 'DeliveredValue']].rename(columns={
            'DeidentID': 'id',
            'DeliveredValue': 'insulin'
        })

    def read_meal_data(self, file_path):
        """Read meal/carb data from MonitorMeal.txt"""
        meal_file = os.path.join(file_path, "MonitorMeal.txt")
        if not os.path.exists(meal_file):
            print(f"Warning: Meal file not found at {meal_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(meal_file, sep='|')
        df['timestamp'] = pd.to_datetime(df['LocalDtTm'])
        return df[['DeidentID', 'timestamp', 'MealSize']].rename(columns={
            'DeidentID': 'id',
            'MealSize': 'carbs'
        })
    
    def round_up_to_5min(self, timestamp_series):
        """Round timestamps UP to the next 5-minute interval"""
        # Add 4 minutes 59 seconds, then floor to 5-minute intervals
        # This ensures rounding UP to the next 5-minute mark
        return (timestamp_series + pd.Timedelta(minutes=4, seconds=59)).dt.floor('5min')
    
    def merge_all_data(self, cgm_data, insulin_data, meal_data):
        """Merge all data sources on id and timestamp with forward aggregation"""
        all_data = []
        
        # Process CGM data - round UP to next 5-minute interval
        if not cgm_data.empty:
            cgm_clean = cgm_data.dropna(subset=['id', 'timestamp']).copy()
            cgm_clean['timestamp_rounded'] = self.round_up_to_5min(cgm_clean['timestamp'])
            
            # Take the last CGM value in each 5-minute window per subject
            cgm_grouped = cgm_clean.groupby(['id', 'timestamp_rounded']).agg({
                'CGM': 'last'  # Take the most recent CGM value in the interval
            }).reset_index()
            
            all_data.append(cgm_grouped.rename(columns={'timestamp_rounded': 'timestamp'}))
        
        # Process basal data - sum over previous 5 minutes
        if not insulin_data.empty:
            insulin_clean = insulin_data.dropna(subset=['id', 'timestamp']).copy()
            insulin_clean['timestamp_rounded'] = self.round_up_to_5min(insulin_clean['timestamp'])
            
            # Sum basal insulin within each 5-minute window
            insulin_grouped = insulin_clean.groupby(['id', 'timestamp_rounded']).agg({
                'insulin': 'sum'
            }).reset_index()
            
            all_data.append(insulin_grouped.rename(columns={'timestamp_rounded': 'timestamp'}))

        # Process meal data - sum carbs over previous 5 minutes
        if not meal_data.empty:
            meal_clean = meal_data.dropna(subset=['id', 'timestamp']).copy()
            meal_clean['timestamp_rounded'] = self.round_up_to_5min(meal_clean['timestamp'])
            
            # Sum carbohydrates within each 5-minute window
            meal_grouped = meal_clean.groupby(['id', 'timestamp_rounded']).agg({
                'carbs': 'sum'
            }).reset_index()
            
            all_data.append(meal_grouped.rename(columns={'timestamp_rounded': 'timestamp'}))
        
        # Merge all data sources using full outer join
        if not all_data:
            print("No data to merge")
            return pd.DataFrame()
        
        df_merged = all_data[0]
        for df in all_data[1:]:
            df_merged = pd.merge(
                df_merged, df,
                on=['id', 'timestamp'],
                how='outer'
            )
        
        # Create a complete 5-minute time grid for each subject to fill gaps
        if not df_merged.empty:
            complete_grids = []
            for subject_id in df_merged['id'].unique():
                subject_data = df_merged[df_merged['id'] == subject_id]
                if not subject_data.empty:
                    min_time = subject_data['timestamp'].min()
                    max_time = subject_data['timestamp'].max()
                    
                    # Create complete 5-minute time grid for this subject
                    time_grid = pd.date_range(start=min_time, end=max_time, freq='5min')
                    subject_grid = pd.DataFrame({
                        'id': subject_id,
                        'timestamp': time_grid
                    })
                    complete_grids.append(subject_grid)
            
            if complete_grids:
                # Combine all subject grids
                complete_grid = pd.concat(complete_grids, ignore_index=True)
                
                # Merge with existing data to fill gaps with NaN
                df_merged = pd.merge(
                    complete_grid, df_merged,
                    on=['id', 'timestamp'],
                    how='left'
                )
        
        return df_merged.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    def load_enrollment_data(self):
        """Load enrollment data from Enrollment.txt file."""
        enrollment_file = os.path.join(self.file_path if hasattr(self, 'file_path') else '', "Enrollment.txt")
        
        # Try multiple possible paths
        possible_paths = [
            enrollment_file,
            "/Users/miriamk.wolff/Documents/Repositories/PhD/GluPredKit/data/raw/CTR3_Public_Dataset/Data Tables/Enrollment.txt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path, sep='|')
                
        print("Warning: Enrollment.txt file not found")
        return pd.DataFrame()
    
    def process_merged_data(self, df_merged):
        """Process and format merged data"""
        df_processed = df_merged.copy()
        
        # Rename timestamp to date
        df_processed = df_processed.rename(columns={'timestamp': 'date'})
        
        # Convert numeric columns to proper data types
        numeric_columns = ['CGM', 'insulin', 'carbs']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Replace 0.0 with NaN for relevant columns (no data vs actual zero)
        zero_to_nan_columns = ['carbs']
        for col in zero_to_nan_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].replace(0.0, np.nan)

        # Add weight, height, age, and insulin types from enrollment data (block sparse per subject)
        enrollment_data = self.load_enrollment_data()
        df_processed['age'] = np.nan
        df_processed['weight'] = np.nan
        df_processed['height'] = np.nan
        df_processed['insulin_type_basal'] = pd.Series(dtype='object')
        df_processed['insulin_type_bolus'] = pd.Series(dtype='object')
        
        if not enrollment_data.empty:
            for subject_id in df_processed['id'].unique():
                enrollment_row = enrollment_data[enrollment_data['DeidentID'] == subject_id]
                if not enrollment_row.empty:
                    row = enrollment_row.iloc[0]
                    
                    # Convert weight from kg to lbs (1 kg = 2.20462 lbs)
                    weight_kg = pd.to_numeric(row.get('Weight'), errors='coerce')
                    weight_lbs = weight_kg * 2.20462 if pd.notna(weight_kg) else np.nan
                    
                    # Convert height from cm to feet (1 cm = 0.0328084 feet)
                    height_cm = pd.to_numeric(row.get('Height'), errors='coerce')
                    height_feet = height_cm * 0.0328084 if pd.notna(height_cm) else np.nan
                    
                    # Age at enrollment
                    age_at_enrollment = pd.to_numeric(row.get('Age at Enrollment'), errors='coerce')

                    gender = 'Male' if row.get('Gender') == 'M' else ('Female' if row.get('Gender') == 'F' else np.nan)
                    age_of_diagnosis = pd.to_numeric(row.get('Age at Diagnosis'), errors='coerce')
                    ethnicity = row.get('Race')

                    print(f"Subject {subject_id}: age {age_at_enrollment}, weight {weight_kg}")

                    # Determine insulin type based on enrollment data
                    insulin_type = np.nan
                    if pd.notna(row.get('Novolog')) and row.get('Novolog') == 1.0:
                        insulin_type = 'NovoLog'
                    elif pd.notna(row.get('Humalog')) and row.get('Humalog') == 1.0:
                        insulin_type = 'Humalog'
                    elif pd.notna(row.get('Regular')) and row.get('Regular') == 1.0:
                        insulin_type = 'Regular'
                    elif pd.notna(row.get('Apidra')) and row.get('Apidra') == 1.0:
                        insulin_type = 'Apidra'
                    elif pd.notna(row.get('InsTypeOther')) and row.get('InsTypeOther') == 1.0:
                        insulin_type = np.nan
                    
                    # Set weight, height, age, and insulin types for all records of this subject
                    df_processed.loc[df_processed['id'] == subject_id, 'weight'] = weight_lbs
                    df_processed.loc[df_processed['id'] == subject_id, 'height'] = height_feet
                    df_processed.loc[df_processed['id'] == subject_id, 'age'] = age_at_enrollment
                    df_processed.loc[df_processed['id'] == subject_id, 'insulin_type_basal'] = insulin_type
                    df_processed.loc[df_processed['id'] == subject_id, 'insulin_type_bolus'] = insulin_type
                    df_processed.loc[df_processed['id'] == subject_id, 'gender'] = gender
                    df_processed.loc[df_processed['id'] == subject_id, 'age_of_diagnosis'] = age_of_diagnosis
                    df_processed.loc[df_processed['id'] == subject_id, 'ethnicity'] = ethnicity

        df_processed['insulin_delivery_modality'] = 'AID'
        df_processed['insulin_delivery_device'] = 'Roche Accu-Check'
        df_processed['insulin_delivery_algorithm'] = 'JAEB DiA Control-to-Range'
        df_processed['cgm_device'] = 'Dexcom G4'
        df_processed['source_file'] = 'CTR3'

        # Select and order the columns to match HUPA-UCM structure exactly
        final_columns = ['date', 'id', 'CGM', 'insulin', 'carbs', 'age', 'weight', 'height',
                         'insulin_delivery_modality', 'insulin_delivery_device', 'cgm_device', 'source_file',
                         'insulin_type_basal', 'insulin_type_bolus', 'gender', 'age_of_diagnosis', 'ethnicity']
        df_processed = df_processed[final_columns]
        return df_processed.sort_values(['id', 'date']).reset_index(drop=True)


def get_dataset_info(file_path):
    """Get information about the CTR3 dataset."""
    cgm_file = os.path.join(file_path, "CGM.txt")
    
    info = {
        'cgm_file_exists': os.path.exists(cgm_file),
        'subjects': [],
        'date_ranges': {}
    }
    
    if info['cgm_file_exists']:
        try:
            df = pd.read_csv(cgm_file, sep='|', nrows=10000)  # Sample first 10k rows
            info['subjects'] = sorted(df['DeidentID'].unique().tolist())
            info['unique_subjects'] = len(info['subjects'])
            
            # Get date ranges for first few subjects
            for subject_id in info['subjects'][:5]:
                subject_data = df[df['DeidentID'] == subject_id]
                if not subject_data.empty:
                    dates = pd.to_datetime(subject_data['InternalTime'], errors='coerce').dropna()
                    if not dates.empty:
                        info['date_ranges'][subject_id] = {
                            'start': dates.min(),
                            'end': dates.max(),
                            'duration_days': (dates.max() - dates.min()).days
                        }
        except Exception as e:
            print(f"Error reading CGM file: {e}")
    
    return info


def main():
    """Main function to run the AZT1D parser with a hard-coded file path."""
    # Hard-coded file path - update this to your actual AZT1D dataset path
    file_path = "data/raw/CTR3_Public_Dataset/Data Tables"

    # Create parser instance
    parser = Parser()

    # Process the data
    result = parser(file_path)
    result.to_csv('data/raw/CTR3.csv', index=False)

    print("Max insulin: ", result['insulin'].max())
    print("Min insulin: ", result['insulin'].min())

    if not result.empty:
        print(f"\nProcessing completed successfully!")
        print(f"Total records processed: {len(result)}")
        print(f"Unique subjects: {result['id'].nunique()}")
        print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    else:
        print("No data was processed. Please check the file path and data format.")


if __name__ == "__main__":
    main()

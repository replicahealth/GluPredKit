#!/usr/bin/env python3
"""
Updated Shanghai T1DM parser that incorporates all the refinements:
- Proper CSII basal interval spanning
- Correct time resampling (round up to nearest 5-min)
- Proper bolus and basal insulin extraction
- Meal label filtering ("data not available" → "Meal without description")
- Correct column structure (demographics separation)
"""

from glupredkit.parsers.base_parser import BaseParser
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import os
import glob


def round_up_to_nearest_5min(dt):
    """Round timestamp UP to nearest 5-minute interval."""
    minutes = dt.minute
    remainder = minutes % 5
    if remainder == 0 and dt.second == 0:
        return dt
    else:
        minutes_to_add = 5 - remainder
        return dt.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)


def parse_insulin_entry(insulin_text):
    """Parse insulin entry to extract insulin name and dose."""
    if pd.isna(insulin_text) or insulin_text == '':
        return None, None
    
    insulin_text = str(insulin_text).strip()
    
    if ';' in insulin_text:
        return insulin_text, None
    
    dose_patterns = [
        r'^(.+?),?\s*(\d+(?:\.\d+)?)\s*iu\s*$',
        r'^(.+?)\s+(\d+(?:\.\d+)?)\s*iu\s*$',
    ]
    
    for pattern in dose_patterns:
        match = re.match(pattern, insulin_text, re.IGNORECASE)
        if match:
            insulin_name = match.group(1).strip()
            try:
                dose = float(match.group(2))
                return insulin_name, dose
            except ValueError:
                continue
    
    return insulin_text, None


def is_short_acting_insulin(insulin_name):
    """Check if insulin is short-acting (bolus)."""
    if not insulin_name:
        return False
    
    insulin_name_lower = insulin_name.lower()
    mixed_keywords = ['30r', '50r', '70/30', '75/25', 'mix']
    if any(keyword in insulin_name_lower for keyword in mixed_keywords):
        return False
    
    short_acting_keywords = [
        'novolin r', 'humulin r', 'gansulin r', 'regular insulin', 'regular',
        'insulin aspart', 'insulin lispro', 'insulin glulisine',
        'novorapid', 'humalog', 'apidra', 'fiasp'
    ]
    
    return any(keyword in insulin_name_lower for keyword in short_acting_keywords)


def is_long_acting_insulin(insulin_name):
    """Check if insulin is long-acting (basal)."""
    if not insulin_name:
        return False
    
    insulin_name_lower = insulin_name.lower()
    long_acting_keywords = [
        'degludec', 'glargine', 'detemir', 'lantus', 'levemir', 'tresiba',
        'basaglar', 'toujeo', 'nph', 'intermediate', 'basal'
    ]
    
    return any(keyword in insulin_name_lower for keyword in long_acting_keywords)


def create_csii_basal_intervals(csii_data):
    """Create CSII basal intervals that span the correct time periods."""
    if csii_data.empty:
        return []
    
    csii_col = "CSII - basal insulin (Novolin R, IU / H)"
    intervals = []
    
    csii_data = csii_data.sort_values('Date').reset_index(drop=True)
    
    for i, row in csii_data.iterrows():
        # Skip non-numeric values
        if not isinstance(row[csii_col], (int, float)):
            continue
            
        start_time = pd.to_datetime(row['Date'])
        rate = row[csii_col] / 12.0  # Convert IU/H to IU per 5-min
        
        if i < len(csii_data) - 1:
            end_time = pd.to_datetime(csii_data.iloc[i + 1]['Date'])
        else:
            end_time = start_time + timedelta(hours=24)
        
        intervals.append((start_time, end_time, rate, 'Novolin R'))
    
    return intervals


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.subject_data = {}
        self.df_expanded = pd.DataFrame()
        
    def __call__(self, file_path: str, *args):
        """Process Shanghai T1DM data with all refinements."""
        print(f"Processing Shanghai T1DM data from {file_path}")
        
        # Get all Excel files
        xlsx_files = glob.glob(os.path.join(file_path, "*.xlsx"))
        xls_files = glob.glob(os.path.join(file_path, "*.xls"))
        files = [f for f in xlsx_files + xls_files 
                if not os.path.basename(f).startswith('~') and 'Summary' not in f]
        
        print(f"Processing {len(files)} data files...")
        
        # Load demographics
        demographics = get_subject_demographics(file_path)
        print(f"Loaded demographics for {len(demographics)} subjects")
        
        all_processed_data = []
        
        for file_idx, file in enumerate(files):
            print(f"\\rProcessing file {file_idx + 1}/{len(files)}: {os.path.basename(file)}", end="", flush=True)
            
            try:
                subject_id = os.path.basename(file).split('_')[0]
                df_raw = pd.read_excel(file)
                
                if df_raw.empty:
                    continue
                
                # Process this subject's data
                df_subject = self.process_subject_file(df_raw, subject_id, demographics)
                if not df_subject.empty:
                    all_processed_data.append(df_subject)
                    
            except Exception as e:
                print(f"\\nError processing {os.path.basename(file)}: {e}")
        
        print("\\n\\nCombining all processed data...")
        
        if all_processed_data:
            df_final = pd.concat(all_processed_data, ignore_index=True)
            df_final['source_file'] = 'ShanghaiT1DM'
            
            # Store in df_expanded
            self.df_expanded = df_final.copy()
            print(f"Stored {len(self.df_expanded)} records in df_expanded")
            
            return df_final
        else:
            print("Error: No data was processed successfully!")
            return pd.DataFrame()
    
    def process_subject_file(self, df_raw, subject_id, demographics):
        """Process a single subject's raw data file."""
        
        # Create time-series structure with proper resampling
        all_timestamps = []
        
        if 'Date' in df_raw.columns:
            for timestamp in df_raw['Date'].dropna():
                rounded_time = round_up_to_nearest_5min(pd.to_datetime(timestamp))
                all_timestamps.append(rounded_time)
        
        if not all_timestamps:
            return pd.DataFrame()
        
        # Create full time range at 5-min intervals
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
        
        # Initialize the subject's time-series dataframe
        df_subject = pd.DataFrame({'date': time_range})
        df_subject['id'] = subject_id
        
        # Initialize all data columns with proper dtypes
        df_subject['CGM'] = np.nan
        df_subject['bolus'] = np.nan
        df_subject['insulin_type_bolus'] = ''
        df_subject['basal'] = np.nan
        df_subject['insulin_type_basal'] = ''
        df_subject['meal_label'] = ''
        
        # Process CGM data
        if 'CGM (mg / dl)' in df_raw.columns:
            for idx, row in df_raw.iterrows():
                if pd.notna(row.get('Date')) and pd.notna(row.get('CGM (mg / dl)')):
                    rounded_time = round_up_to_nearest_5min(pd.to_datetime(row['Date']))
                    time_idx = df_subject[df_subject['date'] == rounded_time].index
                    if len(time_idx) > 0:
                        df_subject.loc[time_idx[0], 'CGM'] = row['CGM (mg / dl)']
        
        # Process CSII bolus data
        csii_bolus_col = "CSII - bolus insulin (Novolin R, IU)"
        if csii_bolus_col in df_raw.columns:
            for idx, row in df_raw.iterrows():
                if pd.notna(row.get('Date')) and pd.notna(row.get(csii_bolus_col)):
                    # Skip non-numeric values
                    if isinstance(row[csii_bolus_col], (int, float)):
                        rounded_time = round_up_to_nearest_5min(pd.to_datetime(row['Date']))
                        time_idx = df_subject[df_subject['date'] == rounded_time].index
                        if len(time_idx) > 0:
                            df_subject.loc[time_idx[0], 'bolus'] = row[csii_bolus_col]
                            df_subject.loc[time_idx[0], 'insulin_type_bolus'] = 'Novolin R'
        
        # Process CSII basal data with proper interval spanning
        csii_basal_col = "CSII - basal insulin (Novolin R, IU / H)"
        if csii_basal_col in df_raw.columns:
            csii_data = df_raw[df_raw[csii_basal_col].notna()].copy()
            if not csii_data.empty:
                basal_intervals = create_csii_basal_intervals(csii_data)
                
                for start_time, end_time, rate, insulin_type in basal_intervals:
                    start_rounded = round_up_to_nearest_5min(start_time)
                    end_rounded = round_up_to_nearest_5min(end_time)
                    
                    mask = (df_subject['date'] >= start_rounded) & (df_subject['date'] < end_rounded)
                    df_subject.loc[mask, 'basal'] = rate
                    df_subject.loc[mask, 'insulin_type_basal'] = insulin_type
        
        # Process S.C. insulin data
        sc_col = "Insulin dose - s.c."
        if sc_col in df_raw.columns:
            for idx, row in df_raw.iterrows():
                if pd.notna(row.get('Date')) and pd.notna(row.get(sc_col)):
                    rounded_time = round_up_to_nearest_5min(pd.to_datetime(row['Date']))
                    time_idx = df_subject[df_subject['date'] == rounded_time].index
                    
                    if len(time_idx) > 0:
                        insulin_name, dose = parse_insulin_entry(row[sc_col])
                        
                        if insulin_name and dose:
                            if is_short_acting_insulin(insulin_name):
                                # Add to bolus
                                current_bolus = df_subject.loc[time_idx[0], 'bolus']
                                if pd.notna(current_bolus):
                                    df_subject.loc[time_idx[0], 'bolus'] = current_bolus + dose
                                    current_type = df_subject.loc[time_idx[0], 'insulin_type_bolus']
                                    df_subject.loc[time_idx[0], 'insulin_type_bolus'] = f"{current_type} + {insulin_name}"
                                else:
                                    df_subject.loc[time_idx[0], 'bolus'] = dose
                                    df_subject.loc[time_idx[0], 'insulin_type_bolus'] = insulin_name
                            
                            elif is_long_acting_insulin(insulin_name):
                                # Add to basal (S.C. long-acting overrides CSII)
                                df_subject.loc[time_idx[0], 'basal'] = dose
                                df_subject.loc[time_idx[0], 'insulin_type_basal'] = insulin_name
        
        # Process meal data with proper filtering
        if 'Dietary intake' in df_raw.columns:
            for idx, row in df_raw.iterrows():
                if pd.notna(row.get('Date')) and pd.notna(row.get('Dietary intake')):
                    rounded_time = round_up_to_nearest_5min(pd.to_datetime(row['Date']))
                    time_idx = df_subject[df_subject['date'] == rounded_time].index
                    if len(time_idx) > 0:
                        meal_text = str(row['Dietary intake']).strip()
                        
                        # Filter out non-informative meal labels
                        if meal_text.lower() == 'fasting for examination':
                            # Set to NaN (will be empty in CSV)
                            pass
                        elif meal_text.lower() == 'data not available':
                            # Rename to "Meal without description"
                            df_subject.loc[time_idx[0], 'meal_label'] = 'Meal without description'
                        else:
                            # Keep the actual meal description
                            df_subject.loc[time_idx[0], 'meal_label'] = meal_text
        
        # Add demographics (only age, height, weight for time-series)
        if subject_id in demographics:
            demo = demographics[subject_id]
            # Only include non-demographic columns in time-series
            df_subject['age'] = demo.get('age', np.nan)
            df_subject['height'] = demo.get('height', np.nan) 
            df_subject['weight'] = demo.get('weight', np.nan)
            df_subject['gender'] = demo.get('gender', np.nan)
            df_subject['age_of_diagnosis'] = demo.get('age_of_diagnosis', np.nan)
            df_subject['ethnicity'] = demo.get('ethnicity', np.nan)

        # Add insulin column (bolus + basal)
        df_subject['insulin'] = np.nan
        bolus_values = df_subject['bolus'].fillna(0)
        basal_values = df_subject['basal'].fillna(0)
        # Only set insulin value if at least one of bolus or basal is not NaN
        insulin_mask = df_subject['bolus'].notna() | df_subject['basal'].notna()
        df_subject.loc[insulin_mask, 'insulin'] = bolus_values + basal_values
        
        # Add insulin_delivery_modality column
        df_subject['insulin_delivery_modality'] = np.nan
        # Default to SAP
        df_subject['insulin_delivery_modality'] = 'SAP'
        # Set to MDI when insulin_type_basal is NOT "Novolin R"
        mdi_mask = df_subject['insulin_type_basal'].notna() & (df_subject['insulin_type_basal'] != 'Novolin R')
        df_subject.loc[mdi_mask, 'insulin_delivery_modality'] = 'MDI'

        # Clean up empty string columns
        df_subject['insulin_type_bolus'] = df_subject['insulin_type_bolus'].replace('', np.nan)
        df_subject['insulin_type_basal'] = df_subject['insulin_type_basal'].replace('', np.nan)
        df_subject['meal_label'] = df_subject['meal_label'].replace('', np.nan)
        
        return df_subject


def get_subject_demographics(file_path):
    """Extract demographic data from the summary file with updated format."""
    summary_file = os.path.join(file_path, "Shanghai_T1DM_Summary.xlsx")
    demographics = {}
    
    if os.path.exists(summary_file):
        try:
            df_summary = pd.read_excel(summary_file)
            for _, row in df_summary.iterrows():
                patient_full_id = str(row['Patient Number'])
                subject_id = patient_full_id.split('_')[0]
                
                # Convert units
                height_feet = row['Height (m)'] * 3.28084
                weight_lbs = row['Weight (kg)'] * 2.20462
                age_of_diagnosis = row['Age (years)'] - row['Duration of Diabetes  (years)']
                
                demographics[subject_id] = {
                    'id': subject_id,
                    'source_file': 'ShanghaiT1DM',
                    'gender': 'Male' if row['Gender (Female=1, Male=2)'] == 2 else 'Female',
                    'age': row['Age (years)'],
                    'height': height_feet,
                    'weight': weight_lbs,
                    'age_of_diagnosis': age_of_diagnosis,
                    'ethnicity': 'Asian',
                    'TDD': np.nan
                }
        except Exception as e:
            print(f"Error loading demographics: {e}")
    
    return demographics


def main():
    """Main function to run the AZT1D parser with a hard-coded file path."""
    # Hard-coded file path - update this to your actual AZT1D dataset path
    file_path = "data/raw/Shanghai_T1DM/"

    # Create parser instance
    parser = Parser()

    # Process the data
    result = parser(file_path)
    result.to_csv('data/raw/ShanghaiT1DM.csv', index=False)

    if not result.empty:
        print(f"\nProcessing completed successfully!")
        print(f"Total records processed: {len(result)}")
        print(f"Unique subjects: {result['id'].nunique()}")
        print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    else:
        print("No data was processed. Please check the file path and data format.")


if __name__ == "__main__":
    main()

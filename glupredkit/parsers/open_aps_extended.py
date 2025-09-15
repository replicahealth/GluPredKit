#!/usr/bin/env python3
"""
OpenAPS Extended parser for processing OpenAPS diabetes dataset.
Processes open_aps.csv and integrates with OpenAPS Data Commons demographics.
"""

from glupredkit.parsers.base_parser import BaseParser
import pandas as pd
import numpy as np
import os


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.df_expanded = pd.DataFrame()
        self.df_demographics = pd.DataFrame()
        
    def __call__(self, file_path: str, *args):
        """
        Process OpenAPS data from open_aps.csv file.
        
        file_path -- the file path to the open_aps.csv file.
        """
        print(f"Processing OpenAPS data from {file_path}")
        
        # Read the open_aps.csv file
        df_raw = pd.read_csv(file_path)
        print(f"Loaded {len(df_raw)} records for {df_raw['id'].nunique()} subjects")
        
        # Process the data to match HUPA-UCM structure
        df_processed = self.process_open_aps_data(df_raw)
        
        if df_processed.empty:
            print("Error: No data was processed successfully!")
            return pd.DataFrame()
        
        print(f"Processed data: {len(df_processed)} records")
        
        # Store in df_expanded
        self.df_expanded = df_processed.copy()
        
        # Create demographics dataframe with one row per ID
        self.create_demographics_df()
        
        # Save demographics to CSV
        self.save_demographics()
        
        print(f"Stored {len(self.df_expanded)} records in df_expanded")
        print(f"Created demographics dataframe with {len(self.df_demographics)} unique subjects")
        
        return df_processed
    
    def load_demographics_data(self):
        """Load demographics data from OpenAPS Data Commons Excel file."""
        demographics_file = "/Users/miriamk.wolff/Documents/Repositories/Replica/TidepoolStudy/downloads/OpenAPS/OpenAPS Data Commons_demographics-n-231.xlsx"
        
        if not os.path.exists(demographics_file):
            print("Warning: OpenAPS demographics file not found")
            return pd.DataFrame()
        
        try:
            df_demo = pd.read_excel(demographics_file)
            # Rename the ID column for easier access
            id_col = 'Your OpenHumans OpenAPS Data Commons "project member ID"'
            df_demo = df_demo.rename(columns={id_col: 'id'})
            return df_demo
        except Exception as e:
            print(f"Error loading demographics file: {e}")
            return pd.DataFrame()
    
    def process_open_aps_data(self, df_raw):
        """Process and format open_aps data to match HUPA-UCM structure"""
        df_processed = df_raw.copy()
        
        # Convert date column to datetime
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # Convert basal by dividing by 12
        df_processed['basal'] = df_processed['basal'] / 12
        
        # Load demographics data for additional columns
        demographics_data = self.load_demographics_data()
        
        # Add missing columns to match HUPA-UCM structure
        df_processed['calories_burned'] = np.nan
        df_processed['heartrate'] = np.nan  
        df_processed['steps'] = np.nan
        
        # Initialize demographic columns
        df_processed['age'] = np.nan
        df_processed['weight'] = np.nan
        df_processed['height'] = np.nan
        df_processed['insulin_delivery_algorithm'] = np.nan
        
        # Fill in demographics data per subject if available
        if not demographics_data.empty:
            for subject_id in df_processed['id'].unique():
                demo_row = demographics_data[demographics_data['id'] == subject_id]
                if not demo_row.empty:
                    row = demo_row.iloc[0]
                    
                    # Parse weight (various formats possible)
                    weight = self.parse_weight(row.get('How much do you weigh?'))
                    
                    # Parse height (various formats possible)  
                    height = self.parse_height(row.get('How tall are you?'))
                    
                    # Get birth year for dynamic age calculation
                    birth_year = self.extract_year(row.get('When were you born?'))
                    
                    # Get and clean algorithm information
                    algorithm_raw = row.get('What type of DIY close loop technology do you use? Select all that you actively use:')
                    algorithm = self.clean_algorithm(algorithm_raw)
                    
                    # Set demographics for all records of this subject
                    df_processed.loc[df_processed['id'] == subject_id, 'weight'] = weight
                    df_processed.loc[df_processed['id'] == subject_id, 'height'] = height
                    df_processed.loc[df_processed['id'] == subject_id, 'insulin_delivery_algorithm'] = algorithm
                    
                    # Calculate dynamic age for each row based on time series date
                    if birth_year:
                        subject_mask = df_processed['id'] == subject_id
                        subject_dates = df_processed.loc[subject_mask, 'date']
                        ages = subject_dates.dt.year - birth_year
                        df_processed.loc[subject_mask, 'age'] = ages
        
        # Add device and delivery method columns (OpenAPS specific)
        df_processed['insulin_delivery_modality'] = 'AID'  # OpenAPS is DIY Automated Insulin Delivery
        df_processed['insulin_delivery_device'] = np.nan  # Various pumps used, set in demographics
        df_processed['cgm_device'] = np.nan  # Various CGMs used, set in demographics
        df_processed['source_file'] = 'OpenAPS'
        
        # Add empty columns to match HUPA-UCM structure
        empty_columns = ['context_description_cache', 'tag', 'absorption_time', 'acceleration',
                         'air_temp', 'galvanic_skin_response', 'insulin_type_basal',
                         'insulin_type_bolus', 'is_pregnant', 'meal_label', 'scheduled_basal',
                         'skin_temp', 'workout_duration', 'workout_intensity', 'workout_label']

        for col in empty_columns:
            df_processed[col] = np.nan

        # Select and order the columns to match HUPA-UCM structure exactly
        final_columns = ['date', 'id', 'CGM', 'calories_burned', 'heartrate', 'steps', 'basal', 'bolus', 'carbs',
                        'age', 'weight', 'height', 'insulin_delivery_modality', 'insulin',
                         'insulin_delivery_device', 'cgm_device', 'source_file']
        
        # Add the special columns
        special_columns = ['is_test', 'context_description_cache', 'tag', 'absorption_time', 'acceleration',
                          'air_temp', 'galvanic_skin_response', 'insulin_delivery_algorithm', 'insulin_type_basal',
                          'insulin_type_bolus', 'is_pregnant', 'meal_label', 'scheduled_basal',
                          'skin_temp', 'workout_duration', 'workout_intensity', 'workout_label']
        
        df_processed = df_processed[final_columns + special_columns]
        
        return df_processed.sort_values(['id', 'date']).reset_index(drop=True)
    
    def parse_weight(self, weight_str):
        """Parse weight from various string formats to pounds."""
        if pd.isna(weight_str):
            return np.nan
        
        weight_str = str(weight_str).lower().strip()
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', weight_str)
        if not numbers:
            return np.nan
        
        weight = float(numbers[0])
        
        # Convert kg to lbs if specified
        if 'kg' in weight_str:
            weight = weight * 2.20462
        # Otherwise assume lbs
        
        return weight
    
    def parse_height(self, height_str):
        """Parse height from various string formats to feet."""
        if pd.isna(height_str):
            return np.nan
        
        height_str = str(height_str).lower().strip()
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', height_str)
        if not numbers:
            return np.nan
        
        # Handle feet and inches (e.g., "5'10", "5 feet 10 inches")
        if "'" in height_str or 'feet' in height_str:
            feet_match = re.search(r'(\d+)', height_str)
            inches_match = re.search(r'(\d+)(?:\s*inch|")', height_str)
            
            if feet_match:
                feet = float(feet_match.group(1))
                inches = float(inches_match.group(1)) if inches_match else 0
                return feet + inches / 12.0
        
        # Handle cm
        elif 'cm' in height_str:
            cm = float(numbers[0])
            return cm * 0.0328084
        
        # Handle just inches
        elif 'inch' in height_str:
            inches = float(numbers[0])
            return inches / 12.0
        
        # Default assume it's already in feet or a decimal feet value
        return float(numbers[0])
    
    def calculate_age(self, birth_info):
        """Calculate age from birth year information."""
        if pd.isna(birth_info):
            return np.nan
        
        import re
        birth_str = str(birth_info).strip()
        
        # Extract year
        year_match = re.search(r'(19|20)\d{2}', birth_str)
        if year_match:
            birth_year = int(year_match.group())
            # Calculate age as of 2020 (approximate median year of OpenAPS data)
            return 2020 - birth_year
        
        return np.nan
    
    def clean_algorithm(self, algorithm_raw):
        """Clean and shorten algorithm descriptions."""
        if pd.isna(algorithm_raw):
            return np.nan
        
        algorithm_str = str(algorithm_raw).strip()
        
        # Define mapping for algorithm cleaning
        algorithm_map = {
            'A "traditional" OpenAPS rig using the oref0 algorithm (i.e. using a Raspberry Pi/Carelink; or an Edison/Explorer Board; etc.)': 'OpenAPS oref0',
            'OpenAPS using the oref1 algorithm and hard-wired "on" SMB/UAM': 'OpenAPS oref1',
            'Using UMA but not SMB from oref1': 'OpenAPS oref1 UMA',
            'OpenAPS Oref1': 'OpenAPS oref1',
            'Loopkit/Loop': 'Loop',
            'AndroidAPS': 'AndroidAPS',
            'Haps': 'Haps'
        }
        
        # Split by comma if multiple algorithms
        algorithms = [alg.strip() for alg in algorithm_str.split(',')]
        cleaned_algorithms = []
        
        for algo in algorithms:
            # Find the best match in our mapping
            cleaned = algorithm_map.get(algo, algo)
            if cleaned not in cleaned_algorithms:  # Avoid duplicates
                cleaned_algorithms.append(cleaned)
        
        return ', '.join(cleaned_algorithms) if cleaned_algorithms else np.nan
    
    def standardize_ethnicity(self, ethnicity_raw):
        """Standardize ethnicity values according to specified mappings."""
        if pd.isna(ethnicity_raw):
            return np.nan
        
        ethnicity_str = str(ethnicity_raw).strip()
        
        # Define ethnicity standardization mapping
        ethnicity_map = {
            'Ashkenaz Jewish': 'White',
            'I prefer not to answer': np.nan,
            "halfbreed' white/latino": 'White, Hispanic/Latino',
            '1/2 Spanish, 1/4 German, 1/4 Prussian': 'White',
            # Keep existing values as they are
            'White': 'White',
            'Asian/Pacific Islander': 'Asian/Pacific Islander',
            'Two or more races': 'Two or more races',
            'Mixed Race': 'Mixed Race'
        }
        
        return ethnicity_map.get(ethnicity_str, ethnicity_str)
    
    def create_demographics_df(self):
        """Create demographics dataframe with one row per unique ID."""
        
        # Get unique subject IDs from processed data
        unique_ids = self.df_expanded['id'].unique() if not self.df_expanded.empty else []
        
        # Load demographics data
        demographics_data = self.load_demographics_data()
        
        demographics_list = []
        for subject_id in unique_ids:
            demo_row = demographics_data[demographics_data['id'] == subject_id] if not demographics_data.empty else pd.DataFrame()
            
            if not demo_row.empty:
                row = demo_row.iloc[0]
                
                # Map gender
                gender_raw = row.get('Gender')
                if pd.isna(gender_raw):
                    gender = np.nan
                elif str(gender_raw).lower() in ['male', 'transgender male']:
                    gender = 'Male'  
                elif str(gender_raw).lower() in ['female']:
                    gender = 'Female'
                else:
                    gender = np.nan
                
                # Parse age of diagnosis from both birth year and diagnosis year
                diagnosis_info = row.get('When were you diagnosed with diabetes?')
                birth_info = row.get('When were you born?')
                age_of_diagnosis = self.parse_age_of_diagnosis(diagnosis_info, birth_info)
                
                # Parse TDD (Total Daily Dose)
                tdd_info = row.get('How many units of insulin do you take per day?')
                tdd = self.parse_tdd(tdd_info)
                
                # Parse and standardize ethnicity
                ethnicity_raw = row.get('Ethnicity origin:')
                ethnicity = self.standardize_ethnicity(ethnicity_raw)
                
                demographics_list.append({
                    'id': subject_id,
                    'gender': gender,
                    'age_of_diagnosis': age_of_diagnosis,
                    'TDD': tdd,
                    'ethnicity': ethnicity,
                    'source_file': 'OpenAPS'
                })
            else:
                # Fallback for subjects not found in demographics
                demographics_list.append({
                    'id': subject_id,
                    'gender': np.nan,
                    'age_of_diagnosis': np.nan,
                    'TDD': np.nan,
                    'ethnicity': np.nan,
                    'source_file': 'OpenAPS'
                })
        
        self.df_demographics = pd.DataFrame(demographics_list)
        self.df_demographics = self.df_demographics.sort_values('id').reset_index(drop=True)
    
    def parse_age_of_diagnosis(self, diagnosis_info, birth_info):
        """Parse age of diagnosis from various formats, using birth year and diagnosis year when possible."""
        import re
        
        # First try to calculate from birth year and diagnosis year
        if pd.notna(diagnosis_info) and pd.notna(birth_info):
            # Extract years from both fields
            diagnosis_year = self.extract_year(diagnosis_info)
            birth_year = self.extract_year(birth_info)
            
            if diagnosis_year and birth_year:
                age_at_diagnosis = diagnosis_year - birth_year
                if 0 <= age_at_diagnosis <= 100:  # Reasonable age range
                    return float(age_at_diagnosis)
        
        # Fallback to parsing age directly from diagnosis info
        if pd.isna(diagnosis_info):
            return np.nan
        
        diagnosis_str = str(diagnosis_info).lower()
        
        # Look for age patterns (e.g., "when I was 25 years old")
        age_match = re.search(r'(\d+)\s*(?:years?\s*old|yrs?)', diagnosis_str)
        if age_match:
            return float(age_match.group(1))
        
        # Look for just numbers if it's a simple age
        number_match = re.search(r'^\d+$', diagnosis_str.strip())
        if number_match:
            age = int(number_match.group())
            if 0 <= age <= 100:  # Reasonable age range
                return float(age)
        
        return np.nan
    
    def extract_year(self, date_info):
        """Extract year from various date formats."""
        if pd.isna(date_info):
            return None
        
        import re
        date_str = str(date_info)
        
        # Look for 4-digit years (1900-2099)
        year_match = re.search(r'(19|20)\d{2}', date_str)
        if year_match:
            return int(year_match.group())
        
        return None
    
    def parse_tdd(self, tdd_info):
        """Parse Total Daily Dose from various formats."""
        if pd.isna(tdd_info):
            return np.nan
        
        import re
        tdd_str = str(tdd_info).strip()
        
        # Extract first number found
        numbers = re.findall(r'\d+\.?\d*', tdd_str)
        if numbers:
            tdd = float(numbers[0])
            if 10 <= tdd <= 200:  # Reasonable TDD range
                return tdd
        
        return np.nan
    
    def save_demographics(self, output_path: str = None):
        """Save the demographics dataframe to CSV."""
        if output_path is None:
            output_path = "open_aps_demographics.csv"
        
        if not self.df_demographics.empty:
            self.df_demographics.to_csv(output_path, index=False)
            print(f"Demographics data saved to {output_path}")
        else:
            print("No demographics data to save. Process data first.")


def get_dataset_info(file_path):
    """Get information about the OpenAPS dataset."""
    if not os.path.exists(file_path):
        return {'file_exists': False}
    
    try:
        df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
        
        info = {
            'file_exists': True,
            'total_records': len(pd.read_csv(file_path, usecols=['id'])),
            'subjects': sorted(df['id'].unique().tolist()),
            'unique_subjects': df['id'].nunique(),
            'columns': df.columns.tolist(),
            'date_ranges': {}
        }
        
        # Get date ranges for first few subjects
        df['date'] = pd.to_datetime(df['date'])
        for subject_id in info['subjects'][:5]:
            subject_data = df[df['id'] == subject_id]
            if not subject_data.empty:
                dates = subject_data['date'].dropna()
                if not dates.empty:
                    info['date_ranges'][subject_id] = {
                        'start': dates.min(),
                        'end': dates.max(),
                        'duration_days': (dates.max() - dates.min()).days
                    }
        
        return info
    except Exception as e:
        return {'file_exists': True, 'error': str(e)}
"""
The IOBP2 parser is processing the data from the IOBP2 dataset and returning the data merged into
the same time grid in a dataframe.
"""
try:
    from .base_parser import BaseParser
except ImportError:
    # For direct execution - import BaseParser directly
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from base_parser import BaseParser

import pandas as pd
import numpy as np
import os


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the folder containing "IOBP2 RCT Public Dataset" folder.
        """
        print(f"Processing IOBP2 data from {file_path}")
        
        # Construct path to the CGM data file
        cgm_data_path = os.path.join(file_path, "IOBP2 RCT Public Dataset", "Data Tables", "IOBP2DeviceiLet.txt")
        
        # Check if CGM data file exists
        if not os.path.exists(cgm_data_path):
            print(f"Error: CGM data file not found at: {cgm_data_path}")
            print("Expected structure: [file_path]/IOBP2 RCT Public Dataset/Data Tables/IOBP2DeviceiLet.txt")
            return pd.DataFrame()
        
        # Process CGM data
        print(f"Loading CGM data from: {cgm_data_path}")
        df_glucose = self.load_cgm_data(cgm_data_path)
        
        if df_glucose.empty:
            print("No CGM data found")
            return pd.DataFrame()
        
        print(f"Loaded CGM data: {df_glucose.shape[0]} records for {df_glucose['id'].nunique()} subjects")

        # Process bolus data
        print(f"Loading bolus data from: {cgm_data_path}")
        df_bolus = self.load_bolus_data(cgm_data_path)
        
        print(f"Loaded bolus data: {df_bolus.shape[0]} records for {df_bolus['id'].nunique() if not df_bolus.empty else 0} subjects")

        # Process basal data
        print(f"Loading basal data from: {cgm_data_path}")
        df_basal = self.load_basal_data(cgm_data_path)
        
        print(f"Loaded basal data: {df_basal.shape[0]} records for {df_basal['id'].nunique() if not df_basal.empty else 0} subjects")

        # Process meal label data
        print(f"Loading meal label data from: {cgm_data_path}")
        df_meal_labels = self.load_meal_label_data(cgm_data_path)
        
        print(f"Loaded meal label data: {df_meal_labels.shape[0]} records for {df_meal_labels['id'].nunique() if not df_meal_labels.empty else 0} subjects")

        # Process age data
        age_data_path = os.path.join(file_path, "IOBP2 RCT Public Dataset", "Data Tables", "IOBP2PtRoster.txt")
        print(f"Loading age data from: {age_data_path}")
        df_age = self.load_age_data(age_data_path)
        
        print(f"Loaded age data: {df_age.shape[0] if not df_age.empty else 0} subjects")

        # Process gender data
        gender_data_path = os.path.join(file_path, "IOBP2 RCT Public Dataset", "Data Tables", "IOBP2DiabScreening.txt")
        print(f"Loading gender data from: {gender_data_path}")
        df_gender = self.load_gender_data(gender_data_path)
        
        print(f"Loaded gender data: {df_gender.shape[0] if not df_gender.empty else 0} subjects")

        # Process height/weight data
        height_weight_data_path = os.path.join(file_path, "IOBP2 RCT Public Dataset", "Data Tables", "IOBP2HeightWeight.txt")
        print(f"Loading height/weight data from: {height_weight_data_path}")
        df_height_weight = self.load_height_weight_data(height_weight_data_path)
        
        print(f"Loaded height/weight data: {df_height_weight.shape[0] if not df_height_weight.empty else 0} records")

        # Process metadata (device, algorithm, ethnicity, etc.)
        print(f"Loading metadata from roster and screening data...")
        df_metadata = self.load_metadata(file_path)
        
        print(f"Loaded metadata: {df_metadata.shape[0] if not df_metadata.empty else 0} subjects")

        # For now, process other data types as empty (to be implemented later)
        _, df_meals, _, _, df_exercise = self.get_dataframes(file_path)

        # Resample and merge data
        df_resampled = self.resample_data(df_glucose, df_meals, df_bolus, df_basal, df_exercise, df_meal_labels, df_age, df_gender, df_height_weight, df_metadata)
        
        return df_resampled

    def load_cgm_data(self, cgm_data_path):
        """
        Load and process CGM data from IOBP2DeviceiLet.txt file.
        """
        try:
            # First try to detect the delimiter by reading a few lines
            with open(cgm_data_path, 'r') as f:
                first_line = f.readline().strip()
                print(f"First line: {first_line[:200]}...")
                
            # Check if it's pipe-delimited based on the column names we can see
            if '|' in first_line:
                print("Detected pipe-delimited format")
                print("Loading large dataset (this may take a moment)...")
                df = pd.read_csv(cgm_data_path, delimiter='|', low_memory=False)
            else:
                print("Using tab-delimited format")
                df = pd.read_csv(cgm_data_path, delimiter='\t', low_memory=False)
            
            print(f"Raw CGM data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for required columns (adjust based on actual file structure)
            required_cols = ['Subject', 'DateTime']  # These might be different in actual file
            
            # Show first few rows to understand structure
            print("First 5 rows of CGM data:")
            print(df.head())
            
            # Process subject ID based on IOBP2 structure
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects")
            else:
                print("Warning: Could not find PtID column")
                df['id'] = 'unknown'
            
            # Process datetime using DeviceDtTm column
            if 'DeviceDtTm' in df.columns:
                print(f"Using datetime column: DeviceDtTm")
                print(f"Sample datetime values: {df['DeviceDtTm'].head()}")
                try:
                    # Optimize datetime parsing by specifying format (based on sample: 8/14/2020 12:01:23 AM)
                    print("Parsing datetime (this may take a moment for large datasets)...")
                    df['date'] = pd.to_datetime(df['DeviceDtTm'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                    
                    # Check for any parsing failures
                    failed_dates = df['date'].isna().sum()
                    if failed_dates > 0:
                        print(f"Warning: {failed_dates} dates could not be parsed")
                        # Fallback to automatic parsing for failed dates
                        mask = df['date'].isna()
                        if mask.any():
                            df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'DeviceDtTm'], errors='coerce')
                    
                    # Remove rows with invalid dates
                    df = df[df['date'].notna()]
                    df.set_index('date', inplace=True)
                    print("Successfully parsed datetime")
                except Exception as e:
                    print(f"Error parsing datetime: {e}")
                    return pd.DataFrame()
            else:
                print("Warning: Could not find DeviceDtTm column")
                return pd.DataFrame()
            
            # Process CGM values using CGMVal column
            if 'CGMVal' in df.columns:
                print(f"Using glucose column: CGMVal")
                df['CGM'] = pd.to_numeric(df['CGMVal'], errors='coerce')
                print(f"Sample CGM values: {df['CGM'].head()}")
            else:
                print("Warning: Could not find CGMVal column")
                # Show available columns to help with debugging
                print("Available columns:", list(df.columns))
                return pd.DataFrame()

            # Filter out invalid glucose values
            df = df[df['CGM'].notna()]

            # Keep only necessary columns
            df_processed = df[['id', 'CGM']].copy()
            
            print(f"Processed CGM data: {df_processed.shape[0]} valid glucose readings")
            print(f"Glucose range: {df_processed['CGM'].min():.1f} - {df_processed['CGM'].max():.1f} mg/dL")
            print(f"Subjects: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading CGM data: {e}")
            return pd.DataFrame()

    def load_bolus_data(self, cgm_data_path):
        """
        Load and process bolus data from IOBP2DeviceiLet.txt file.
        """
        try:
            # First try to detect the delimiter by reading a few lines
            with open(cgm_data_path, 'r') as f:
                first_line = f.readline().strip()
                
            # Check if it's pipe-delimited based on the column names we can see
            if '|' in first_line:
                print("Detected pipe-delimited format for bolus data")
                print("Loading large dataset for bolus processing (this may take a moment)...")
                df = pd.read_csv(cgm_data_path, delimiter='|', low_memory=False)
            else:
                print("Using tab-delimited format for bolus data")
                df = pd.read_csv(cgm_data_path, delimiter='\t', low_memory=False)
            
            print(f"Raw data shape for bolus processing: {df.shape}")
            
            # Show first few rows to understand structure
            print("First 3 rows of bolus data:")
            print(df.head(3))
            
            # Process subject ID based on IOBP2 structure
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects for bolus data")
            else:
                print("Warning: Could not find PtID column")
                df['id'] = 'unknown'
            
            # Process datetime using DeviceDtTm column
            if 'DeviceDtTm' in df.columns:
                print(f"Using datetime column: DeviceDtTm")
                try:
                    # Optimize datetime parsing by specifying format (based on sample: 8/14/2020 12:01:23 AM)
                    print("Parsing datetime for bolus data (this may take a moment for large datasets)...")
                    df['date'] = pd.to_datetime(df['DeviceDtTm'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                    
                    # Check for any parsing failures
                    failed_dates = df['date'].isna().sum()
                    if failed_dates > 0:
                        print(f"Warning: {failed_dates} dates could not be parsed")
                        # Fallback to automatic parsing for failed dates
                        mask = df['date'].isna()
                        if mask.any():
                            df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'DeviceDtTm'], errors='coerce')
                    
                    # Remove rows with invalid dates
                    df = df[df['date'].notna()]
                    df.set_index('date', inplace=True)
                    print("Successfully parsed datetime for bolus data")
                except Exception as e:
                    print(f"Error parsing datetime: {e}")
                    return pd.DataFrame()
            else:
                print("Warning: Could not find DeviceDtTm column")
                return pd.DataFrame()
            
            # Process bolus values using BolusDelivPrev and MealBolusDelivPrev columns
            bolus_total = 0
            if 'BolusDelivPrev' in df.columns and 'MealBolusDelivPrev' in df.columns:
                print("Processing bolus data from BolusDelivPrev and MealBolusDelivPrev")
                
                # Convert to numeric
                df['BolusDelivPrev'] = pd.to_numeric(df['BolusDelivPrev'], errors='coerce').fillna(0)
                df['MealBolusDelivPrev'] = pd.to_numeric(df['MealBolusDelivPrev'], errors='coerce').fillna(0)
                
                # Sum both bolus columns
                df['bolus'] = df['BolusDelivPrev'] + df['MealBolusDelivPrev']
                
                print(f"Sample bolus values: {df[['BolusDelivPrev', 'MealBolusDelivPrev', 'bolus']].head()}")
                print(f"Total non-zero bolus records: {(df['bolus'] > 0).sum()}")
                
                bolus_total = (df['bolus'] > 0).sum()
                
            elif 'BolusDelivPrev' in df.columns:
                print("Processing bolus data from BolusDelivPrev only")
                df['BolusDelivPrev'] = pd.to_numeric(df['BolusDelivPrev'], errors='coerce').fillna(0)
                df['bolus'] = df['BolusDelivPrev']
                bolus_total = (df['bolus'] > 0).sum()
                
            else:
                print("Warning: Could not find bolus columns")
                return pd.DataFrame()
                
            # Keep only necessary columns
            df_processed = df[['id', 'bolus']].copy()
            
            print(f"Processed bolus data: {df_processed.shape[0]} total records")
            print(f"Total bolus records: {bolus_total}")
            print(f"Subjects: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading bolus data: {e}")
            return pd.DataFrame()

    def load_basal_data(self, cgm_data_path):
        """
        Load and process basal data from IOBP2DeviceiLet.txt file.
        """
        try:
            # First try to detect the delimiter by reading a few lines
            with open(cgm_data_path, 'r') as f:
                first_line = f.readline().strip()
                
            # Check if it's pipe-delimited based on the column names we can see
            if '|' in first_line:
                print("Detected pipe-delimited format for basal data")
                print("Loading large dataset for basal processing (this may take a moment)...")
                df = pd.read_csv(cgm_data_path, delimiter='|', low_memory=False)
            else:
                print("Using tab-delimited format for basal data")
                df = pd.read_csv(cgm_data_path, delimiter='\t', low_memory=False)
            
            print(f"Raw data shape for basal processing: {df.shape}")
            
            # Show first few rows to understand structure
            print("First 3 rows of basal data:")
            print(df.head(3))
            
            # Process subject ID based on IOBP2 structure
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects for basal data")
            else:
                print("Warning: Could not find PtID column")
                df['id'] = 'unknown'
            
            # Process datetime using DeviceDtTm column
            if 'DeviceDtTm' in df.columns:
                print(f"Using datetime column: DeviceDtTm")
                try:
                    # Optimize datetime parsing by specifying format (based on sample: 8/14/2020 12:01:23 AM)
                    print("Parsing datetime for basal data (this may take a moment for large datasets)...")
                    df['date'] = pd.to_datetime(df['DeviceDtTm'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                    
                    # Check for any parsing failures
                    failed_dates = df['date'].isna().sum()
                    if failed_dates > 0:
                        print(f"Warning: {failed_dates} dates could not be parsed")
                        # Fallback to automatic parsing for failed dates
                        mask = df['date'].isna()
                        if mask.any():
                            df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'DeviceDtTm'], errors='coerce')
                    
                    # Remove rows with invalid dates
                    df = df[df['date'].notna()]
                    df.set_index('date', inplace=True)
                    print("Successfully parsed datetime for basal data")
                except Exception as e:
                    print(f"Error parsing datetime: {e}")
                    return pd.DataFrame()
            else:
                print("Warning: Could not find DeviceDtTm column")
                return pd.DataFrame()
            
            # Process basal values using BasalDelivPrev column
            if 'BasalDelivPrev' in df.columns:
                print("Processing basal data from BasalDelivPrev")
                
                # Convert to numeric
                df['BasalDelivPrev'] = pd.to_numeric(df['BasalDelivPrev'], errors='coerce').fillna(0)
                
                # Use basal column directly
                df['basal'] = df['BasalDelivPrev']
                
                print(f"Sample basal values: {df['basal'].head()}")
                print(f"Total non-zero basal records: {(df['basal'] > 0).sum()}")
                
                basal_total = (df['basal'] > 0).sum()
                
            else:
                print("Warning: Could not find BasalDelivPrev column")
                return pd.DataFrame()
                
            # Keep only necessary columns
            df_processed = df[['id', 'basal']].copy()
            
            print(f"Processed basal data: {df_processed.shape[0]} total records")
            print(f"Total basal records: {basal_total}")
            print(f"Subjects: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading basal data: {e}")
            return pd.DataFrame()

    def load_meal_label_data(self, cgm_data_path):
        """
        Load and process meal label data from IOBP2DeviceiLet.txt file.
        """
        try:
            # First try to detect the delimiter by reading a few lines
            with open(cgm_data_path, 'r') as f:
                first_line = f.readline().strip()
                
            # Check if it's pipe-delimited based on the column names we can see
            if '|' in first_line:
                print("Detected pipe-delimited format for meal label data")
                print("Loading large dataset for meal label processing (this may take a moment)...")
                df = pd.read_csv(cgm_data_path, delimiter='|', low_memory=False)
            else:
                print("Using tab-delimited format for meal label data")
                df = pd.read_csv(cgm_data_path, delimiter='\t', low_memory=False)
            
            print(f"Raw data shape for meal label processing: {df.shape}")
            
            # Show first few rows to understand structure
            print("First 3 rows of meal label data:")
            print(df.head(3))
            
            # Process subject ID based on IOBP2 structure
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects for meal label data")
            else:
                print("Warning: Could not find PtID column")
                df['id'] = 'unknown'
            
            # Process datetime using DeviceDtTm column
            if 'DeviceDtTm' in df.columns:
                print(f"Using datetime column: DeviceDtTm")
                try:
                    # Optimize datetime parsing by specifying format (based on sample: 8/14/2020 12:01:23 AM)
                    print("Parsing datetime for meal label data (this may take a moment for large datasets)...")
                    df['date'] = pd.to_datetime(df['DeviceDtTm'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                    
                    # Check for any parsing failures
                    failed_dates = df['date'].isna().sum()
                    if failed_dates > 0:
                        print(f"Warning: {failed_dates} dates could not be parsed")
                        # Fallback to automatic parsing for failed dates
                        mask = df['date'].isna()
                        if mask.any():
                            df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'DeviceDtTm'], errors='coerce')
                    
                    # Remove rows with invalid dates
                    df = df[df['date'].notna()]
                    df.set_index('date', inplace=True)
                    print("Successfully parsed datetime for meal label data")
                except Exception as e:
                    print(f"Error parsing datetime: {e}")
                    return pd.DataFrame()
            else:
                print("Warning: Could not find DeviceDtTm column")
                return pd.DataFrame()
            
            # Process meal label using MealBolusDelivPrev column
            if 'MealBolusDelivPrev' in df.columns:
                print("Processing meal label data from MealBolusDelivPrev")
                
                # Convert to numeric
                df['MealBolusDelivPrev'] = pd.to_numeric(df['MealBolusDelivPrev'], errors='coerce').fillna(0)
                
                # Create meal label: NaN by default, "Meal" if meal bolus > 0
                df['meal_label'] = np.nan
                df.loc[df['MealBolusDelivPrev'] > 0, 'meal_label'] = 'Meal'
                
                print(f"Sample meal bolus values: {df['MealBolusDelivPrev'].head()}")
                print(f"Total meal bolus records: {(df['MealBolusDelivPrev'] > 0).sum()}")
                
                meal_total = (df['MealBolusDelivPrev'] > 0).sum()
                
            else:
                print("Warning: Could not find MealBolusDelivPrev column")
                return pd.DataFrame()
                
            # Keep only necessary columns
            df_processed = df[['id', 'meal_label']].copy()
            
            print(f"Processed meal label data: {df_processed.shape[0]} total records")
            print(f"Total meal events: {meal_total}")
            print(f"Subjects: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading meal label data: {e}")
            return pd.DataFrame()

    def load_age_data(self, age_data_path):
        """
        Load and process age data from IOBP2PtRoster.txt file.
        """
        try:
            # Check if age data file exists
            if not os.path.exists(age_data_path):
                print(f"Warning: Age data file not found at: {age_data_path}")
                return pd.DataFrame()
            
            print("Loading age data (this may take a moment)...")
            # Load the roster file - it's pipe-delimited
            df = pd.read_csv(age_data_path, delimiter='|', low_memory=False)
            
            print(f"Raw age data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows to understand structure
            print("First 5 rows of age data:")
            print(df.head())
            
            # Process subject ID to match other data
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects in age data")
            else:
                print("Warning: Could not find PtID column in age data")
                return pd.DataFrame()
            
            # Process age data
            if 'AgeAsofEnrollDt' in df.columns:
                print("Processing age data from AgeAsofEnrollDt")
                df['age'] = pd.to_numeric(df['AgeAsofEnrollDt'], errors='coerce')
                print(f"Sample age values: {df['age'].head()}")
                print(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
            else:
                print("Warning: Could not find AgeAsofEnrollDt column")
                print("Available columns:", list(df.columns))
                return pd.DataFrame()

            # Keep only necessary columns  
            df_processed = df[['id', 'age']].copy()
            
            # Remove any rows with missing age data
            df_processed = df_processed[df_processed['age'].notna()]
            
            print(f"Processed age data: {df_processed.shape[0]} subjects with valid age")
            print(f"Subjects: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading age data: {e}")
            return pd.DataFrame()

    def load_gender_data(self, gender_data_path):
        """
        Load and process gender data from IOBP2DiabScreening.txt file.
        """
        try:
            # Check if gender data file exists
            if not os.path.exists(gender_data_path):
                print(f"Warning: Gender data file not found at: {gender_data_path}")
                return pd.DataFrame()
            
            print("Loading gender data (this may take a moment)...")
            # Load the screening file - it's pipe-delimited
            df = pd.read_csv(gender_data_path, delimiter='|', low_memory=False)
            
            print(f"Raw gender data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows to understand structure
            print("First 5 rows of gender data:")
            print(df.head())
            
            # Process subject ID to match other data
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects in gender data")
            else:
                print("Warning: Could not find PtID column in gender data")
                return pd.DataFrame()
            
            # Process gender data
            if 'Sex' in df.columns:
                print("Processing gender data from Sex column")
                # Map M to Male and F to Female
                gender_mapping = {'M': 'Male', 'F': 'Female'}
                df['gender'] = df['Sex'].map(gender_mapping)
                
                print(f"Sample gender values: {df[['Sex', 'gender']].head()}")
                print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
            else:
                print("Warning: Could not find Sex column")
                print("Available columns:", list(df.columns))
                return pd.DataFrame()

            # Keep only necessary columns  
            df_processed = df[['id', 'gender']].copy()
            
            # Remove any rows with missing gender data
            df_processed = df_processed[df_processed['gender'].notna()]
            
            print(f"Processed gender data: {df_processed.shape[0]} subjects with valid gender")
            print(f"Subjects: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading gender data: {e}")
            return pd.DataFrame()

    def load_height_weight_data(self, height_weight_data_path):
        """
        Load and process height/weight data from IOBP2HeightWeight.txt file.
        Converts all weights to lbs and heights to feet, with timestamps at 8 AM.
        """
        try:
            # Check if height/weight data file exists
            if not os.path.exists(height_weight_data_path):
                print(f"Warning: Height/weight data file not found at: {height_weight_data_path}")
                return pd.DataFrame()
            
            print("Loading height/weight data (this may take a moment)...")
            # Load the height/weight file - it's pipe-delimited
            df = pd.read_csv(height_weight_data_path, delimiter='|', low_memory=False)
            
            print(f"Raw height/weight data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows to understand structure
            print("First 5 rows of height/weight data:")
            print(df.head())
            
            # Process subject ID to match other data
            if 'PtID' in df.columns:
                df['id'] = 'IOBP2-' + df['PtID'].astype(str)
                print(f"Found {df['PtID'].nunique()} unique subjects in height/weight data")
            else:
                print("Warning: Could not find PtID column in height/weight data")
                return pd.DataFrame()
            
            processed_records = []
            
            # Process weight data
            if 'Weight' in df.columns and 'WeightUnits' in df.columns and 'WeightAssessDt' in df.columns:
                print("Processing weight data...")
                weight_data = df[['id', 'Weight', 'WeightUnits', 'WeightAssessDt']].copy()
                weight_data = weight_data[weight_data['Weight'].notna() & weight_data['WeightAssessDt'].notna()]
                
                # Convert weights to lbs
                weight_data['weight_lbs'] = weight_data['Weight'].astype(float)
                kg_mask = weight_data['WeightUnits'].str.lower() == 'kg'
                weight_data.loc[kg_mask, 'weight_lbs'] *= 2.20462  # Convert kg to lbs
                
                # Parse dates and set time to 8 AM
                weight_data['date'] = pd.to_datetime(weight_data['WeightAssessDt'], errors='coerce')
                weight_data['date'] = weight_data['date'].dt.date
                weight_data['date'] = pd.to_datetime(weight_data['date'].astype(str) + ' 08:00:00')
                
                # Prepare records
                for _, row in weight_data.iterrows():
                    if pd.notna(row['date']) and pd.notna(row['weight_lbs']):
                        processed_records.append({
                            'id': row['id'],
                            'date': row['date'],
                            'weight': row['weight_lbs'],
                            'height': np.nan
                        })
                
                print(f"Processed {len(weight_data)} weight records")
                print(f"Weight range: {weight_data['weight_lbs'].min():.1f} - {weight_data['weight_lbs'].max():.1f} lbs")
            
            # Process height data  
            if 'Height' in df.columns and 'HeightUnits' in df.columns and 'HeightAssessDt' in df.columns:
                print("Processing height data...")
                height_data = df[['id', 'Height', 'HeightUnits', 'HeightAssessDt']].copy()
                height_data = height_data[height_data['Height'].notna() & height_data['HeightAssessDt'].notna()]
                
                # Convert heights to feet
                height_data['height_feet'] = height_data['Height'].astype(float)
                
                # Convert cm to inches first, then to feet
                cm_mask = height_data['HeightUnits'].str.lower() == 'cm'
                height_data.loc[cm_mask, 'height_feet'] /= 2.54  # Convert cm to inches
                
                # Convert all to feet (both cm->inches and original inches)
                height_data['height_feet'] /= 12  # Convert inches to feet
                
                # Parse dates and set time to 8 AM
                height_data['date'] = pd.to_datetime(height_data['HeightAssessDt'], errors='coerce')
                height_data['date'] = height_data['date'].dt.date
                height_data['date'] = pd.to_datetime(height_data['date'].astype(str) + ' 08:00:00')
                
                # Prepare records
                for _, row in height_data.iterrows():
                    if pd.notna(row['date']) and pd.notna(row['height_feet']):
                        processed_records.append({
                            'id': row['id'],
                            'date': row['date'],
                            'weight': np.nan,
                            'height': row['height_feet']
                        })
                
                print(f"Processed {len(height_data)} height records")
                print(f"Height range: {height_data['height_feet'].min():.2f} - {height_data['height_feet'].max():.2f} feet")
            
            if not processed_records:
                print("No valid height/weight records found")
                return pd.DataFrame()
            
            # Create DataFrame from processed records
            df_processed = pd.DataFrame(processed_records)
            df_processed.set_index('date', inplace=True)
            
            print(f"Total processed height/weight records: {len(df_processed)}")
            print(f"Subjects with height/weight data: {sorted(df_processed['id'].unique())}")

            return df_processed
            
        except Exception as e:
            print(f"Error loading height/weight data: {e}")
            return pd.DataFrame()

    def load_metadata(self, file_path):
        """
        Load and process metadata from IOBP2 roster and screening files.
        Extracts insulin_delivery_device, insulin_delivery_algorithm, insulin_delivery_modality,
        cgm_device, ethnicity, age_of_diagnosis, is_pregnant following process_iobp2_data.py pattern.
        """
        try:
            # Construct paths to roster and screening files
            roster_path = os.path.join(file_path, "IOBP2 RCT Public Dataset", "Data Tables", "IOBP2PtRoster.txt")
            screening_path = os.path.join(file_path, "IOBP2 RCT Public Dataset", "Data Tables", "IOBP2DiabScreening.txt")
            
            # Check if files exist
            if not os.path.exists(roster_path):
                print(f"Warning: Roster file not found at: {roster_path}")
                return pd.DataFrame()
            if not os.path.exists(screening_path):
                print(f"Warning: Screening file not found at: {screening_path}")
                return pd.DataFrame()
            
            print("Loading roster and screening data for metadata...")
            
            # Load roster and screening data
            roster = pd.read_csv(roster_path, delimiter='|', low_memory=False)
            screening = pd.read_csv(screening_path, delimiter='|', low_memory=False)
            
            print(f"Roster shape: {roster.shape}")
            print(f"Screening shape: {screening.shape}")
            
            # Merge roster with screening data
            merged_data = roster.merge(screening, on='PtID', how='inner')
            print(f"Merged data shape: {merged_data.shape}")
            
            # Process subject ID to match other data
            merged_data['id'] = 'IOBP2-' + merged_data['PtID'].astype(str)
            
            # Create metadata dataframe
            df_metadata = pd.DataFrame()
            df_metadata['id'] = merged_data['id']
            
            # insulin_delivery_algorithm - Map based on treatment group and device
            def map_insulin_delivery_algorithm(trt_group, device, is_mdi):
                if trt_group in ['BP', 'BPFiasp']:
                    return 'Bionic Pancreas'
                else:
                    if 'Control:IQ' in str(device):
                        return 'Control:IQ'
                    elif 'Basal:IQ' in str(device):
                        return 'Basal:IQ'
                    elif '630G' in str(device):
                        return 'SmartGuard'
                    elif '670G' in str(device):
                        return 'SmartGuard'
                    elif '530G' in str(device):
                        return 'Low-Glucose Suspend'
                    elif 'Other' in str(device):
                        return np.nan
                    elif pd.isna(device):
                        if is_mdi == 1:
                            return 'Multiple Daily Injections'
                        else:
                            return np.nan
                    else:
                        return 'basal-bolus'
            
            df_metadata['insulin_delivery_algorithm'] = merged_data.apply(
                lambda x: map_insulin_delivery_algorithm(x['TrtGroup'], x['PumpType'], x['InsModInjections']), axis=1
            )
            
            # insulin_delivery_device - Map pump types  
            def map_insulin_delivery_device(trt_group, pump_type, is_mdi):
                if trt_group in ['BP', 'BPFiasp']:
                    return 'Beta Bionics Gen 4 iLet'
                
                if pd.isna(pump_type):
                    if is_mdi == 1:
                        return 'Insulin Pen'
                    else:
                        return np.nan
                
                pump_str = str(pump_type).strip()
                
                if 'OmniPod' in pump_str:
                    return 'OmniPod'
                elif 'Other' in pump_str:
                    return np.nan
                elif 'Tandem' in pump_str:
                    return 't:slim X2'
                elif 'Medtronic' in pump_str:
                    return pump_str.replace('Medtronic', 'MiniMed')
                else:
                    return pump_str
            
            df_metadata['insulin_delivery_device'] = merged_data.apply(
                lambda x: map_insulin_delivery_device(x['TrtGroup'], x['PumpType'], x['InsModInjections']), axis=1
            )
            
            # insulin_delivery_modality - Based on algorithm
            def map_insulin_delivery_modality(algorithm):
                if algorithm == 'Bionic Pancreas':
                    return 'AID'  # Automated Insulin Delivery
                elif algorithm in ['Multiple Daily Injections']:
                    return 'MDI'
                elif algorithm in ['basal-bolus']:
                    return 'SAP'
                else:
                    return 'AID'
            
            df_metadata['insulin_delivery_modality'] = df_metadata['insulin_delivery_algorithm'].apply(map_insulin_delivery_modality)
            
            # cgm_device - All IOBP2 participants used Dexcom G6
            df_metadata['cgm_device'] = 'Dexcom G6'
            
            # ethnicity - Combine ethnicity and race
            def combine_ethnicity_race(row):
                ethnicity = str(row['Ethnicity']) if pd.notna(row['Ethnicity']) else ''
                race = str(row['Race']) if pd.notna(row['Race']) else ''
                
                if ethnicity == 'Hispanic or Latino':
                    if race == 'Unknown/not reported':
                        return 'Hispanic/Latino'
                    else:
                        return race + ', Hispanic/Latino'
                elif race == 'White':
                    return 'White'
                elif race == 'Black/African American':
                    return 'Black/African American'
                elif race == 'Asian':
                    return 'Asian'
                elif race == 'More than one race':
                    return 'More than one race'
                elif race == 'Unknown/not reported':
                    return np.nan
                else:
                    return race if race else np.nan
            
            df_metadata['ethnicity'] = merged_data.apply(combine_ethnicity_race, axis=1)
            
            # age_of_diagnosis - From DiagAge column
            df_metadata['age_of_diagnosis'] = pd.to_numeric(merged_data['DiagAge'], errors='coerce')
            
            # is_pregnant - Pregnancy is exclusion criterion for IOBP2
            df_metadata['is_pregnant'] = False
            
            # Remove any rows with missing subject ID
            df_metadata = df_metadata[df_metadata['id'].notna()]
            
            print(f"Processed metadata for {len(df_metadata)} subjects")
            print(f"Subjects: {sorted(df_metadata['id'].unique())}")
            
            # Show distributions
            print("Metadata distributions:")
            print(f"  Insulin delivery algorithms: {df_metadata['insulin_delivery_algorithm'].value_counts(dropna=False).head().to_dict()}")
            print(f"  Insulin delivery devices: {df_metadata['insulin_delivery_device'].value_counts(dropna=False).head().to_dict()}")
            print(f"  Insulin delivery modalities: {df_metadata['insulin_delivery_modality'].value_counts(dropna=False).to_dict()}")
            print(f"  Ethnicity: {df_metadata['ethnicity'].value_counts(dropna=False).head().to_dict()}")
            
            return df_metadata
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return pd.DataFrame()

    def resample_data(self, df_glucose, df_meals, df_bolus, df_basal, df_exercise, df_meal_labels=None, df_age=None, df_gender=None, df_height_weight=None, df_metadata=None):
        """
        Resample and merge all dataframes into a unified time grid.
        """
        if df_glucose.empty:
            print("No glucose data to resample")
            return pd.DataFrame()
        
        print("Resampling data to 5-minute intervals...")
        
        processed_dfs = []
        
        # Process each subject separately
        for subject_id in df_glucose['id'].unique():
            print(f"Processing subject {subject_id}")
            
            # Get subject's glucose data
            df_subject_glucose = df_glucose[df_glucose['id'] == subject_id].copy()
            
            if df_subject_glucose.empty:
                continue
                
            # Sort by datetime to ensure proper processing
            df_subject_glucose = df_subject_glucose.sort_index()
            
            # Resample glucose to 5-minute intervals
            df_subject = df_subject_glucose[['CGM']].resample('5min', label='right').mean()
            
            # Process bolus data for this subject if available
            if not df_bolus.empty:
                df_subject_bolus = df_bolus[df_bolus['id'] == subject_id].copy()
                
                if not df_subject_bolus.empty:
                    # Sort by datetime 
                    df_subject_bolus = df_subject_bolus.sort_index()
                    
                    # Calculate cumulative sum of bolus over 5-minute rolling windows
                    df_subject_bolus['bolus_cumsum'] = df_subject_bolus['bolus'].rolling(
                        window='5min', 
                        min_periods=1
                    ).sum()
                    
                    # Resample bolus to 5-minute intervals, using last value (cumulative sum)
                    df_subject_bolus_resampled = df_subject_bolus[['bolus_cumsum']].resample('5min', label='right').last()
                    df_subject_bolus_resampled.rename(columns={'bolus_cumsum': 'bolus'}, inplace=True)
                    
                    # Merge glucose and bolus data
                    df_subject = pd.merge(df_subject, df_subject_bolus_resampled, left_index=True, right_index=True, how='outer')
                else:
                    df_subject['bolus'] = np.nan
            else:
                df_subject['bolus'] = np.nan
            
            # Process basal data for this subject if available
            if not df_basal.empty:
                df_subject_basal = df_basal[df_basal['id'] == subject_id].copy()
                
                if not df_subject_basal.empty:
                    # Sort by datetime 
                    df_subject_basal = df_subject_basal.sort_index()
                    
                    # Calculate cumulative sum of basal over 5-minute rolling windows
                    df_subject_basal['basal_cumsum'] = df_subject_basal['basal'].rolling(
                        window='5min', 
                        min_periods=1
                    ).sum()
                    
                    # Resample basal to 5-minute intervals, using last value (cumulative sum)
                    df_subject_basal_resampled = df_subject_basal[['basal_cumsum']].resample('5min', label='right').last()
                    df_subject_basal_resampled.rename(columns={'basal_cumsum': 'basal'}, inplace=True)
                    
                    # Merge with existing data
                    df_subject = pd.merge(df_subject, df_subject_basal_resampled, left_index=True, right_index=True, how='outer')
                else:
                    df_subject['basal'] = np.nan
            else:
                df_subject['basal'] = np.nan
            
            # Process meal label data for this subject if available
            if df_meal_labels is not None and not df_meal_labels.empty:
                df_subject_meal_labels = df_meal_labels[df_meal_labels['id'] == subject_id].copy()
                
                if not df_subject_meal_labels.empty:
                    # Sort by datetime 
                    df_subject_meal_labels = df_subject_meal_labels.sort_index()
                    
                    # For meal labels, check if there was any meal in the previous 5 minutes
                    # Create a numeric indicator first, then apply rolling window
                    df_subject_meal_labels['meal_indicator'] = (df_subject_meal_labels['meal_label'] == 'Meal').astype(int)
                    
                    # Use rolling window to detect if any meal occurred in the window
                    df_subject_meal_labels['meal_sum'] = df_subject_meal_labels['meal_indicator'].rolling(
                        window='5min', 
                        min_periods=1
                    ).sum()
                    
                    # Convert back to text labels
                    df_subject_meal_labels['meal_in_window'] = np.nan
                    df_subject_meal_labels.loc[df_subject_meal_labels['meal_sum'] > 0, 'meal_in_window'] = 'Meal'
                    
                    # Resample meal labels to 5-minute intervals, using last value
                    df_subject_meal_resampled = df_subject_meal_labels[['meal_in_window']].resample('5min', label='right').last()
                    df_subject_meal_resampled.rename(columns={'meal_in_window': 'meal_label'}, inplace=True)
                    
                    # Merge with existing data
                    df_subject = pd.merge(df_subject, df_subject_meal_resampled, left_index=True, right_index=True, how='outer')
                else:
                    df_subject['meal_label'] = np.nan
            else:
                df_subject['meal_label'] = np.nan
            
            # Add subject ID back
            df_subject['id'] = subject_id
            
            # Add age data for this subject if available (block sparse - constant per subject)
            if df_age is not None and not df_age.empty:
                subject_age_data = df_age[df_age['id'] == subject_id]
                if not subject_age_data.empty:
                    # Age is constant for the subject, so use the first (and only) value
                    age_value = subject_age_data['age'].iloc[0]
                    df_subject['age'] = age_value
                else:
                    df_subject['age'] = np.nan
            else:
                df_subject['age'] = np.nan
            
            # Add gender data for this subject if available (block sparse - constant per subject)
            if df_gender is not None and not df_gender.empty:
                subject_gender_data = df_gender[df_gender['id'] == subject_id]
                if not subject_gender_data.empty:
                    # Gender is constant for the subject, so use the first (and only) value
                    gender_value = subject_gender_data['gender'].iloc[0]
                    df_subject['gender'] = gender_value
                else:
                    df_subject['gender'] = np.nan
            else:
                df_subject['gender'] = np.nan
            
            # Add height/weight data for this subject if available (time-based with forward fill)
            if df_height_weight is not None and not df_height_weight.empty:
                subject_height_weight_data = df_height_weight[df_height_weight['id'] == subject_id].copy()
                
                if not subject_height_weight_data.empty:
                    # Sort by datetime to ensure proper forward fill
                    subject_height_weight_data = subject_height_weight_data.sort_index()
                    
                    # Handle duplicates by grouping by index and taking the last non-null value
                    subject_height_weight_data = subject_height_weight_data.groupby(level=0).last()
                    
                    # Resample height and weight separately to 5-minute intervals
                    height_data = subject_height_weight_data[['height']].dropna()
                    weight_data = subject_height_weight_data[['weight']].dropna()
                    
                    # Only resample if we have data
                    if not height_data.empty:
                        height_resampled = height_data.resample('5min').asfreq()
                        height_resampled['height'] = height_resampled['height'].ffill()
                        df_subject = pd.merge(df_subject, height_resampled, left_index=True, right_index=True, how='left')
                        df_subject['height'] = df_subject['height'].ffill()
                    else:
                        df_subject['height'] = np.nan
                    
                    if not weight_data.empty:
                        weight_resampled = weight_data.resample('5min').asfreq()
                        weight_resampled['weight'] = weight_resampled['weight'].ffill()
                        df_subject = pd.merge(df_subject, weight_resampled, left_index=True, right_index=True, how='left')
                        df_subject['weight'] = df_subject['weight'].ffill()
                    else:
                        df_subject['weight'] = np.nan
                else:
                    df_subject['height'] = np.nan
                    df_subject['weight'] = np.nan
            else:
                df_subject['height'] = np.nan
                df_subject['weight'] = np.nan
            
            # Add metadata for this subject if available (block sparse - constant per subject)
            if df_metadata is not None and not df_metadata.empty:
                subject_metadata = df_metadata[df_metadata['id'] == subject_id]
                if not subject_metadata.empty:
                    # Metadata is constant for the subject, so use the first (and only) values
                    metadata_row = subject_metadata.iloc[0]
                    df_subject['insulin_delivery_device'] = metadata_row['insulin_delivery_device']
                    df_subject['insulin_delivery_algorithm'] = metadata_row['insulin_delivery_algorithm']  
                    df_subject['insulin_delivery_modality'] = metadata_row['insulin_delivery_modality']
                    df_subject['cgm_device'] = metadata_row['cgm_device']
                    df_subject['ethnicity'] = metadata_row['ethnicity']
                    df_subject['age_of_diagnosis'] = metadata_row['age_of_diagnosis']
                    df_subject['is_pregnant'] = metadata_row['is_pregnant']
                else:
                    df_subject['insulin_delivery_device'] = np.nan
                    df_subject['insulin_delivery_algorithm'] = np.nan
                    df_subject['insulin_delivery_modality'] = np.nan
                    df_subject['cgm_device'] = np.nan
                    df_subject['ethnicity'] = np.nan
                    df_subject['age_of_diagnosis'] = np.nan
                    df_subject['is_pregnant'] = np.nan
            else:
                df_subject['insulin_delivery_device'] = np.nan
                df_subject['insulin_delivery_algorithm'] = np.nan
                df_subject['insulin_delivery_modality'] = np.nan
                df_subject['cgm_device'] = np.nan
                df_subject['ethnicity'] = np.nan
                df_subject['age_of_diagnosis'] = np.nan
                df_subject['is_pregnant'] = np.nan
            
            # For now, add empty columns for other data types (to be implemented)
            df_subject['carbs'] = np.nan
            df_subject['workout_label'] = np.nan
            df_subject['workout_duration'] = np.nan
            df_subject['calories_burned'] = np.nan
            
            # Ensure homogeneous 5-minute intervals
            df_subject = df_subject.resample('5min').asfreq()
            
            # Sort by index
            df_subject.sort_index(inplace=True)
            
            processed_dfs.append(df_subject)
            
            bolus_count = df_subject['bolus'].notna().sum()
            nonzero_bolus = (df_subject['bolus'] > 0).sum() if bolus_count > 0 else 0
            
            basal_count = df_subject['basal'].notna().sum()
            nonzero_basal = (df_subject['basal'] > 0).sum() if basal_count > 0 else 0
            
            meal_count = (df_subject['meal_label'] == 'Meal').sum()
            age_value = df_subject['age'].iloc[0] if df_subject['age'].notna().any() else "N/A"
            gender_value = df_subject['gender'].iloc[0] if df_subject['gender'].notna().any() else "N/A"
            
            # Height/weight statistics
            height_values = df_subject['height'].dropna()
            weight_values = df_subject['weight'].dropna()
            height_info = f"{height_values.iloc[0]:.2f}ft" if len(height_values) > 0 else "N/A"
            weight_info = f"{weight_values.iloc[0]:.1f}lbs" if len(weight_values) > 0 else "N/A"
            
            # Check if height/weight change over time
            if len(height_values) > 1 and height_values.nunique() > 1:
                height_info += f" (varies: {height_values.min():.2f}-{height_values.max():.2f}ft)"
            if len(weight_values) > 1 and weight_values.nunique() > 1:
                weight_info += f" (varies: {weight_values.min():.1f}-{weight_values.max():.1f}lbs)"
            
            # Metadata statistics
            device_value = df_subject['insulin_delivery_device'].iloc[0] if df_subject['insulin_delivery_device'].notna().any() else "N/A"
            algorithm_value = df_subject['insulin_delivery_algorithm'].iloc[0] if df_subject['insulin_delivery_algorithm'].notna().any() else "N/A"
            ethnicity_value = df_subject['ethnicity'].iloc[0] if df_subject['ethnicity'].notna().any() else "N/A"
            
            print(f"Subject {subject_id}: {df_subject.shape[0]} time points, "
                  f"{df_subject['CGM'].notna().sum()} glucose readings, "
                  f"{nonzero_bolus} bolus events, "
                  f"{nonzero_basal} basal events, "
                  f"{meal_count} meal events, "
                  f"age: {age_value}, gender: {gender_value}, "
                  f"height: {height_info}, weight: {weight_info}, "
                  f"device: {device_value}, algorithm: {algorithm_value}")
        
        if not processed_dfs:
            print("No subjects processed")
            return pd.DataFrame()
        
        # Combine all subjects
        df_final = pd.concat(processed_dfs)
        
        # Summary statistics
        total_glucose = df_final['CGM'].notna().sum()
        total_bolus_events = (df_final['bolus'] > 0).sum() if 'bolus' in df_final.columns else 0
        total_basal_events = (df_final['basal'] > 0).sum() if 'basal' in df_final.columns else 0
        total_meal_events = (df_final['meal_label'] == 'Meal').sum() if 'meal_label' in df_final.columns else 0
        subjects_with_age = df_final['age'].notna().sum() > 0 if 'age' in df_final.columns else False
        subjects_with_gender = df_final['gender'].notna().sum() > 0 if 'gender' in df_final.columns else False
        subjects_with_height = df_final['height'].notna().sum() > 0 if 'height' in df_final.columns else False
        subjects_with_weight = df_final['weight'].notna().sum() > 0 if 'weight' in df_final.columns else False
        
        print(f"Final resampled data: {df_final.shape[0]} total time points for {len(processed_dfs)} subjects")
        print(f"Total glucose readings: {total_glucose}")
        print(f"Total bolus events: {total_bolus_events}")
        print(f"Total basal events: {total_basal_events}")
        print(f"Total meal events: {total_meal_events}")
        if subjects_with_age:
            age_stats = df_final.groupby('id')['age'].first().describe()
            print(f"Age statistics: mean={age_stats['mean']:.1f}, std={age_stats['std']:.1f}, range={age_stats['min']:.0f}-{age_stats['max']:.0f}")
        if subjects_with_gender:
            gender_dist = df_final.groupby('id')['gender'].first().value_counts()
            print(f"Gender distribution: {gender_dist.to_dict()}")
        if subjects_with_height:
            height_stats = df_final.groupby('id')['height'].first().describe()
            print(f"Height statistics: mean={height_stats['mean']:.2f}ft, std={height_stats['std']:.2f}ft, range={height_stats['min']:.2f}-{height_stats['max']:.2f}ft")
        if subjects_with_weight:
            weight_stats = df_final.groupby('id')['weight'].first().describe()
            print(f"Weight statistics: mean={weight_stats['mean']:.1f}lbs, std={weight_stats['std']:.1f}lbs, range={weight_stats['min']:.1f}-{weight_stats['max']:.1f}lbs")
        
        # Metadata statistics
        if 'insulin_delivery_device' in df_final.columns and df_final['insulin_delivery_device'].notna().sum() > 0:
            device_dist = df_final.groupby('id')['insulin_delivery_device'].first().value_counts()
            print(f"Insulin delivery devices: {device_dist.to_dict()}")
        if 'insulin_delivery_algorithm' in df_final.columns and df_final['insulin_delivery_algorithm'].notna().sum() > 0:
            algorithm_dist = df_final.groupby('id')['insulin_delivery_algorithm'].first().value_counts()
            print(f"Insulin delivery algorithms: {algorithm_dist.to_dict()}")
        if 'insulin_delivery_modality' in df_final.columns and df_final['insulin_delivery_modality'].notna().sum() > 0:
            modality_dist = df_final.groupby('id')['insulin_delivery_modality'].first().value_counts()
            print(f"Insulin delivery modalities: {modality_dist.to_dict()}")
        if 'ethnicity' in df_final.columns and df_final['ethnicity'].notna().sum() > 0:
            ethnicity_dist = df_final.groupby('id')['ethnicity'].first().value_counts()
            print(f"Ethnicity distribution: {ethnicity_dist.head().to_dict()}")
        if 'age_of_diagnosis' in df_final.columns and df_final['age_of_diagnosis'].notna().sum() > 0:
            diag_age_stats = df_final.groupby('id')['age_of_diagnosis'].first().describe()
            print(f"Age of diagnosis statistics: mean={diag_age_stats['mean']:.1f}, std={diag_age_stats['std']:.1f}, range={diag_age_stats['min']:.0f}-{diag_age_stats['max']:.0f}")
        
        return df_final

    def get_dataframes(self, file_path):
        """
        Extract the data from the dataset and process them into separate dataframes.
        """
        # TODO: Implement data extraction
        # This should:
        # 1. Read raw data files from file_path
        # 2. Parse different data types (glucose, meals, insulin, exercise)
        # 3. Convert to appropriate formats
        # 4. Set datetime indices
        # 5. Clean and validate data
        
        # Placeholder - return empty dataframes
        df_glucose = pd.DataFrame()
        df_meals = pd.DataFrame() 
        df_bolus = pd.DataFrame()
        df_basal = pd.DataFrame()
        df_exercise = pd.DataFrame()
        
        print("Glucose data processed")
        print("Meal data processed") 
        print("Bolus data processed")
        print("Basal data processed")
        print("Exercise data processed")
        
        return df_glucose, df_meals, df_bolus, df_basal, df_exercise

    def add_metadata_columns(self, df, file_path):
        """
        Add demographics and metadata columns to the processed dataframe.
        """
        # TODO: Implement metadata extraction
        # This should:
        # 1. Extract patient demographics (age, gender, etc.)
        # 2. Add device information (CGM type, insulin delivery method)
        # 3. Add clinical metadata (diagnosis age, ethnicity, etc.)
        # 4. Merge with main dataframe
        
        # Placeholder implementation
        unique_subjects = df['id'].unique() if 'id' in df.columns else []
        
        print(f"Added metadata columns for {len(unique_subjects)} subjects")
        
        return df


def main():
    """
    Main function to run IOBP2 parser as a standalone script.
    """
    print("=== IOBP2 Dataset Parser ===")
    
    # File paths
    input_path = "data/raw/"
    output_path = "data/processed/IOBP2.csv"
    
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        print("Please ensure the IOBP2 dataset is available in the data/raw/iobp2 directory")
        return
    
    try:
        # Initialize parser
        parser = Parser()
        
        # Process the data
        print(f"Loading data from: {input_path}")
        df_processed = parser(input_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed data
        if not df_processed.empty:
            df_processed.to_csv(output_path, index=True)  # Save with datetime index
            print(f"Processed data saved to: {output_path}")
            print(f"Dataset shape: {df_processed.shape}")
            print(f"Columns: {list(df_processed.columns)}")
            
            # Show sample data
            if len(df_processed) > 0:
                print("\nSample data:")
                print(df_processed.head())
        else:
            print("Warning: No data was processed. Please check the implementation.")
            
    except Exception as e:
        print(f"Error processing IOBP2 data: {e}")
        raise


if __name__ == "__main__":
    main()
"""
DiaTrend parser for processing DiaTrend dataset.

The DiaTrend parser processes DiaTrend data and returns merged data on the same time grid.
"""

from .base_parser import BaseParser
import pandas as pd
import numpy as np
import os


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the DiaTrend dataset root folder.
        """
        print(f"Processing DiaTrend data from {file_path}")
        
        # Load data from different sources
        df_glucose = self.load_cgm_data(file_path)
        df_bolus_and_carbs = self.load_bolus_and_carbs_data(file_path)
        df_basal = self.load_basal_data(file_path)
        df_demographics = self.load_demographics_data(file_path)

        # Resample and merge data
        df_resampled = self.resample_data(df_glucose, df_bolus_and_carbs, df_basal, df_demographics)

        return df_resampled

    def load_cgm_data(self, file_path):
        """
        Load and process CGM data from DiaTrend Excel files.
        Only includes subjects with CGM, Bolus, and Basal sheets.
        """
        print("Loading CGM data...")
        
        try:
            # Define path to DiaTrend folder
            diatrend_path = os.path.join(file_path, "DiaTrend")
            
            if not os.path.exists(diatrend_path):
                print(f"DiaTrend folder not found at: {diatrend_path}")
                return pd.DataFrame()
            
            # Define required sheets
            required_sheets = {'CGM', 'Bolus', 'Basal'}
            
            # Find valid subjects
            valid_subjects = []
            for i in range(1, 56):  # Check subjects 1-55
                subject_file = os.path.join(diatrend_path, f"Subject{i}.xlsx")
                if os.path.exists(subject_file):
                    try:
                        xl_file = pd.ExcelFile(subject_file)
                        if required_sheets.issubset(set(xl_file.sheet_names)):
                            valid_subjects.append(i)
                    except Exception as e:
                        print(f"Warning: Could not read {subject_file}: {e}")
                        continue
            
            print(f"Found {len(valid_subjects)} valid subjects: {valid_subjects}")
            
            if not valid_subjects:
                print("No valid subjects found")
                return pd.DataFrame()
            
            # Load CGM data from valid subjects
            cgm_data_list = []
            
            for subject_id in valid_subjects:
                try:
                    subject_file = os.path.join(diatrend_path, f"Subject{subject_id}.xlsx")
                    
                    # Read CGM sheet
                    cgm_df = pd.read_excel(subject_file, sheet_name='CGM')
                    
                    # Process data
                    if not cgm_df.empty and 'date' in cgm_df.columns and 'mg/dl' in cgm_df.columns:
                        # Add subject ID
                        cgm_df['id'] = f"Subject{subject_id}"
                        
                        # Rename glucose column to standard name
                        cgm_df = cgm_df.rename(columns={'mg/dl': 'CGM'})
                        
                        # Set date as index
                        cgm_df['date'] = pd.to_datetime(cgm_df['date'])
                        cgm_df.set_index('date', inplace=True)
                        
                        # Keep only necessary columns
                        cgm_df = cgm_df[['id', 'CGM']]
                        
                        # Remove invalid glucose values
                        cgm_df = cgm_df[cgm_df['CGM'].notna()]
                        
                        cgm_data_list.append(cgm_df)
                        print(f"  Subject {subject_id}: {len(cgm_df)} CGM readings")
                    
                except Exception as e:
                    print(f"Warning: Error processing Subject{subject_id}: {e}")
                    continue
            
            if not cgm_data_list:
                print("No CGM data could be loaded")
                return pd.DataFrame()
            
            # Combine all subjects
            df_combined = pd.concat(cgm_data_list)
            
            print(f"Loaded CGM data: {len(df_combined)} total readings from {len(valid_subjects)} subjects")
            print(f"Glucose range: {df_combined['CGM'].min():.1f} - {df_combined['CGM'].max():.1f} mg/dL")
            print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
            
            return df_combined
            
        except Exception as e:
            print(f"Error loading CGM data: {e}")
            return pd.DataFrame()

    def load_bolus_and_carbs_data(self, file_path):
        """
        Load and process bolus insulin data from DiaTrend Excel files.
        Only includes subjects with CGM, Bolus, and Basal sheets.
        """
        print("Loading bolus data...")
        
        try:
            # Define path to DiaTrend folder
            diatrend_path = os.path.join(file_path, "DiaTrend")
            
            if not os.path.exists(diatrend_path):
                print(f"DiaTrend folder not found at: {diatrend_path}")
                return pd.DataFrame()
            
            # Define required sheets
            required_sheets = {'CGM', 'Bolus', 'Basal'}
            
            # Find valid subjects (same logic as CGM)
            valid_subjects = []
            for i in range(1, 56):  # Check subjects 1-55
                subject_file = os.path.join(diatrend_path, f"Subject{i}.xlsx")
                if os.path.exists(subject_file):
                    try:
                        xl_file = pd.ExcelFile(subject_file)
                        if required_sheets.issubset(set(xl_file.sheet_names)):
                            valid_subjects.append(i)
                    except Exception as e:
                        print(f"Warning: Could not read {subject_file}: {e}")
                        continue
            
            print(f"Processing bolus data for {len(valid_subjects)} valid subjects")
            
            if not valid_subjects:
                print("No valid subjects found")
                return pd.DataFrame()
            
            # Load bolus data from valid subjects
            bolus_data_list = []
            
            for subject_id in valid_subjects:
                try:
                    subject_file = os.path.join(diatrend_path, f"Subject{subject_id}.xlsx")
                    
                    # Read Bolus sheet
                    bolus_df = pd.read_excel(subject_file, sheet_name='Bolus')
                    
                    # Process data
                    if not bolus_df.empty and 'date' in bolus_df.columns and 'normal' in bolus_df.columns:
                        # Add subject ID
                        bolus_df['id'] = f"Subject{subject_id}"
                        
                        # Rename normal column to standard name
                        bolus_df = bolus_df.rename(columns={'normal': 'bolus'})
                        
                        # Process carbs column if available
                        if 'carbInput' in bolus_df.columns:
                            bolus_df = bolus_df.rename(columns={'carbInput': 'carbs'})
                            # Fill NaN carbs with 0 and ensure non-negative
                            bolus_df['carbs'] = bolus_df['carbs'].fillna(0.0)
                            bolus_df['carbs'] = bolus_df['carbs'].clip(lower=0.0)
                        else:
                            # No carbs column, add empty carbs
                            bolus_df['carbs'] = 0.0
                        
                        # Set date as index
                        bolus_df['date'] = pd.to_datetime(bolus_df['date'])
                        bolus_df.set_index('date', inplace=True)
                        
                        # Keep only necessary columns
                        bolus_df = bolus_df[['id', 'bolus', 'carbs']]
                        
                        # Remove invalid bolus values and keep only non-zero bolus
                        bolus_df = bolus_df[bolus_df['bolus'].notna()]
                        bolus_df = bolus_df[bolus_df['bolus'] > 0]  # Only positive bolus values
                        
                        bolus_data_list.append(bolus_df)
                        
                        # Enhanced logging with carb information
                        total_carbs = bolus_df['carbs'].sum()
                        carb_events = (bolus_df['carbs'] > 0).sum()
                        print(f"  Subject {subject_id}: {len(bolus_df)} bolus events, "
                              f"{carb_events} with carbs (total: {total_carbs:.1f}g)")
                    
                except Exception as e:
                    print(f"Warning: Error processing Subject{subject_id}: {e}")
                    continue
            
            if not bolus_data_list:
                print("No bolus data could be loaded")
                return pd.DataFrame()
            
            # Combine all subjects
            df_combined = pd.concat(bolus_data_list)
            
            print(f"Loaded bolus data: {len(df_combined)} total bolus events from {len(valid_subjects)} subjects")
            print(f"Bolus range: {df_combined['bolus'].min():.1f} - {df_combined['bolus'].max():.1f} units")
            print(f"Carbs range: {df_combined['carbs'].min():.1f} - {df_combined['carbs'].max():.1f} grams")
            print(f"Total carbs: {df_combined['carbs'].sum():.1f}g, events with carbs: {(df_combined['carbs'] > 0).sum()}")
            print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
            
            return df_combined
            
        except Exception as e:
            print(f"Error loading bolus data: {e}")
            return pd.DataFrame()

    def load_basal_data(self, file_path):
        """
        Load and process basal insulin data from DiaTrend Excel files.
        Basal rate (U/h) * duration (ms converted to hours) = total units.
        Distributes basal units evenly across the duration at 5-minute intervals.
        """
        print("Loading basal data...")
        
        try:
            # Define path to DiaTrend folder
            diatrend_path = os.path.join(file_path, "DiaTrend")
            
            if not os.path.exists(diatrend_path):
                print(f"DiaTrend folder not found at: {diatrend_path}")
                return pd.DataFrame()
            
            # Define required sheets
            required_sheets = {'CGM', 'Bolus', 'Basal'}
            
            # Find valid subjects (same logic as CGM/bolus)
            valid_subjects = []
            for i in range(1, 56):  # Check subjects 1-55
                subject_file = os.path.join(diatrend_path, f"Subject{i}.xlsx")
                if os.path.exists(subject_file):
                    try:
                        xl_file = pd.ExcelFile(subject_file)
                        if required_sheets.issubset(set(xl_file.sheet_names)):
                            valid_subjects.append(i)
                    except Exception as e:
                        print(f"Warning: Could not read {subject_file}: {e}")
                        continue
            
            print(f"Processing basal data for {len(valid_subjects)} valid subjects")
            
            if not valid_subjects:
                print("No valid subjects found")
                return pd.DataFrame()
            
            # Load and process basal data from valid subjects
            all_basal_records = []
            
            for subject_id in valid_subjects:
                try:
                    subject_file = os.path.join(diatrend_path, f"Subject{subject_id}.xlsx")
                    
                    # Read Basal sheet
                    basal_df = pd.read_excel(subject_file, sheet_name='Basal')
                    
                    if basal_df.empty or 'date' not in basal_df.columns or 'rate' not in basal_df.columns or 'duration' not in basal_df.columns:
                        print(f"  Subject {subject_id}: Missing required basal columns")
                        continue
                    
                    # Filter out invalid records
                    valid_basal = basal_df[
                        basal_df['rate'].notna() & 
                        basal_df['duration'].notna() & 
                        (basal_df['rate'] > 0) & 
                        (basal_df['duration'] > 0)
                    ].copy()
                    
                    if valid_basal.empty:
                        print(f"  Subject {subject_id}: No valid basal records")
                        continue
                    
                    # Convert duration from ms to hours
                    valid_basal['duration_hours'] = valid_basal['duration'] / (1000 * 60 * 60)
                    
                    # Calculate total basal units for each record
                    valid_basal['total_units'] = valid_basal['rate'] * valid_basal['duration_hours']
                    
                    # Process each basal record to distribute over time
                    subject_basal_records = []
                    
                    for _, record in valid_basal.iterrows():
                        start_time = pd.to_datetime(record['date'])
                        duration_ms = record['duration']
                        total_units = record['total_units']
                        
                        # Calculate end time
                        end_time = start_time + pd.Timedelta(milliseconds=duration_ms)
                        
                        # Create 5-minute intervals for this basal period
                        time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
                        
                        if len(time_range) > 1:  # Need at least 2 points to have intervals
                            # Distribute units evenly across intervals
                            units_per_interval = total_units / (len(time_range) - 1)
                            
                            # Create records for each interval (excluding the last time point)
                            for time_point in time_range[:-1]:
                                subject_basal_records.append({
                                    'date': time_point,
                                    'id': f"Subject{subject_id}",
                                    'basal': units_per_interval
                                })
                    
                    if subject_basal_records:
                        all_basal_records.extend(subject_basal_records)
                        print(f"  Subject {subject_id}: {len(subject_basal_records)} basal intervals, "
                              f"{len(valid_basal)} basal periods, "
                              f"total: {valid_basal['total_units'].sum():.1f} units")
                    
                except Exception as e:
                    print(f"Warning: Error processing Subject{subject_id}: {e}")
                    continue
            
            if not all_basal_records:
                print("No basal data could be processed")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_basal = pd.DataFrame(all_basal_records)
            df_basal['date'] = pd.to_datetime(df_basal['date'])
            df_basal.set_index('date', inplace=True)
            
            # Group by datetime and sum (in case of overlapping intervals)
            df_combined = df_basal.groupby(['date', 'id'])['basal'].sum().reset_index()
            df_combined.set_index('date', inplace=True)
            
            print(f"Loaded basal data: {len(df_combined)} total basal intervals from {len(valid_subjects)} subjects")
            print(f"Basal range: {df_combined['basal'].min():.4f} - {df_combined['basal'].max():.4f} units per 5-min")
            print(f"Total basal units: {df_combined['basal'].sum():.1f}")
            print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
            
            return df_combined
            
        except Exception as e:
            print(f"Error loading basal data: {e}")
            return pd.DataFrame()

    def load_demographics_data(self, file_path):
        """
        Load and process demographics data from SubjectDemographics_Feb2025.xlsx.
        Creates block sparse (constant per subject) columns for insulin_delivery_device and cgm_device.
        """
        print("Loading demographics data...")
        
        try:
            # Define path to demographics file
            demo_file = os.path.join(file_path, "DiaTrend", "SubjectDemographics_Feb2025.xlsx")
            
            if not os.path.exists(demo_file):
                print(f"Demographics file not found at: {demo_file}")
                return pd.DataFrame()
            
            # Read demographics file
            demo_df = pd.read_excel(demo_file)
            
            print(f"Raw demographics data shape: {demo_df.shape}")
            print(f"Columns: {list(demo_df.columns)}")
            
            # Check required columns
            if 'Subject' not in demo_df.columns:
                print("Error: 'Subject' column not found in demographics")
                return pd.DataFrame()
            
            # Process subject IDs to match our format
            demo_df['id'] = 'Subject' + demo_df['Subject'].astype(str)
            
            # Get our valid subjects (those with CGM, Bolus, and Basal data)
            valid_subjects = [29, 30, 31, 36, 37, 38, 39, 42, 45, 46, 47, 49, 50, 51, 52, 53, 54]
            valid_subject_ids = [f"Subject{i}" for i in valid_subjects]
            
            # Filter to only valid subjects
            demo_filtered = demo_df[demo_df['id'].isin(valid_subject_ids)].copy()
            
            print(f"Found demographics for {len(demo_filtered)} valid subjects")
            
            if demo_filtered.empty:
                print("No demographics data for valid subjects")
                return pd.DataFrame()
            
            # Process insulin delivery device
            if 'insulin pump model' in demo_df.columns:
                demo_filtered['insulin_delivery_device'] = demo_filtered['insulin pump model']

                t_slim_pump_name = 't:slim X2'

                # Standardize pump names
                pump_mapping = {
                    # Tandem variants
                    'Tandem Tslim x2': t_slim_pump_name,
                    'Tandem t:slimx2': t_slim_pump_name,
                    'Tandem t-Slim': t_slim_pump_name,
                    'Tandem T-Slim X-2': t_slim_pump_name,
                    'Tslim x2': t_slim_pump_name,
                    'Tandem tslim': t_slim_pump_name,
                    'Tandem tslim x2': t_slim_pump_name,
                    'tandem tslim x2': t_slim_pump_name,
                    'Tandem tslim X-2': t_slim_pump_name,
                    'Tandem tslim X2': t_slim_pump_name,
                    'Tandem t:slim': t_slim_pump_name,
                    'Tandem t slim x2': t_slim_pump_name,
                    'Tandem tslim control IQ': t_slim_pump_name,
                    'Tandem Tslim Control IQ': t_slim_pump_name,
                    'Tandem': t_slim_pump_name,
                    
                    # Omnipod variants
                    'Omnipod Dash': 'OmniPod DASH',
                    'omnipod DASH': 'OmniPod DASH',
                    'Omnipod Eros': 'OmniPod EROS',
                    'Omnipod EROS': 'OmniPod EROS',
                    'Omnipod': 'OmniPod',
                    'Insulet Omnipod': 'OmniPod',
                    
                    # Medtronic variants
                    'Medtronic Minimed 600 series': 'MiniMed 600 series',
                    'Medtronic 670G': 'MiniMed 670G',
                    'Medtronic Minimed 670G': 'MiniMed 670G',
                    'Medtronic Minimed 500 series': 'MiniMed 500 series',
                }
                
                demo_filtered['insulin_delivery_device'] = demo_filtered['insulin_delivery_device'].replace(pump_mapping)
                
                print(f"Insulin delivery devices: {demo_filtered['insulin_delivery_device'].value_counts().to_dict()}")
            else:
                demo_filtered['insulin_delivery_device'] = None
                print("Warning: 'insulin pump model' column not found")
            
            # Process CGM device
            if 'CGM model' in demo_df.columns:
                demo_filtered['cgm_device'] = demo_filtered['CGM model']
                
                # Standardize CGM names
                cgm_mapping = {
                    'Dexcom G6': 'Dexcom G6',
                    'Dexcom G5': 'Dexcom G5', 
                    'Dexcom': 'Dexcom',
                    'Medtronic Guardian': 'Medtronic Guardian',
                    'Medtronic Guardian 3': 'Medtronic Guardian 3',
                    'Medtronic Minimed 670G': 'Medtronic Guardian',  # 670G has integrated Guardian
                    'Medtronic': 'Medtronic Guardian'
                }
                
                demo_filtered['cgm_device'] = demo_filtered['cgm_device'].replace(cgm_mapping)
                
                print(f"CGM devices: {demo_filtered['cgm_device'].value_counts().to_dict()}")
            else:
                demo_filtered['cgm_device'] = None
                print("Warning: 'CGM model' column not found")
            
            # Process ethnicity from Race column
            if 'Race' in demo_df.columns:
                demo_filtered['ethnicity'] = demo_filtered['Race']
                
                # Standardize ethnicity names
                ethnicity_mapping = {
                    'White/Caucasian': 'White'
                }
                
                demo_filtered['ethnicity'] = demo_filtered['ethnicity'].replace(ethnicity_mapping)
                
                print(f"Ethnicity: {demo_filtered['ethnicity'].value_counts().to_dict()}")
            else:
                demo_filtered['ethnicity'] = None
                print("Warning: 'Race' column not found")
            
            # Process gender from Gender column
            if 'Gender' in demo_df.columns:
                demo_filtered['gender'] = demo_filtered['Gender']
                print(f"Gender: {demo_filtered['gender'].value_counts().to_dict()}")
            else:
                demo_filtered['gender'] = 'Unknown'
                print("Warning: 'Gender' column not found")
            
            # Process insulin delivery algorithm and modality based on device
            demo_filtered['insulin_delivery_algorithm'] = None
            demo_filtered['insulin_delivery_modality'] = None
            
            # We only know modality and algorithm for MiniMed 670G
            minimed_670g_mask = demo_filtered['insulin_delivery_device'] == 'MiniMed 670G'
            demo_filtered.loc[minimed_670g_mask, 'insulin_delivery_algorithm'] = 'SmartGuard'
            demo_filtered.loc[minimed_670g_mask, 'insulin_delivery_modality'] = 'AID'
            
            print(f"Insulin delivery algorithm: {demo_filtered['insulin_delivery_algorithm'].value_counts(dropna=False).to_dict()}")
            print(f"Insulin delivery modality: {demo_filtered['insulin_delivery_modality'].value_counts(dropna=False).to_dict()}")
            
            # Process age from Age column
            if 'Age' in demo_df.columns:
                # Handle mixed format: numeric ages and age ranges
                def process_age(age_val):
                    if pd.isna(age_val):
                        return None
                    age_str = str(age_val).strip()
                    
                    # Try to convert to numeric directly
                    try:
                        return float(age_str)
                    except ValueError:
                        pass
                    
                    # Handle age ranges by taking midpoint
                    if ' - ' in age_str and 'yrs' in age_str:
                        age_range = age_str.replace(' yrs', '').strip()
                        if ' - ' in age_range:
                            parts = age_range.split(' - ')
                            if len(parts) == 2:
                                try:
                                    min_age = float(parts[0])
                                    max_age = float(parts[1])
                                    return (min_age + max_age) / 2
                                except ValueError:
                                    pass
                    
                    # If we can't parse, return None
                    return None
                
                demo_filtered['age'] = demo_filtered['Age'].apply(process_age)
                
                valid_ages = demo_filtered['age'].dropna()
                if len(valid_ages) > 0:
                    age_min = valid_ages.min()
                    age_max = valid_ages.max()
                    age_mean = valid_ages.mean()
                    print(f"Age range: {age_min:.1f} - {age_max:.1f} years")
                    print(f"Mean age: {age_mean:.1f} years")
                else:
                    print("No valid age data found")
            else:
                demo_filtered['age'] = None
                print("Warning: 'Age' column not found")
            
            # Keep only necessary columns
            result_df = demo_filtered[['id', 'insulin_delivery_device', 'cgm_device', 'ethnicity', 'gender', 'age', 'insulin_delivery_algorithm', 'insulin_delivery_modality']].copy()
            
            print(f"Processed demographics for {len(result_df)} subjects")
            
            return result_df
            
        except Exception as e:
            print(f"Error loading demographics data: {e}")
            return pd.DataFrame()

    def resample_data(self, df_glucose, df_bolus_and_carbs, df_basal, df_demographics):
        """
        Resample and merge all dataframes into a unified time grid.
        CGM values are resampled to 5-minute intervals using mean.
        Bolus doses are summed for the previous 5-minute interval.
        Demographics data is added as block sparse (constant per subject).
        """
        print("Resampling data to 5-minute intervals...")
        
        if df_glucose.empty:
            print("No glucose data to resample")
            return pd.DataFrame()
        
        # Get unique subjects
        subjects = df_glucose['id'].unique()
        print(f"Processing {len(subjects)} subjects")
        
        processed_dfs = []
        
        for subject_id in subjects:
            print(f"  Processing {subject_id}...")
            
            # Get subject's glucose data
            df_subject_glucose = df_glucose[df_glucose['id'] == subject_id].copy()
            
            if df_subject_glucose.empty:
                print(f"    No glucose data for {subject_id}")
                continue
            
            # Resample glucose to 5-minute intervals using mean
            df_subject = df_subject_glucose[['CGM']].resample('5min', label='right').mean()
            
            # Add subject ID back
            df_subject['id'] = subject_id
            
            # Process bolus data for this subject if available
            if not df_bolus_and_carbs.empty:
                df_subject_bolus = df_bolus_and_carbs[df_bolus_and_carbs['id'] == subject_id].copy()
                
                if not df_subject_bolus.empty:
                    # Sum bolus and carb doses for each 5-minute interval
                    # Use label='right' to match glucose resampling
                    df_bolus_resampled = df_subject_bolus[['bolus', 'carbs']].resample('5min', label='right').sum()
                    
                    # Merge with glucose data
                    df_subject = pd.merge(df_subject, df_bolus_resampled, left_index=True, right_index=True, how='outer')
                else:
                    # No bolus data for this subject, add empty bolus and carbs columns
                    df_subject['bolus'] = np.nan
                    df_subject['carbs'] = np.nan
            else:
                # No bolus data at all, add empty bolus and carbs columns
                df_subject['bolus'] = np.nan
                df_subject['carbs'] = np.nan

            # Process basal data for this subject if available
            if not df_basal.empty:
                df_subject_basal = df_basal[df_basal['id'] == subject_id].copy()
                
                if not df_subject_basal.empty:
                    # Sum basal doses for each 5-minute interval (already distributed)
                    # Use label='right' to match glucose resampling
                    df_basal_resampled = df_subject_basal[['basal']].resample('5min', label='right').sum()
                    
                    # Merge with existing data
                    df_subject = pd.merge(df_subject, df_basal_resampled, left_index=True, right_index=True, how='outer')
                else:
                    # No basal data for this subject, add empty basal column
                    df_subject['basal'] = np.nan
            else:
                # No basal data at all, add empty basal column
                df_subject['basal'] = np.nan

            # Add demographics data as block sparse (constant per subject)
            if not df_demographics.empty:
                subject_demo = df_demographics[df_demographics['id'] == subject_id]
                if not subject_demo.empty:
                    # Add demographics as constant columns for this subject
                    demo_record = subject_demo.iloc[0]
                    df_subject['insulin_delivery_device'] = demo_record['insulin_delivery_device']
                    df_subject['cgm_device'] = demo_record['cgm_device']
                    df_subject['ethnicity'] = demo_record['ethnicity']
                    df_subject['gender'] = demo_record['gender']
                    df_subject['age'] = demo_record['age']
                    df_subject['insulin_delivery_algorithm'] = demo_record['insulin_delivery_algorithm']
                    df_subject['insulin_delivery_modality'] = demo_record['insulin_delivery_modality']
                else:
                    # No demographics for this subject
                    df_subject['insulin_delivery_device'] = None
                    df_subject['cgm_device'] = None
                    df_subject['ethnicity'] = None
                    df_subject['gender'] = None
                    df_subject['age'] = None
                    df_subject['insulin_delivery_algorithm'] = None
                    df_subject['insulin_delivery_modality'] = None
            else:
                # No demographics data loaded
                df_subject['insulin_delivery_device'] = None
                df_subject['cgm_device'] = None
                df_subject['ethnicity'] = None
                df_subject['gender'] = None
                df_subject['age'] = None
                df_subject['insulin_delivery_algorithm'] = None
                df_subject['insulin_delivery_modality'] = None
            
            # Keep only subjects with rows with valid glucose, bolus, basal, or carbs data
            valid_rows = (df_subject['CGM'].notna() | 
                         (df_subject['bolus'] > 0) | 
                         (df_subject['basal'] > 0) | 
                         (df_subject['carbs'] > 0))

            if not df_subject[valid_rows].empty:
                df_subject['id'] = subject_id  # Ensure all rows have id after resampling
                processed_dfs.append(df_subject)
                
                # Log statistics
                valid_glucose = df_subject['CGM'].notna().sum()
                total_bolus = df_subject['bolus'].sum()
                bolus_events = (df_subject['bolus'] > 0).sum()
                total_basal = df_subject['basal'].sum()
                basal_intervals = (df_subject['basal'] > 0).sum()
                total_carbs = df_subject['carbs'].sum()
                carb_events = (df_subject['carbs'] > 0).sum()
                
                print(f"    {subject_id}: {len(df_subject)} time points, {valid_glucose} glucose readings, "
                      f"{bolus_events} bolus events (total: {total_bolus:.1f} units), "
                      f"{basal_intervals} basal intervals (total: {total_basal:.1f} units), "
                      f"{carb_events} carb events (total: {total_carbs:.1f}g)")

                # Check for 5-minute intervals
                valid_intervals = (df_subject.index.to_series().diff().dropna() == pd.Timedelta("5min")).all()
                if valid_intervals:
                    print(f"Subject {subject_id} has perfect 5-minute intervals")
                else:
                    print(f"Warning: Subject {subject_id} does not have valid 5-minute intervals!")

        if not processed_dfs:
            print("No subjects processed")
            return pd.DataFrame()
        
        # Combine all subjects
        df_final = pd.concat(processed_dfs)
        
        # Calculate summary statistics
        total_glucose = df_final['CGM'].notna().sum()
        total_bolus_events = (df_final['bolus'] > 0).sum()
        total_bolus_units = df_final['bolus'].sum()
        total_basal_intervals = (df_final['basal'] > 0).sum()
        total_basal_units = df_final['basal'].sum()
        total_carb_events = (df_final['carbs'] > 0).sum()
        total_carb_grams = df_final['carbs'].sum()
        
        print(f"Final resampled data: {len(df_final)} total time points")
        print(f"Total glucose readings: {total_glucose}")
        print(f"Total bolus events: {total_bolus_events} (total: {total_bolus_units:.1f} units)")
        print(f"Total basal intervals: {total_basal_intervals} (total: {total_basal_units:.1f} units)")
        print(f"Total carb events: {total_carb_events} (total: {total_carb_grams:.1f}g)")
        print(f"Date range: {df_final.index.min()} to {df_final.index.max()}")

        df_final['insulin'] = df_final['bolus'].fillna(0) + df_final['basal']
        df_final['source_file'] = 'DiaTrend'
        
        return df_final


def main():
    """
    Main function to run DiaTrend parser as a standalone script.
    """
    print("=== DiaTrend Dataset Parser ===")
    
    # File paths
    input_path = "data/raw/"
    output_path = "data/processed/"
    
    # Initialize parser
    parser = Parser()
    
    # Process data
    try:
        df = parser(input_path)
        
        if df.empty:
            print("No data was processed.")
            return
        
        # Save processed data
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "DiaTrend.csv")
        df.to_csv(output_file, index=True)
        
        print(f"✓ Saved DiaTrend dataframe to: {output_file}")
        print(f"Dataset shape: {df.shape}")
        
    except Exception as e:
        print(f"Error processing DiaTrend data: {e}")
        return None
    
    return df


if __name__ == "__main__":
    df = main()

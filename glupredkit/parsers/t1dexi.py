"""
The T1DEXI parser is processing the .xpt data from the Ohio T1DM datasets and returning the data merged into
the same time grid in a dataframe.

To do:
- correct the dates that are now in the future
- workouts and similar should have a separate column for duration instead
-
"""
from .base_parser import BaseParser
import pandas as pd
import isodate
import xport
import numpy as np
import datetime
import zipfile_deflate64
import zipfile
import tempfile
import os


class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.subject_ids = []

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the T1DEXI dataset root folder.
        """
        print(f"Processing data from {file_path}")
        self.subject_ids = get_valid_subject_ids(file_path)
        (df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, height_dict, weight_dict,
         steps_or_cal_burn_dict) = self.get_dataframes(file_path)
        df_resampled = self.resample_data(df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict,
                                          steps_or_cal_burn_dict)

        # Add demographics and metadata columns
        print("Adding demographics and metadata columns...")
        df_resampled = self.add_metadata_columns(df_resampled, height_dict, weight_dict, file_path)

        return df_resampled

    def resample_data(self, df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict,
                      steps_or_cal_burn_dict):
        # There are 88 subjects on MDI in the dataset --> 502 - 88 = 414. In the youth version: 261 - 37 = 224.
        # Total: 763 with, 638 without MDI
        # We use only the subjects not on MDI in this parser
        processed_dfs = []
        count = 1
        for subject_id in self.subject_ids:
            df_subject = df_glucose[df_glucose['id'] == subject_id].copy()
            df_subject['id'] = df_subject['id'].astype('int')
            df_subject = df_subject.resample('5min', label='right').mean()
            df_subject['id'] = df_subject['id'].astype('object')
            df_subject.sort_index(inplace=True)

            df_subject_meals = df_meals[df_meals['id'] == subject_id].copy()
            if not df_subject_meals.empty:
                df_subject_meal_name = df_subject_meals[df_subject_meals['meal_label'].notna()][['meal_label']].resample('5min', label='right').agg(
                    lambda x: ', '.join(x))
                df_subject_carbs = df_subject_meals[df_subject_meals['carbs'].notna()][['carbs']].resample('5min', label='right').sum()

                df_subject = pd.merge(df_subject, df_subject_meal_name, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_carbs, on="date", how='outer')

                # Fill NaN where numbers were turned to 0 or strings were turned to ''
                df_subject['meal_label'] = df_subject['meal_label'].replace('', np.nan)
                df_subject['carbs'] = df_subject['carbs'].infer_objects().replace(0, np.nan)
            else:
                df_subject['meal_label'] = np.nan
                df_subject['carbs'] = np.nan

            df_subject_bolus = df_bolus[df_bolus['id'] == subject_id].copy()
            if not df_subject_bolus.empty:
                df_subject_bolus = df_subject_bolus[['bolus']].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_bolus, on="date", how='outer')
                df_subject['bolus'] = df_subject['bolus'].replace(0, np.nan)
            else:
                df_subject['bolus'] = np.nan

            df_subject_basal = df_basal[df_basal['id'] == subject_id].copy()
            if not df_subject_basal.empty:
                df_subject_basal = df_subject_basal[['basal']].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_basal, on="date", how='outer')
            else:
                df_subject['basal'] = np.nan

            df_subject_exercise = df_exercise[df_exercise['id'] == subject_id].copy()
            if not df_subject_exercise.empty:
                df_subject_exercise_workout = df_subject_exercise[['workout_label']].resample('5min', label='right').agg(
                    lambda x: ', '.join(sorted(set(x))))
                df_subject_exercise_workout_intensity = df_subject_exercise[['workout_intensity']].resample('5min',
                                                                                                            label='right').mean()
                df_subject_exercise_workout_duration = df_subject_exercise[['workout_duration']].resample('5min',
                                                                                                          label='right').sum()

                df_subject = pd.merge(df_subject, df_subject_exercise_workout, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_exercise_workout_intensity, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_exercise_workout_duration, on="date", how='outer')

                # Fill NaN for empty strings
                df_subject['workout_label'] = df_subject['workout_label'].replace('', np.nan)
            else:
                df_subject['workout_label'] = np.nan
                df_subject['workout_intensity'] = np.nan
                df_subject['workout_duration'] = np.nan

            if subject_id in heartrate_dict and not heartrate_dict[subject_id].empty:
                df_subject_heartrate = heartrate_dict[subject_id]
                df_subject_heartrate = df_subject_heartrate[['heartrate']].resample('5min', label='right').mean()
                df_subject = pd.merge(df_subject, df_subject_heartrate, on="date", how='outer')
            else:
                print(f"No heartrate data for subject {subject_id}")
                df_subject['heartrate'] = np.nan

            if subject_id in steps_or_cal_burn_dict and not steps_or_cal_burn_dict[subject_id].empty:
                df_subject_steps_or_cal_burn = steps_or_cal_burn_dict[subject_id]
                if 'steps' in df_subject_steps_or_cal_burn.columns:
                    col_name = 'steps'
                    df_subject['calories_burned'] = np.nan
                else:
                    col_name = 'calories_burned'
                    df_subject['steps'] = np.nan
                df_subject_steps_or_cal_burn = df_subject_steps_or_cal_burn[[col_name]].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_steps_or_cal_burn, on="date", how='outer')
            else:
                print(f"No steps or calories burned data for subject {subject_id}")
                df_subject['steps'] = np.nan
                df_subject['calories_burned'] = np.nan

            df_subject.sort_index(inplace=True)

            # Ensuring homogenous time intervals
            df_subject = df_subject.resample('5min').asfreq()

            # Some rows might have gotten added nan values for the subject id after resampling
            df_subject['id'] = subject_id

            # Remove current and following 8 hrs of insulin if outlier
            for dose_col in ['insulin', 'bolus']:
                if dose_col in df_subject.columns:
                    bad_idx = df_subject.index[(df_subject[dose_col] < 0) | (df_subject[dose_col] > 50)]
                    if len(bad_idx) > 0:
                        print(f"Warning: Subject {subject_id} has {len(bad_idx)} outlier {dose_col} values. "
                              "We set the value and the following eight hours of data to nan.")
                        rows_to_nan = []
                        for idx in bad_idx:
                            loc = df_subject.index.get_loc(idx)  # safe unless duplicates exist
                            rows_to_nan.extend(range(loc, loc + 96))
                        rows_to_nan = [i for i in rows_to_nan if i < len(df_subject)]
                        insulin_col = df_subject.columns.get_loc(dose_col)
                        df_subject.iloc[rows_to_nan, insulin_col] = np.nan

            # Check for subjects without cgm or insulin data
            cgm_present = df_subject['CGM'].notna().any()
            insulin_present = (df_subject['bolus'].gt(0).any() or df_subject['basal'].gt(0).any())
            if not cgm_present or not insulin_present:
                print(f"Warning: Dropping subject {subject_id}. CGM present: {cgm_present}. "
                      f"Insulin present: {insulin_present}")
            else:
                processed_dfs.append(df_subject)
                print(f"{count}/{len(self.subject_ids)} are prepared")

            count += 1

        df_final = pd.concat(processed_dfs)

        df_final['insulin'] = df_final['bolus'].fillna(0) + df_final['basal']

        return df_final

    def get_dataframes(self, file_path):
        """
        Extract the data that we are interested in from the dataset, and process them into our format and into separate
        dataframes.
        """
        df_glucose = get_df_glucose(file_path, self.subject_ids)
        print("Glucose data processed", df_glucose)
        df_meals = get_df_meals(file_path, self.subject_ids)
        print("Meal data processed", df_meals)
        df_insulin = get_df_insulin(file_path, self.subject_ids)
        print("Insulin data processed", df_insulin)
        df_bolus = get_df_bolus(df_insulin)
        print("Bolus data processed", df_bolus)
        df_basal = get_df_basal(df_insulin)
        print("Basal data processed", df_basal)
        df_exercise = get_df_exercise(file_path, self.subject_ids)
        print("Exercise data processed", df_exercise)
        heartrate_dict, height_dict, weight_dict = get_vital_sign_dicts(file_path, self.subject_ids)
        print("Heartrate dict processed", heartrate_dict[self.subject_ids[0]])
        print("Height dict processed", height_dict[self.subject_ids[0]])
        print("Weight dict processed", weight_dict[self.subject_ids[0]])
        steps_or_cal_burn_dict = get_steps_or_cal_burn_dict(file_path, self.subject_ids)
        print("Steps or calories burned dict processed", steps_or_cal_burn_dict[self.subject_ids[0]])

        return (df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, height_dict, weight_dict,
                steps_or_cal_burn_dict)

    def add_metadata_columns(self, df, height_dict, weight_dict, file_path):
        """Add demographics and metadata columns to the processed dataframe"""
        
        # Determine if this is T1DEXIP dataset
        is_t1dexip = 'T1DEXIP' in file_path
        
        # Get demographics and device data
        print("Getting demographics data...")
        demographics = get_demographics_data(file_path, is_t1dexip)
        
        print("Getting insulin delivery device data...")
        device_info = get_insulin_delivery_devices(file_path)
        
        # Extract real age/diagnosis data from the dataset
        print("Extracting age and diagnosis data from raw dataset...")
        age_diagnosis_data = get_age_diagnosis_data(file_path)
        
        # Extract real insulin data from the dataset
        print("Extracting insulin types from raw dataset...")
        insulin_df = extract_insulin_data_from_cm(file_path, use_deflate64=not is_t1dexip)
        
        # Create patient insulin mapping based on real data
        print("Creating patient insulin mapping...")
        insulin_mapping = create_patient_insulin_mapping(insulin_df, self.subject_ids, device_info)
        
        # Process each subject's metadata
        unique_subjects = df['id'].unique()
        metadata_records = []
        
        for subject_id in unique_subjects:
            subject_demographics = demographics.get(subject_id, {})
            subject_device = device_info.get(subject_id, '')
            subject_age_diagnosis = age_diagnosis_data.get(subject_id, {})
            height_data = height_dict.get(subject_id)
            weight_data = weight_dict.get(subject_id)

            height = np.nan
            weight = np.nan
            if height_data is None:
                print(f"Warning: Subject id '{subject_id}' does not exist for height!")
            else:
                height = height_data['height'].mean()
            if weight_data is None:
                print(f"Warning: Subject id '{subject_id}' does not exist for weight!")
            else:
                weight = weight_data['weight'].mean()

            # Basic demographics
            age = subject_demographics.get('age')
            gender = subject_demographics.get('gender')
            race = subject_demographics.get('race')
            ethnic = subject_demographics.get('ethnic')
            treatment_group = subject_demographics.get('arm')
            
            # Process insulin delivery information
            normalized_device, algorithm = normalize_device_name_and_get_algorithm(subject_device)
            modality = categorize_insulin_delivery_modality(subject_device)
            cgm_device = 'Dexcom G6'  # All users were on Dexcom G6 from the T1DEXI paper
            
            # Process ethnicity
            ethnicity = standardize_ethnicity(race, ethnic)

            # Get real insulin types from extracted data
            patient_insulin = insulin_mapping.get(str(subject_id), {})
            bolus_insulin = patient_insulin.get('bolus')
            basal_insulin = patient_insulin.get('basal')
            
            metadata_records.append({
                'id': subject_id,
                'insulin_delivery_device': normalized_device,
                'insulin_delivery_algorithm': algorithm,
                'insulin_delivery_modality': modality,
                'cgm_device': cgm_device,
                'ethnicity': ethnicity,
                'age': age,
                'gender': gender,
                'age_of_diagnosis': subject_age_diagnosis,
                'insulin_type_bolus': bolus_insulin,
                'insulin_type_basal': basal_insulin,
                'is_pregnant': np.nan,  # No pregnancy data found in original files
                'height': height,
                'weight': weight,
                'treatment_group': treatment_group,
            })
        
        # Create metadata DataFrame
        df_metadata = pd.DataFrame(metadata_records)

        print("Treatment_groups:", df_metadata['treatment_group'].value_counts(dropna=False))
        print("Gender:", df_metadata['gender'].value_counts(dropna=False))

        # Merge metadata with main dataframe
        df = df.reset_index()
        df = df.merge(df_metadata, on='id', how='left')
        
        print(f"Added metadata columns for {len(unique_subjects)} subjects")
        print("New columns:", [col for col in df_metadata.columns if col != 'id'])
        
        # Print statistics about real data found
        diagnosis_count = df_metadata['age_of_diagnosis'].notna().sum()
        insulin_count = df_metadata['insulin_type_bolus'].notna().sum()
        print(f"Real diagnosis age data found for {diagnosis_count}/{len(unique_subjects)} subjects")
        print(f"Real insulin data found for {insulin_count}/{len(unique_subjects)} subjects")
        
        return df


def standardize_ethnicity(race, ethnic):
    """Standardize ethnicity following DCLP format"""
    
    # Handle missing data
    if pd.isna(race) and pd.isna(ethnic):
        return None
    
    ethnicities = []
    
    # Process race information
    if pd.notna(race):
        race_str = str(race).strip()
        if race_str.upper() == 'WHITE':
            ethnicities.append('White')
        elif race_str.upper() == 'BLACK/AFRICAN AMERICAN':
            ethnicities.append('Black/African American')
        elif race_str.upper() == 'ASIAN':
            ethnicities.append('Asian')
        elif race_str.upper() == 'AMERICAN INDIAN/ALASKAN NATIVE':
            ethnicities.append('American Indian/Alaskan Native')
        elif race_str.upper() == 'MULTIPLE':
            # We'll handle multiple separately if needed
            pass
        elif race_str.upper() in ['NOT REPORTED', 'UNKNOWN', 'DO NOT WISH TO ANSWER']:
            # Don't add anything for these cases
            pass
    
    # Process Hispanic/Latino ethnicity
    if pd.notna(ethnic):
        ethnic_str = str(ethnic).strip()
        if ethnic_str.upper() == 'HISPANIC OR LATINO':
            ethnicities.append('Hispanic/Latino')
    
    # Return combined ethnicity or None if nothing found
    if ethnicities:
        return ', '.join(ethnicities)
    else:
        return None


def categorize_insulin_delivery_modality(device):
    """Categorize insulin delivery device into modality"""
    
    if pd.isna(device) or device == '':
        return None
    
    device_str = str(device).upper()
    
    # MDI - Multiple Daily Injections
    if 'MULTIPLE DAILY INJECTIONS' in device_str:
        return 'MDI'
    
    # SAP - Sensor Augmented Pump
    # Omnipod (excluding Omnipod 5)
    if 'OMNIPOD' in device_str and 'OMNIPOD 5' not in device_str:
        return 'SAP'
    
    # Medtronic 770G and 670G in manual mode
    if ('770G' in device_str and 'MANUAL' in device_str) or ('670G' in device_str and 'MANUAL' in device_str):
        return 'SAP'
    
    # Older Medtronic models (lower number than 670G)
    medtronic_sap_models = ['630G', '640G', '551', '530G', '751', '522', '523', '723']
    if any(model in device_str for model in medtronic_sap_models):
        return 'SAP'
    
    # Check for Paradigm series (older models)
    if 'PARADIGM' in device_str:
        return 'SAP'
    
    # AID - Automated Insulin Delivery (everything else)
    # This includes:
    # - Tandem with Control-IQ or Basal-IQ
    # - Medtronic 670G/770G in Auto Mode
    # - Omnipod 5
    # - Any other modern pump systems
    return 'AID'


def normalize_device_name_and_get_algorithm(device):
    """Normalize device name and determine insulin delivery algorithm"""
    
    if pd.isna(device) or device == '':
        return None, None
    
    device_str = str(device).upper()
    
    # MDI - Multiple Daily Injections
    if 'MULTIPLE DAILY INJECTIONS' in device_str:
        return 'Multiple Daily Injections', np.nan
    
    # OmniPod systems
    if 'OMNIPOD' in device_str:
        if 'OMNIPOD 5' in device_str:
            return 'Omnipod 5', 'Omnipod 5'
        elif 'INSULET OMNIPOD INSULIN MANAGEMENT SYSTEM' in device_str:
            return 'Omnipod', 'Basal-Bolus'
        else:
            return 'Omnipod', 'Basal-Bolus'
    
    # Tandem t:slim X2 systems
    if 'TANDEM T:SLIM X2' in device_str:
        if 'CONTROL IQ' in device_str or 'CONTROL-IQ' in device_str:
            return 't:slim X2', 'Control-IQ'
        elif 'BASAL IQ' in device_str or 'BASAL-IQ' in device_str:
            return 't:slim X2', 'Basal-IQ'
        elif device_str == 'TANDEM T:SLIM X2':
            return 't:slim X2', np.nan
        else:
            return 't:slim X2', np.nan
    
    # Other Tandem systems (like Tandem T:Slim without X2)
    if 'TANDEM T:SLIM' in device_str and 'X2' not in device_str:
        return 't:slim', np.nan
    
    # Medtronic systems
    if 'MEDTRONIC' in device_str:
        # Manual mode systems
        if 'MANUAL' in device_str:
            if '670G' in device_str:
                return 'MiniMed 670G', 'Basal-Bolus'
            elif '770G' in device_str:
                return 'MiniMed 770G', 'Basal-Bolus'
            else:
                # Extract model number for other manual systems
                for model in ['780G', '630G', '640G', '551', '530G', '751', '522', '523', '723']:
                    if model in device_str:
                        return f'MiniMed {model}', 'Basal-Bolus'
                return device_str, 'Basal-Bolus'
        
        # Newer models in auto mode
        elif '670G' in device_str:
            return 'MiniMed 670G', '670G'
        elif '770G' in device_str:
            return 'MiniMed 770G', '770G'
        elif '780G' in device_str:
            return 'MiniMed 780G', '780G'
        
        # Older models (always basal-bolus)
        elif any(model in device_str for model in ['630G', '640G', '551', '530G', '751', '522', '523', '723']):
            for model in ['630G', '640G', '551', '530G', '751', '522', '523', '723']:
                if model in device_str:
                    return f'MiniMed {model}', 'Basal-Bolus'
        
        # Paradigm series
        elif 'PARADIGM' in device_str:
            return 'MiniMed Paradigm', 'Basal-Bolus'
    
    # Default case - return normalized version of original
    normalized = device.title()  # Convert to proper case
    return normalized, np.nan


def standardize_insulin_name(cmtrt, cmscat):
    """Standardize insulin names to consistent format"""
    
    if pd.isna(cmtrt):
        return None
        
    cmtrt_upper = str(cmtrt).upper()
    
    # Handle generic insulin based on subcategory
    if cmtrt_upper in ['BOLUS INSULIN', 'RAPID ACTING INSULIN']:
        return 'Bolus Insulin (Generic)'
    elif cmtrt_upper in ['BASAL INSULIN', 'LONG ACTING INSULIN']:
        return 'Basal Insulin (Generic)'
    elif cmtrt_upper in ['PUMP OR CLOSED LOOP INSULIN', 'INSULIN PUMP']:
        return 'Pump Insulin (Generic)'
    
    # Humalog (Lispro)
    if 'HUMALOG' in cmtrt_upper:
        if 'U-200' in cmtrt_upper:
            return 'Humalog U-200 (Lispro)'
        return 'Humalog (Lispro)'
    
    # Novolog/NovoRapid (Aspart)
    if 'NOVOLOG' in cmtrt_upper or 'NOVORAPID' in cmtrt_upper:
        return 'Novolog (Aspart)'
    
    # Apidra (Glulisine)
    if 'APIDRA' in cmtrt_upper:
        return 'Apidra (Glulisine)'
    
    # Fiasp (Fast-acting Aspart)
    if 'FIASP' in cmtrt_upper:
        return 'Fiasp (Aspart)'
    
    # Admelog (Lispro biosimilar)
    if 'ADMELOG' in cmtrt_upper:
        return 'Admelog (Lispro)'
    
    # Lyumjev (Ultra-rapid Lispro)
    if 'LYUMJEV' in cmtrt_upper:
        return 'Lyumjev (Lispro)'
    
    # Lantus (Glargine)
    if 'LANTUS' in cmtrt_upper:
        return 'Lantus (Glargine)'
    
    # Basaglar (Glargine biosimilar)
    if 'BASAGLAR' in cmtrt_upper:
        return 'Basaglar (Glargine)'
    
    # Tresiba (Degludec)
    if 'TRESIBA' in cmtrt_upper:
        return 'Tresiba (Degludec)'
    
    # Levemir (Detemir)
    if 'LEVEMIR' in cmtrt_upper:
        return 'Levemir (Detemir)'
    
    # Toujeo (Glargine U-300)
    if 'TOUJEO' in cmtrt_upper:
        return 'Toujeo (Glargine)'
    
    # Semglee (Glargine biosimilar)
    if 'SEMGLEE' in cmtrt_upper:
        return 'Semglee (Glargine)'
    
    # Regular insulin
    if 'REGULAR' in cmtrt_upper or cmtrt_upper == 'R':
        return 'Regular Insulin'
    
    # Return original if no match
    return cmtrt


def categorize_insulin_type(insulin_name, cmscat):
    """Categorize insulin as bolus, basal, or pump"""
    
    if pd.isna(insulin_name):
        return None
        
    insulin_upper = str(insulin_name).upper()
    cmscat_upper = str(cmscat).upper() if not pd.isna(cmscat) else ''
    
    # Pump/Closed loop
    if 'PUMP' in cmscat_upper or 'CLOSED LOOP' in cmscat_upper or 'PUMP' in insulin_upper:
        return 'pump'
    
    # Basal insulins (long-acting)
    if any(basal in insulin_upper for basal in ['LANTUS', 'BASAGLAR', 'TRESIBA', 'LEVEMIR', 'TOUJEO', 'SEMGLEE', 'BASAL', 'GLARGINE', 'DEGLUDEC', 'DETEMIR']):
        return 'basal'
    
    # Bolus insulins (rapid-acting)
    if any(bolus in insulin_upper for bolus in ['HUMALOG', 'NOVOLOG', 'NOVORAPID', 'APIDRA', 'FIASP', 'ADMELOG', 'LYUMJEV', 'BOLUS', 'LISPRO', 'ASPART', 'GLULISINE']):
        return 'bolus'
    
    # Regular insulin is typically used as bolus
    if 'REGULAR' in insulin_upper:
        return 'bolus'
    
    return 'unknown'


def extract_insulin_data_from_cm(file_path, use_deflate64=False):
    """Extract insulin data from CM.xpt file in ZIP archive"""
    
    insulin_data = []
    
    try:
        if use_deflate64:
            with zipfile_deflate64.ZipFile(file_path, 'r') as zip_file:
                cm_files = [f for f in zip_file.namelist() if f.endswith('/CM.xpt')]
                
                if cm_files:
                    with zip_file.open(cm_files[0]) as xpt_file:
                        df_cm = xport.to_dataframe(xpt_file)
        else:
            with zipfile.ZipFile(file_path, 'r') as z:
                with tempfile.TemporaryDirectory() as temp_dir:
                    cm_files = [f for f in z.namelist() if f.endswith('/CM.xpt')]
                    
                    if cm_files:
                        z.extract(cm_files[0], temp_dir)
                        full_path = os.path.join(temp_dir, cm_files[0])
                        
                        df_cm = pd.read_sas(full_path, format='xport')
                        
                        # Convert bytes to strings
                        for col in df_cm.columns:
                            if df_cm[col].dtype == 'object':
                                try:
                                    df_cm[col] = df_cm[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                                except:
                                    pass
        
        # Filter for diabetes treatments
        if 'CMCAT' in df_cm.columns:
            diabetes_meds = df_cm[df_cm['CMCAT'].str.upper() == 'DIABETES TREATMENT']
            
            for _, row in diabetes_meds.iterrows():
                usubjid = row['USUBJID']
                cmtrt = row['CMTRT']
                cmscat = row.get('CMSCAT', '')
                
                standardized_name = standardize_insulin_name(cmtrt, cmscat)
                insulin_type = categorize_insulin_type(standardized_name, cmscat)
                
                insulin_data.append({
                    'usubjid': usubjid,
                    'original_name': cmtrt,
                    'standardized_name': standardized_name,
                    'insulin_type': insulin_type,
                    'subcategory': cmscat
                })
    
    except Exception as e:
        print(f"Error extracting insulin data: {e}")
    
    return pd.DataFrame(insulin_data)


def create_patient_insulin_mapping(insulin_df, patient_ids, device_info):
    """Create mapping of patients to their insulin types based on real data"""
    
    patient_insulin_map = {}
    
    # Group by patient
    for usubjid, group in insulin_df.groupby('usubjid'):
        patient_insulins = {
            'bolus': None,
            'basal': None,
            'pump': None
        }
        
        for _, row in group.iterrows():
            insulin_type = row['insulin_type']
            standardized_name = row['standardized_name']
            
            if insulin_type in patient_insulins:
                patient_insulins[insulin_type] = standardized_name
        
        # Get delivery modality for this patient
        patient_id = str(usubjid)
        device = device_info.get(usubjid, '')
        modality = categorize_insulin_delivery_modality(device)
        
        # Assign insulin types based on delivery modality and available data
        if modality == 'MDI':
            # MDI: use specific bolus and basal
            bolus_insulin = patient_insulins['bolus']
            basal_insulin = patient_insulins['basal']
        elif modality in ['AID', 'SAP']:
            # Pump users: use pump insulin or bolus for both
            pump_insulin = patient_insulins['pump'] or patient_insulins['bolus']
            bolus_insulin = pump_insulin
            basal_insulin = pump_insulin
        else:
            bolus_insulin = None
            basal_insulin = None
            
        patient_insulin_map[patient_id] = {
            'bolus': bolus_insulin,
            'basal': basal_insulin
        }
    
    return patient_insulin_map


def get_demographics_data(file_path, is_t1dexip=False):
    """Get demographics data (age, race, ethnicity) from DM files"""
    
    demographics = {}
    
    try:
        if is_t1dexip:
            with zipfile.ZipFile(file_path, 'r') as z:
                with tempfile.TemporaryDirectory() as temp_dir:
                    dm_files = [f for f in z.namelist() if f.endswith('/DM.xpt')]
                    
                    if dm_files:
                        dm_file = dm_files[0]
                        z.extract(dm_file, temp_dir)
                        full_path = os.path.join(temp_dir, dm_file)
                        
                        df = pd.read_sas(full_path, format='xport')
                        
                        # Convert bytes to strings
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                try:
                                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                                except:
                                    pass
                        
                        for idx, row in df.iterrows():
                            patient_id = row['USUBJID']
                            gender_map = {'M': 'Male', 'F': 'Female'}
                            gender = gender_map.get(row['SEX']) if pd.notna(row['SEX']) else None

                            demographics[patient_id] = {
                                'age': row.get('AGE'),
                                'gender': gender,
                                'race': row.get('RACE'),
                                'ethnic': row.get('ETHNIC'),
                                'arm': row.get('ARM')
                            }
        else:
            with zipfile_deflate64.ZipFile(file_path, 'r') as zip_file:
                dm_files = [f for f in zip_file.namelist() if f.endswith('/DM.xpt')]
                
                if dm_files:
                    dm_file = dm_files[0]
                    
                    with zip_file.open(dm_file) as xpt_file:
                        df = xport.to_dataframe(xpt_file)
                    
                    for idx, row in df.iterrows():
                        patient_id = row['USUBJID']
                        gender_map = {'M': 'Male', 'F': 'Female'}
                        gender = gender_map.get(row['SEX']) if pd.notna(row['SEX']) else None

                        demographics[patient_id] = {
                            'age': row.get('AGE'),
                            'gender': gender,
                            'race': row.get('RACE'),
                            'ethnic': row.get('ETHNIC'),
                            'arm': row.get('ARM')
                        }
    
    except Exception as e:
        print(f"Error getting demographics data: {e}")
    
    return demographics


def get_insulin_delivery_devices(file_path):
    """Get insulin delivery device information from DX files"""
    
    device_info = {}
    
    try:
        df_device = get_df_from_zip_deflate_64(file_path, 'DX.xpt')
        
        for idx, row in df_device.iterrows():
            patient_id = row['USUBJID']
            device_info[patient_id] = row.get('DXTRT', '')
    
    except Exception as e:
        print(f"Error getting device information: {e}")
    
    return device_info


# ======= MAIN PROCESSING FUNCTIONS =======

def get_valid_subject_ids(file_path):
    df_device = get_df_from_zip_deflate_64(file_path, 'DX.xpt')
    #subject_ids_not_on_mdi = df_device[df_device['DXTRT'] != 'MULTIPLE DAILY INJECTIONS']['USUBJID'].unique()
    print("ALL SUBJECTS ARE INCLUDED")
    return df_device['USUBJID'].unique()


def get_vital_sign_dicts(file_path, subject_ids):
    # Heartrate data is processed by chunks because the heartrate file is so big and creates memory problems
    row_count = 0
    chunksize = 1000000
    file_name = '/VS.xpt'
    heartrate_dict = {}
    height_dict = {}
    weight_dict = {}

    with zipfile_deflate64.ZipFile(file_path, 'r') as zip_file:
        matched_files = [f for f in zip_file.namelist() if f.endswith(file_name)]
        matched_file = matched_files[0]
        with zip_file.open(matched_file) as xpt_file:
            for chunk in pd.read_sas(xpt_file, format='xport', chunksize=chunksize):
                row_count += chunksize
                if isinstance(chunk, pd.DataFrame):
                    df = chunk.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
                elif isinstance(chunk, pd.Series):
                    df = chunk.map(lambda x: x.decode() if isinstance(x, bytes) else x)
                else:
                    raise TypeError("chunk must be a pandas Series or DataFrame")

                # For some reason, the heart rate mean gives more data in T1DEXI version, while not in T1DEXIP
                # The heart rate from both devices in the study
                df_heartrate = df[(df['VSTEST'] == 'Heart Rate') | (df['VSTEST'] == 'Heart Rate, Mean')]

                # Filter the DataFrame for subject_ids before looping over unique values
                df_heartrate = df_heartrate[df_heartrate['USUBJID'].isin(subject_ids)]

                for subject_id in df_heartrate['USUBJID'].unique():
                    filtered_rows = df_heartrate[df_heartrate['USUBJID'] == subject_id].copy()
                    filtered_rows['date'] = create_sas_date_for_column(filtered_rows['VSDTC'])
                    filtered_rows.loc[:, 'VSSTRESN'] = pd.to_numeric(filtered_rows['VSSTRESN'])
                    filtered_rows.rename(columns={'VSSTRESN': 'heartrate'}, inplace=True)
                    filtered_rows = filtered_rows[['heartrate', 'date']]
                    filtered_rows.set_index('date', inplace=True)
                    filtered_rows = filtered_rows.resample('5min', label='right').mean()

                    # If the USUBJID is already in the dictionary, append the new data
                    if subject_id in heartrate_dict:
                        heartrate_dict[subject_id] = pd.concat([heartrate_dict[subject_id], filtered_rows])
                    else:
                        # Otherwise, create a new DataFrame for this USUBJID
                        heartrate_dict[subject_id] = filtered_rows

                # Do the same for height
                df_height = df[df['VSTEST'] == 'Height']
                df_height = df_height[df_height['USUBJID'].isin(subject_ids)]

                for subject_id in df_height['USUBJID'].unique():
                    filtered_rows = df_height[df_height['USUBJID'] == subject_id].copy()
                    filtered_rows.loc[:, 'VSSTRESN'] = pd.to_numeric(filtered_rows['VSSTRESN'])
                    filtered_rows.rename(columns={'VSSTRESN': 'height'}, inplace=True)
                    filtered_rows['height'] = filtered_rows['height'] / 12  # From inches to feet

                    # If the USUBJID is already in the dictionary, append the new data
                    if subject_id in height_dict:
                        height_dict[subject_id] = pd.concat([height_dict[subject_id], filtered_rows])
                    else:
                        # Otherwise, create a new DataFrame for this USUBJID
                        height_dict[subject_id] = filtered_rows

                # Do the same for weight
                df_weight = df[df['VSTEST'] == 'Weight']
                df_weight = df_weight[df_weight['USUBJID'].isin(subject_ids)]

                for subject_id in df_height['USUBJID'].unique():
                    filtered_rows = df_weight[df_weight['USUBJID'] == subject_id].copy()
                    filtered_rows.loc[:, 'VSSTRESN'] = pd.to_numeric(filtered_rows['VSSTRESN'])
                    filtered_rows.rename(columns={'VSSTRESN': 'weight'}, inplace=True)

                    # If the USUBJID is already in the dictionary, append the new data
                    if subject_id in weight_dict:
                        weight_dict[subject_id] = pd.concat([weight_dict[subject_id], filtered_rows])
                    else:
                        # Otherwise, create a new DataFrame for this USUBJID
                        weight_dict[subject_id] = filtered_rows

            return heartrate_dict, height_dict, weight_dict


def get_steps_or_cal_burn_dict(file_path, subject_ids):
    # Steps data is processed by chunks because the steps file is so big and creates memory problems
    row_count = 0
    chunksize = 10000
    file_name = '/FA.xpt'
    results_dict = {}

    with zipfile_deflate64.ZipFile(file_path, 'r') as zip_file:
        matched_files = [f for f in zip_file.namelist() if f.endswith(file_name)]
        matched_file = matched_files[0]
        with zip_file.open(matched_file) as xpt_file:
            for chunk in pd.read_sas(xpt_file, format='xport', chunksize=chunksize):
                row_count += chunksize
                df_fa = chunk.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

                # Filter the DataFrame for subject_ids before looping over unique values
                df_fa = df_fa[df_fa['USUBJID'].isin(subject_ids)]
                df_fa['date'] = create_sas_date_for_column(df_fa['FADTC'])

                for subject_id in df_fa['USUBJID'].unique():
                    filtered_rows = df_fa[df_fa['USUBJID'] == subject_id].copy()

                    if 'T1DEXIP' in file_path:
                        filtered_rows = filtered_rows[filtered_rows['FATESTCD'] == 'CALBURN']
                        filtered_rows.loc[:, 'FASTRESN'] = pd.to_numeric(filtered_rows['FASTRESN'])
                        filtered_rows.rename(columns={'FASTRESN': 'calories_burned'}, inplace=True)
                        filtered_rows = filtered_rows[['calories_burned', 'date']]
                    else:
                        filtered_rows = filtered_rows[filtered_rows['FAOBJ'] == '10-SECOND INTERVAL STEP COUNT']
                        filtered_rows.loc[:, 'FASTRESN'] = pd.to_numeric(filtered_rows['FASTRESN'])
                        filtered_rows.rename(columns={'FASTRESN': 'steps'}, inplace=True)
                        filtered_rows = filtered_rows[['steps', 'date']]

                    filtered_rows.set_index('date', inplace=True)
                    filtered_rows = filtered_rows.resample('5min', label='right').sum()

                    # If the USUBJID is already in the dictionary, append the new data
                    if subject_id in results_dict:
                        results_dict[subject_id] = pd.concat([results_dict[subject_id], filtered_rows])
                    else:
                        # Otherwise, create a new DataFrame for this USUBJID
                        results_dict[subject_id] = filtered_rows
            return results_dict

def get_df_glucose(file_path, subject_ids):
    df_glucose = get_df_from_zip_deflate_64(file_path, 'LB.xpt', subject_ids=subject_ids)
    df_glucose['date'] = create_sas_date_for_column(df_glucose['LBDTC'])
    df_glucose.loc[:, 'LBSTRESN'] = pd.to_numeric(df_glucose['LBSTRESN'], errors='coerce')
    df_glucose.rename(columns={'LBSTRESN': 'CGM', 'USUBJID': 'id'}, inplace=True)
    df_glucose = df_glucose[df_glucose['LBTESTCD'] == 'GLUC']  # Filtering out glucose (from hba1c)
    df_glucose = df_glucose[['id', 'CGM', 'date']]
    df_glucose.set_index('date', inplace=True)
    return df_glucose


def get_df_meals(file_path, subject_ids):
    df_meals = get_df_from_zip_deflate_64(file_path, 'FAMLPM.xpt', subject_ids=subject_ids)
    df_meals = df_meals[df_meals['FACAT'] == 'CONSUMED']  # Consumed is taken - returned
    df_meals = df_meals[df_meals['FATESTCD'] == 'DCARBT']  # Extracting dietary carbohydrates
    df_meals['date'] = create_sas_date_for_column(df_meals['FADTC'])
    df_meals.loc[:, 'FAORRES'] = pd.to_numeric(df_meals['FAORRES'], errors='coerce')
    df_meals.rename(columns={'FAORRES': 'carbs', 'PTMLDESC': 'meal_label', 'USUBJID': 'id'}, inplace=True)
    df_meals = df_meals[['carbs', 'meal_label', 'id', 'date']]
    df_meals.set_index('date', inplace=True)
    return df_meals


def get_df_insulin(file_path, subject_ids):
    df_insulin = get_df_from_zip_deflate_64(file_path, 'FACM.xpt', subject_ids=subject_ids)
    df_insulin['date'] = create_sas_date_for_column(df_insulin['FADTC'])
    df_insulin.loc[:, 'FASTRESN'] = pd.to_numeric(df_insulin['FASTRESN'], errors='coerce')
    df_insulin.rename(columns={'FASTRESN': 'dose', 'FAORRESU': 'unit', 'USUBJID': 'id', 'INSNMBOL':
        'delivered bolus', 'INSEXBOL': 'extended bolus', 'FADUR': 'duration', 'FATEST': 'type'}, inplace=True)
    df_insulin['duration'] = df_insulin['duration'].apply(lambda x: parse_duration(x))
    df_insulin = df_insulin.drop_duplicates()
    return df_insulin[['id', 'unit', 'dose', 'date', 'duration', 'delivered bolus', 'extended bolus', 'INSSTYPE',
                       'type']]


def get_df_bolus(df_insulin):
    df_bolus = df_insulin[df_insulin['type'] == 'BOLUS INSULIN'].copy()
    # Let values in delivered bolus override dose for extended boluses
    df_bolus.loc[df_bolus['delivered bolus'].notna(), 'dose'] = df_bolus[df_bolus['delivered bolus'].notna()]['delivered bolus']
    df_bolus.drop(columns=['delivered bolus'], inplace=True)
    # For the square boluses, "delivered bolus" is nan, but dose should be 0.0 because all of the bolus dose is extended
    df_bolus.loc[df_bolus['INSSTYPE'] == 'square', 'dose'] = 0.0

    # Get extended boluses as a dataframe with doses spread across 5-minute intervals
    df_bolus['extended bolus'] = df_bolus['extended bolus'].replace(0.0, np.nan)
    extended_boluses = df_bolus[df_bolus['extended bolus'].notna()].copy()
    new_rows = []
    for _, row in extended_boluses.iterrows():
        new_rows.extend(split_duration(row, 'extended bolus'))

    # Only create extended_boluses DataFrame if there are new rows
    if new_rows:
        extended_boluses = pd.DataFrame(new_rows)
        extended_boluses.drop(columns=['duration'], inplace=True)
        extended_boluses.rename(columns={'extended bolus': 'dose'}, inplace=True)
        merged_bolus_df = pd.concat([df_bolus[['date', 'dose', 'id']], extended_boluses[['date', 'dose', 'id']]], ignore_index=True)
    else:
        merged_bolus_df = df_bolus[['date', 'dose', 'id']].copy()

    merged_bolus_df.rename(columns={'dose': 'bolus'}, inplace=True)
    merged_bolus_df.set_index('date', inplace=True)

    return merged_bolus_df


def get_df_basal(df_insulin):
    df_basal = df_insulin[df_insulin['type'] != 'BOLUS INSULIN'][['INSSTYPE', 'dose', 'unit', 'date', 'id', 'duration']]
    df_basal = df_basal[df_basal['unit'] == 'U']
    df_basal['duration'] = df_basal['duration'].fillna(pd.Timedelta(minutes=0))
    df_basal = df_basal.drop_duplicates(subset=['id', 'date'])  # Drop sample when start dates are the same

    # manipulate duration to match with the next sample if the id is the same
    df_basal['next_date'] = df_basal['date'].shift(-1)  # Get next row's date
    df_basal['next_id'] = df_basal['id'].shift(-1)  # Get next row's id
    df_basal['duration'] = df_basal.apply(
        lambda row: (row['next_date'] - row['date']) if row['id'] == row['next_id'] else row['duration'],
        axis=1
    )
    # Split basal insulin into time interval bins so that it is distributed in the resampling
    new_rows = []
    for _, row in df_basal.iterrows():
        new_rows.extend(split_duration(row, 'dose'))
    df_basal = pd.DataFrame(new_rows)
    df_basal.drop(columns=['duration'], inplace=True)
    df_basal['dose'] = df_basal['dose']  # Basal in absolute U
    df_basal.set_index('date', inplace=True)
    df_basal.rename(columns={'dose': 'basal'}, inplace=True)
    return df_basal


def get_df_exercise(file_path, subject_ids):
    df_exercise = get_df_from_zip_deflate_64(file_path, '/PR.xpt', subject_ids=subject_ids)
    df_exercise['date'] = pd.to_datetime(create_sas_date_for_column(df_exercise['PRSTDTC']))

    if 'T1DEXIP' in file_path:
        exercise_map = {
            '': 0,
            'Low: Pretty easy': 1.7,
            'Mild: Working a bit': 3.3,
            'Feeling the burn - Mild': 5.0,
            'Moderate: Working to keep up': 6.7,
            'Heavy: Hard to keep going but did it': 8.3,
            'Exhaustive: Too tough/Had to stop': 10
        }
        df_exercise.loc[:, 'EXCINTSY'] = df_exercise['EXCINTSY'].map(exercise_map)
    else:
        # Original values are 0, 1 and 2, but we add 1 and multiply with 3.3 so the scale is from 0-10
        df_exercise.loc[:, 'EXCINTSY'] = pd.to_numeric(df_exercise['EXCINTSY'])
        df_exercise.loc[:, 'EXCINTSY'] = df_exercise['EXCINTSY'] + 1
        df_exercise['EXCINTSY'] *= 3.3
    df_exercise.rename(
        columns={'USUBJID': 'id', 'PRCAT': 'workout_label', 'PLNEXDUR': 'workout_duration',
                 'EXCINTSY': 'workout_intensity'}, inplace=True)
    df_exercise = df_exercise[
        ['workout_label', 'workout_duration', 'workout_intensity', 'id', 'date']]
    df_exercise.set_index('date', inplace=True)
    return df_exercise


def split_duration(row, value_column):
    """
    For bolus doses or basal rates with a duration, we split the dose injection across 5-minute intervals by adding
    new rows for every 5-minute window in duration, and equally split the original dose across those rows.
    """
    rounded_duration = round(row['duration'] / pd.Timedelta(minutes=5)) * pd.Timedelta(minutes=5)
    num_intervals = rounded_duration // pd.Timedelta(minutes=5)
    if num_intervals < 1:
        num_intervals = 1
    value_per_interval = row[value_column] / num_intervals
    new_rows = []
    for i in range(int(num_intervals)):
        new_row = {
            'date': row['date'] + pd.Timedelta(minutes=5 * i),
            value_column: value_per_interval,
            'duration': pd.Timedelta(minutes=5),
            'id': row['id'],
        }
        new_rows.append(new_row)
    return new_rows


def parse_duration(duration_str):
    try:
        return pd.to_timedelta(isodate.parse_duration(duration_str)) if duration_str else np.nan
    except isodate.ISO8601Error:
        return np.nan


def create_sas_date_for_column(series):
    sas_epoch = datetime.datetime(1960, 1, 1)
    series = pd.to_datetime(series, unit='s', origin=sas_epoch)
    return series


def get_df_from_zip_deflate_64(zip_path, file_name, subject_ids=None):
    with zipfile_deflate64.ZipFile(zip_path, 'r') as zip_file:
        # Find the file in the archive that ends with the specified file_name
        matched_files = [f for f in zip_file.namelist() if f.endswith(file_name)]

        if not matched_files:
            raise FileNotFoundError(f"No file ending with '{file_name}' found in the zip archive.")

        # Use the first match (if multiple matches, refine criteria as needed)
        matched_file = matched_files[0]

        # Read the .xpt file directly from the zip
        with zip_file.open(matched_file) as xpt_file:
            df = xport.to_dataframe(xpt_file)  # Load the .xpt file into a DataFrame

        if subject_ids is not None:
            df = df[df['USUBJID'].isin(subject_ids)]
        return df


def get_age_diagnosis_data(file_path):
    file_name = "/FA.xpt"
    needed_columns = ["USUBJID", "FAOBJ", "FAORRES"]  # columns to keep
    chunk_size = 10000

    age_diagnosis_data = {}

    with zipfile.ZipFile(file_path, "r") as zip_file:
        # Find the target XPT file
        matched_file = next(f for f in zip_file.namelist() if f.endswith(file_name))

        with zip_file.open(matched_file) as xpt_file:
            # Iterate in chunks
            for chunk in pd.read_sas(xpt_file, format="xport", chunksize=chunk_size):
                # Keep only needed columns
                chunk = chunk[needed_columns]
                chunk = chunk.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

                # Filter for diabetes onset
                chunk_filtered = chunk[chunk['FAOBJ'] == 'DIABETES ONSET']

                # Clean and convert age values
                chunk_filtered['FAORRES'] = chunk_filtered['FAORRES'].replace('6-<12', '0')
                chunk_filtered['FAORRES'] = pd.to_numeric(chunk_filtered['FAORRES'], errors='coerce')

                # Populate dictionary
                for _, row in chunk_filtered.iterrows():
                    subj_id = row['USUBJID']
                    age_diagnosis_data[subj_id] = row['FAORRES']

    return age_diagnosis_data


def main():
    parser = Parser()
    df = parser("data/raw/t1dexi/T1DEXI.zip")
    df.to_csv('T1DEXI.csv')


if __name__ == "__main__":
    main()



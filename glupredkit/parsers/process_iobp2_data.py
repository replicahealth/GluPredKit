#!/usr/bin/env python3
"""
Process IOBP2 data: Generate user data expansion and apply standardizations
"""

import pandas as pd
import numpy as np
import os

def generate_iobp2_data():
    """Generate IOBP2 user data expansion"""
    # Set up file paths
    base_path = "data/raw/IOBP2 RCT Public Dataset/Data Tables"
    output_path = "data/user_data_expansion/"
    
    print("IOBP2 User Data Expansion Pipeline")
    print("=" * 50)
    
    # Step 1: Load roster data
    print("Step 1: Reading participant roster...")
    roster = pd.read_csv(os.path.join(base_path, 'IOBP2PtRoster.txt'), delimiter='|')
    print(f"Roster shape: {roster.shape}")
    print("Treatment groups:")
    print(roster['TrtGroup'].value_counts().to_dict())
    
    # Step 2: Load screening data for device information
    print("\nStep 2: Reading screening data...")
    screening = pd.read_csv(os.path.join(base_path, 'IOBP2DiabScreening.txt'), delimiter='|')
    print(f"Screening shape: {screening.shape}")
    print("Unique pump types:")
    print(screening['PumpType'].value_counts().to_dict())

    # Step 3: Filter completed participants only
    print("\nStep 3: Filtering completed participants...")
    completed_roster = roster[roster['RCTPtStatus'] == 'Completed'].copy()
    print(f"Completed participants: {len(completed_roster)}")
    
    # Step 4: Merge roster with screening data
    print("\nStep 4: Merging roster with screening data...")
    merged_data = completed_roster.merge(screening, on='PtID', how='inner')
    print(f"Merged data shape: {merged_data.shape}")
    
    # Step 5: Create user data expansion dataframe
    print("\nStep 5: Creating user data expansion...")
    user_data_expansion = pd.DataFrame()
    
    # id
    user_data_expansion['id'] = merged_data['PtID']

    # insulin_delivery_algorithm - Map based on treatment group and device
    def map_insulin_delivery_algorithm(trt_group, device, is_mdi):
        if trt_group in ['BP', 'BPFiasp']:
            # Bionic Pancreas used automated insulin delivery
            return 'Bionic Pancreas'
        else:
            # Default based on device
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

    user_data_expansion['insulin_delivery_algorithm'] = merged_data.apply(
        lambda x: map_insulin_delivery_algorithm(x['TrtGroup'], x['PumpType'], x['InsModInjections']), axis=1
    )

    #user_data_expansion['PumpType'] = merged_data['PumpType']
    #user_data_expansion['InsModInjections'] = merged_data['InsModInjections']
    #user_data_expansion['InsModPump'] = merged_data['InsModPump']
    #user_data_expansion['InsModInhaled'] = merged_data['InsModInhaled']
    #user_data_expansion['TrtGroup'] = merged_data['TrtGroup']

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
    
    user_data_expansion['insulin_delivery_device'] = merged_data.apply(
        lambda x: map_insulin_delivery_device(x['TrtGroup'], x['PumpType'], x['InsModInjections']), axis=1
    )

    user_data_expansion['cgm_device'] = 'Dexcom G6'
    
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

    user_data_expansion['ethnicity'] = merged_data.apply(combine_ethnicity_race, axis=1)
    
    # age_of_diagnosis - Not available in IOBP2, set to NaN
    user_data_expansion['age_of_diagnosis'] = merged_data['DiagAge']
    
    # is_pregnant - Pregnancy is exclusion criterion
    user_data_expansion['is_pregnant'] = False
    
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
    
    user_data_expansion['insulin_delivery_modality'] = user_data_expansion['insulin_delivery_algorithm'].apply(map_insulin_delivery_modality)
    
    # insulin_type_bolus and insulin_type_basal - Use common insulin for IOBP2 era
    user_data_expansion['insulin_type_bolus'] = 'Novolog (Aspart)'
    user_data_expansion['insulin_type_basal'] = 'Novolog (Aspart)'  # Same for pump users
    
    print(f"User data expansion created with {len(user_data_expansion)} participants")
    
    print("\nDistributions:")
    print(f"Treatment groups: {merged_data['TrtGroup'].value_counts().to_dict()}")
    print(f"Insulin delivery devices: {user_data_expansion['insulin_delivery_device'].value_counts(dropna=False).to_dict()}")
    print(f"Insulin delivery algorithms: {user_data_expansion['insulin_delivery_algorithm'].value_counts(dropna=False).to_dict()}")
    print(f"Insulin delivery modalities: {user_data_expansion['insulin_delivery_modality'].value_counts(dropna=False).to_dict()}")
    print(f"CGM devices: {user_data_expansion['cgm_device'].value_counts(dropna=False).to_dict()}")
    print(f"Ethnicity: {user_data_expansion['ethnicity'].value_counts(dropna=False).to_dict()}")

    return user_data_expansion

def update_iobp2_data(df):
    """Apply standardizations to IOBP2 data"""
    print("\n" + "=" * 50)
    print("APPLYING IOBP2 DATA STANDARDIZATIONS")
    print("=" * 50)
    
    # Show current null counts
    print("Current null value counts:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count} null values")

    print(f"\nStandardization complete for {len(df)} participants")
    return df

def main():
    """Main processing function"""
    output_path = "data/user_data_expansion/"
    
    # Step 1: Generate IOBP2 data
    df = generate_iobp2_data()
    
    # Step 2: Apply standardizations
    df = update_iobp2_data(df)
    
    # Step 3: Save dataframe
    print("\nSaving dataframe...")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "IOBP2.csv")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved IOBP2 dataframe to: {output_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("IOBP2 DATA PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total participants: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("Dataset ready for analysis!")
    
    return df

if __name__ == "__main__":
    df = main()
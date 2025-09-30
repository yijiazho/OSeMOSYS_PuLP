#!/usr/bin/env python3
"""
Add Storage Efficiency Parameters to OSeMOSYS Input File
=======================================================

This script adds realistic storage efficiency and loss parameters to the UTOPIA model:
- StorageChargeEfficiency: 0.90 (90% charge efficiency)
- StorageDischargeEfficiency: 0.90 (90% discharge efficiency)
- StorageStandingLossRate: 0.005 (0.5% annual standing loss)

Author: Claude AI Assistant
"""

import pandas as pd
from pathlib import Path


def add_storage_parameters():
    """
    Add storage efficiency parameters to the UTOPIA input file.
    """
    print("Adding storage efficiency parameters to UTOPIA model...")

    # File paths
    input_file = Path('Input_Data/UTOPIA_BASE_with_storage_losses.xlsx')

    # Load the current parameters
    df_params = pd.read_excel(input_file, sheet_name='PARAMETERS')

    print(f"Current parameters shape: {df_params.shape}")
    print(f"Columns: {list(df_params.columns)}")

    # Define the storage parameters to add
    storage_params = [
        {
            'PARAM': 'StorageChargeEfficiency',
            'VALUE': 0.90,
            'STORAGE': 'DAM',
            'REGION': 'UTOPIA',
            'YEAR': None  # Applies to all years
        },
        {
            'PARAM': 'StorageDischargeEfficiency',
            'VALUE': 0.90,
            'STORAGE': 'DAM',
            'REGION': 'UTOPIA',
            'YEAR': None
        },
        {
            'PARAM': 'StorageStandingLossRate',
            'VALUE': 0.005,
            'STORAGE': 'DAM',
            'REGION': 'UTOPIA',
            'YEAR': None
        }
    ]

    # Create new parameter rows
    new_rows = []

    for param_def in storage_params:
        # Create a row with all required columns
        new_row = {col: None for col in df_params.columns}

        # Fill in the parameter-specific values
        new_row['PARAM'] = param_def['PARAM']
        new_row['VALUE'] = param_def['VALUE']
        new_row['STORAGE'] = param_def['STORAGE']
        new_row['REGION'] = param_def['REGION']
        new_row['YEAR'] = param_def['YEAR']

        new_rows.append(new_row)

        print(f"Adding parameter: {param_def['PARAM']} = {param_def['VALUE']}")

    # Convert new rows to DataFrame
    new_params_df = pd.DataFrame(new_rows)

    # Append to existing parameters
    updated_params_df = pd.concat([df_params, new_params_df], ignore_index=True)

    print(f"Updated parameters shape: {updated_params_df.shape}")
    print(f"Added {len(new_rows)} new parameters")

    # Save back to Excel file
    # We need to preserve all other sheets

    # Read all sheets from the original file
    with pd.ExcelFile(input_file) as xls:
        all_sheets = {}
        for sheet_name in xls.sheet_names:
            if sheet_name == 'PARAMETERS':
                all_sheets[sheet_name] = updated_params_df
            else:
                all_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)

    # Write back to Excel with all sheets
    with pd.ExcelWriter(input_file, engine='openpyxl') as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Successfully updated {input_file}")

    # Verify the addition
    verify_parameters(input_file)


def verify_parameters(file_path):
    """
    Verify that the storage parameters were added correctly.
    """
    print("\\nVerifying added parameters...")

    df_params = pd.read_excel(file_path, sheet_name='PARAMETERS')

    # Check for the new storage parameters
    storage_efficiency_params = ['StorageChargeEfficiency', 'StorageDischargeEfficiency', 'StorageStandingLossRate']

    for param_name in storage_efficiency_params:
        param_rows = df_params[df_params['PARAM'] == param_name]

        if not param_rows.empty:
            print(f"✓ {param_name}: {param_rows['VALUE'].iloc[0]} (STORAGE: {param_rows['STORAGE'].iloc[0]})")
        else:
            print(f"✗ {param_name}: NOT FOUND")

    print("\\nAll existing storage-related parameters:")
    all_storage = df_params[df_params['PARAM'].str.contains('Storage', case=False, na=False)]
    if not all_storage.empty:
        print(all_storage[['PARAM', 'VALUE', 'STORAGE', 'TECHNOLOGY']].to_string(index=False))
    else:
        print("No storage parameters found!")


def main():
    """
    Main function to add storage parameters.
    """
    print("OSeMOSYS Storage Parameter Enhancement")
    print("=" * 40)

    # Check if input file exists
    input_file = Path('Input_Data/UTOPIA_BASE_with_storage_losses.xlsx')
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please ensure the file was copied correctly.")
        return

    try:
        add_storage_parameters()
        print("\\n" + "=" * 40)
        print("Storage parameters added successfully!")
        print("\\nRecommended values used:")
        print("• StorageChargeEfficiency: 0.90 (90% charging efficiency)")
        print("• StorageDischargeEfficiency: 0.90 (90% discharge efficiency)")
        print("• StorageStandingLossRate: 0.005 (0.5% annual standing loss)")
        print("\\nRound-trip efficiency: 81% (0.90 × 0.90)")

    except Exception as e:
        print(f"Error adding storage parameters: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
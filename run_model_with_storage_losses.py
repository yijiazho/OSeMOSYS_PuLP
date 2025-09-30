#!/usr/bin/env python3
"""
Run OSeMOSYS Model with Storage Losses
=====================================

This script runs the OSeMOSYS model using the enhanced input file with storage efficiency parameters.
It modifies the input file path and runs the model to generate results with storage losses.

Author: Claude AI Assistant
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_osemosys_with_storage_losses():
    """
    Run the OSeMOSYS model with the enhanced storage parameters.
    """
    print("Running OSeMOSYS model with storage efficiency parameters...")

    # Backup the original OSeMOSYS script
    original_script = 'OSeMOSYS_PuLP.py'
    backup_script = 'OSeMOSYS_PuLP_backup.py'

    if not Path(backup_script).exists():
        shutil.copy2(original_script, backup_script)
        print(f"Created backup: {backup_script}")

    # Read the original script
    with open(original_script, 'r') as f:
        content = f.read()

    # Modify the input file name to use our enhanced version
    modified_content = content.replace(
        'inputFile = "UTOPIA_BASE.xlsx"',
        'inputFile = "UTOPIA_BASE_with_storage_losses.xlsx"'
    )

    # Also modify the output suffix to distinguish the results
    modified_content = modified_content.replace(
        '_suffix = "_losses"' if '_suffix = "_losses"' in modified_content else '_suffix = ""',
        '_suffix = "_losses"'
    )

    # If no suffix line exists, add it after the inputFile line
    if '_suffix' not in modified_content:
        modified_content = modified_content.replace(
            'inputFile = "UTOPIA_BASE_with_storage_losses.xlsx"',
            'inputFile = "UTOPIA_BASE_with_storage_losses.xlsx"\n_suffix = "_losses"'
        )

    # Write the modified script temporarily
    temp_script = 'OSeMOSYS_PuLP_with_losses.py'
    with open(temp_script, 'w') as f:
        f.write(modified_content)

    print(f"Created temporary script: {temp_script}")
    print("Input file: UTOPIA_BASE_with_storage_losses.xlsx")
    print("Expected output: UTOPIA_BASE_with_storage_losses_result_losses.xlsx")

    try:
        # Run the modified OSeMOSYS script
        print("\\nStarting OSeMOSYS model run...")
        print("This may take several minutes...")

        result = subprocess.run([sys.executable, temp_script],
                              capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print("✓ Model run completed successfully!")
            print("\\nModel output:")
            print(result.stdout[-1000:])  # Last 1000 characters of output

            # Check for output file
            expected_output = "Output_Data/UTOPIA_BASE_with_storage_losses_result_losses.xlsx"
            if Path(expected_output).exists():
                print(f"✓ Output file created: {expected_output}")
            else:
                print("⚠ Output file not found at expected location")
                # List files in Output_Data to see what was created
                output_files = list(Path("Output_Data").glob("*losses*"))
                if output_files:
                    print("Found output files:")
                    for f in output_files:
                        print(f"  {f}")

        else:
            print("✗ Model run failed!")
            print("Error output:")
            print(result.stderr)
            print("\\nStandard output:")
            print(result.stdout)

    except subprocess.TimeoutExpired:
        print("✗ Model run timed out after 10 minutes")
    except Exception as e:
        print(f"✗ Error running model: {e}")

    finally:
        # Clean up temporary script
        if Path(temp_script).exists():
            os.remove(temp_script)
            print(f"Cleaned up temporary script: {temp_script}")


def main():
    """
    Main function to run the enhanced model.
    """
    print("OSeMOSYS Model Run with Storage Efficiency Parameters")
    print("=" * 55)

    # Check if required files exist
    required_files = [
        "OSeMOSYS_PuLP.py",
        "Input_Data/UTOPIA_BASE_with_storage_losses.xlsx"
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  ✗ {f}")
        return

    print("Required files found:")
    for f in required_files:
        print(f"  ✓ {f}")

    # Ensure Output_Data directory exists
    Path("Output_Data").mkdir(exist_ok=True)

    run_osemosys_with_storage_losses()


if __name__ == "__main__":
    main()
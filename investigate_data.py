#!/usr/bin/env python3
"""
Investigate the OSeMOSYS-PuLP results data to understand capacity patterns
"""

import pandas as pd
import numpy as np

def investigate_capacity_data():
    """Detailed investigation of capacity data"""
    print("=== INVESTIGATING CAPACITY DATA ===\n")
    
    # Load the results
    results_file = 'Output_Data/UTOPIA_BASE_result.xlsx'
    
    # Read NewCapacity and TotCapacityAnn sheets
    new_capacity = pd.read_excel(results_file, sheet_name='NewCapacity')
    total_capacity = pd.read_excel(results_file, sheet_name='TotCapacityAnn')
    
    print("1. NEW CAPACITY ANALYSIS")
    print("-" * 30)
    
    # Check NewCapacity data
    print(f"NewCapacity shape: {new_capacity.shape}")
    print(f"Technologies: {sorted(new_capacity['TECHNOLOGY'].unique())}")
    print(f"Years: {sorted(new_capacity['YEAR'].unique())}")
    
    # Check non-zero new capacity investments
    new_cap_nonzero = new_capacity[new_capacity['VAR_VALUE'] > 0]
    print(f"\nNon-zero new capacity entries: {len(new_cap_nonzero)}")
    
    if len(new_cap_nonzero) > 0:
        print("\nNew capacity investments:")
        for _, row in new_cap_nonzero.iterrows():
            print(f"  {row['TECHNOLOGY']}: {row['VAR_VALUE']:.3f} in {row['YEAR']}")
    else:
        print("No new capacity investments found!")
    
    print("\n" + "="*50)
    print("2. TOTAL CAPACITY ANALYSIS")
    print("-" * 30)
    
    # Check TotalCapacity data
    print(f"TotCapacityAnn shape: {total_capacity.shape}")
    
    # Look at non-zero total capacities
    total_cap_nonzero = total_capacity[total_capacity['VAR_VALUE'] > 0]
    print(f"Non-zero total capacity entries: {len(total_cap_nonzero)}")
    
    if len(total_cap_nonzero) > 0:
        print("\nTotal capacity by technology (first few years):")
        
        # Group by technology and show capacity over years
        for tech in sorted(total_cap_nonzero['TECHNOLOGY'].unique()):
            tech_data = total_cap_nonzero[total_cap_nonzero['TECHNOLOGY'] == tech]
            capacities = tech_data.groupby('YEAR')['VAR_VALUE'].first().head(5)
            print(f"\n{tech}:")
            for year, cap in capacities.items():
                print(f"  {year}: {cap:.3f}")
    
    print("\n" + "="*50)
    print("3. DETAILED CAPACITY COMPARISON")
    print("-" * 30)
    
    # Check if all technologies really have the same capacity
    pivot_total = total_cap_nonzero.pivot(index='YEAR', columns='TECHNOLOGY', values='VAR_VALUE').fillna(0)
    
    if not pivot_total.empty:
        print("Capacity matrix (first 5 years, first 5 technologies):")
        print(pivot_total.iloc[:5, :5])
        
        # Check for identical patterns
        print("\nChecking for identical capacity patterns:")
        techs = pivot_total.columns
        for i, tech1 in enumerate(techs):
            for tech2 in techs[i+1:]:
                if (pivot_total[tech1] == pivot_total[tech2]).all():
                    print(f"  {tech1} and {tech2} have identical capacity patterns")
    
    print("\n" + "="*50)
    print("4. RESIDUAL CAPACITY INVESTIGATION")
    print("-" * 30)
    
    # Check if this is due to residual capacity
    # Look at technologies that have capacity but no new investments
    total_by_tech = total_cap_nonzero.groupby('TECHNOLOGY')['VAR_VALUE'].first()
    new_by_tech = new_cap_nonzero.groupby('TECHNOLOGY')['VAR_VALUE'].sum() if len(new_cap_nonzero) > 0 else pd.Series()
    
    print("Technologies with total capacity but no new investments:")
    for tech in total_by_tech.index:
        total_cap = total_by_tech[tech]
        new_cap = new_by_tech.get(tech, 0)
        if new_cap == 0 and total_cap > 0:
            print(f"  {tech}: Total={total_cap:.3f}, New=0 → Likely residual capacity")

if __name__ == "__main__":
    investigate_capacity_data()
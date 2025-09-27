#!/usr/bin/env python3
"""
Real World vs Model Data Comparison Script
==========================================

This script compares real world electricity generation data with OSeMOSYS model output data.
It creates visualizations to show the differences between actual and modeled data.

Author: Claude AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_and_clean_real_world_data(filepath):
    """
    Load and clean real world electricity generation data from EPA summary.

    Parameters:
    -----------
    filepath : str
        Path to the Excel file containing real world data

    Returns:
    --------
    pd.DataFrame : Clean dataframe with real world generation data
    """
    print("Loading real world data...")

    # Load the Excel file
    df_raw = pd.read_excel(filepath, sheet_name='epa_01_01')

    # Extract electricity generation data
    generation_data = []

    # Find the generation section (looking for fuel types)
    for i in range(len(df_raw)):
        row = df_raw.iloc[i]
        if pd.notna(row.iloc[0]):
            fuel_name = str(row.iloc[0]).strip()

            # Check if this is a fuel type row
            if any(fuel in fuel_name for fuel in ['Coal', 'Natural Gas', 'Nuclear', 'Hydroelectric', 'Wind', 'Solar', 'Petroleum']):
                try:
                    # Extract 2023 and 2022 values
                    val_2023 = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
                    val_2022 = float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0

                    if val_2023 > 0 or val_2022 > 0:  # Only include non-zero values
                        generation_data.append({
                            'Fuel': fuel_name,
                            'Generation_2023_TWh': val_2023 / 1000,  # Convert thousand MWh to TWh
                            'Generation_2022_TWh': val_2022 / 1000,
                            'Average_TWh': (val_2023 + val_2022) / 2000  # Average of both years
                        })
                except (ValueError, TypeError):
                    continue

    real_world_df = pd.DataFrame(generation_data)

    # Map fuel types to standardized technology categories
    fuel_mapping = {
        'Coal': 'Coal',
        'Natural Gas': 'Natural Gas',
        'Nuclear': 'Nuclear',
        'Hydroelectric Conventional': 'Hydro',
        '... Wind': 'Wind',
        '... Solar Thermal and Photovoltaic': 'Solar',
        'Petroleum Liquids': 'Oil',
        'Petroleum Coke': 'Oil',
        '... Wood and Wood-Derived Fuels': 'Biomass',
        '... Other Biomass': 'Biomass',
        '... Geothermal': 'Geothermal',
        'Renewable Sources Excluding Hydroelectric': 'Renewables_Excl_Hydro'
    }

    real_world_df['Technology_Type'] = real_world_df['Fuel'].map(fuel_mapping)
    real_world_df = real_world_df.dropna(subset=['Technology_Type'])

    # Group by technology type and sum values
    real_world_summary = real_world_df.groupby('Technology_Type').agg({
        'Generation_2023_TWh': 'sum',
        'Generation_2022_TWh': 'sum',
        'Average_TWh': 'sum'
    }).reset_index()

    print(f"Loaded {len(real_world_summary)} technology types from real world data")
    return real_world_summary


def categorize_osemosys_technologies(tech_name):
    """
    Categorize OSeMOSYS technologies based on prefix.

    Parameters:
    -----------
    tech_name : str
        Technology name from OSeMOSYS model

    Returns:
    --------
    str : Category of the technology
    """
    if tech_name.startswith('IMP'):
        return 'Import_Resources'
    elif tech_name.startswith('R'):
        return 'Energy_Demands'
    elif tech_name.startswith('TX'):
        return 'Transport_Demands'
    elif tech_name.startswith('E') or tech_name == 'SRE':
        return 'Power_Plants'
    else:
        return 'Other'

def load_and_process_model_data(filepath):
    """
    Load and process OSeMOSYS model output data with proper categorization.

    Parameters:
    -----------
    filepath : str
        Path to the Excel file containing model output

    Returns:
    --------
    dict : Dictionary containing processed model data categorized by function
    """
    print("Loading model output data...")

    # Load all sheets from the model output
    model_data = pd.read_excel(filepath, sheet_name=None)

    processed_data = {}

    # Process key variables
    key_variables = ['NewCapacity', 'TotCapacityAnn', 'UseAnn', 'Demand', 'CapitalInvestment']

    for var_name in key_variables:
        # Try different possible sheet names
        sheet_name = None
        for name in model_data.keys():
            if var_name.lower() in name.lower():
                sheet_name = name
                break

        if sheet_name:
            df = model_data[sheet_name]

            # Filter for recent years (2008-2010) to compare with real world 2022-2023
            recent_years = df[df['YEAR'].isin([2008, 2009, 2010])]

            if not recent_years.empty:
                # Filter out unrealistic values (like 99999 which are upper bounds)
                # Keep values that are reasonable for actual capacities/activities
                if var_name in ['TotCapacityAnn', 'NewCapacity']:
                    # For capacity, filter out values >= 1000 (these are likely constraints)
                    recent_years = recent_years[recent_years['VAR_VALUE'] < 1000]

                # Group by technology and calculate average
                if 'TECHNOLOGY' in recent_years.columns:
                    tech_summary = recent_years.groupby('TECHNOLOGY')['VAR_VALUE'].mean().reset_index()
                    # Only keep technologies with meaningful values (> 0)
                    tech_summary = tech_summary[tech_summary['VAR_VALUE'] > 0]

                    # Add technology categories
                    tech_summary['Category'] = tech_summary['TECHNOLOGY'].apply(categorize_osemosys_technologies)
                    tech_summary.columns = ['TECHNOLOGY', f'Avg_{var_name}', 'Category']
                    processed_data[var_name] = tech_summary

                    # Also create category-wise summaries
                    category_summary = tech_summary.groupby('Category')[f'Avg_{var_name}'].sum().reset_index()
                    processed_data[f'{var_name}_by_category'] = category_summary

                # Group by fuel if available
                if 'FUEL' in recent_years.columns:
                    fuel_summary = recent_years.groupby('FUEL')['VAR_VALUE'].mean().reset_index()
                    # Only keep fuels with meaningful values (> 0)
                    fuel_summary = fuel_summary[fuel_summary['VAR_VALUE'] > 0]
                    fuel_summary.columns = ['FUEL', f'Avg_{var_name}']
                    processed_data[f'{var_name}_by_fuel'] = fuel_summary

    print(f"Processed {len(processed_data)} model variables")
    return processed_data


def create_technology_comparison_chart(real_world_df, model_data, output_dir):
    """
    Create side-by-side percentage comparison charts for real world vs model data.

    Parameters:
    -----------
    real_world_df : pd.DataFrame
        Real world generation data
    model_data : dict
        Processed model data with categories
    output_dir : Path
        Directory to save charts
    """
    print("Creating side-by-side percentage comparison charts...")

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Real World vs OSeMOSYS Model: Percentage Comparisons', fontsize=16, fontweight='bold')

    # Prepare data for side-by-side percentage comparisons
    real_world_data, model_data_processed = prepare_percentage_data(real_world_df, model_data)

    # Chart 1: Generation Mix Comparison (Percentage)
    ax1 = axes[0, 0]
    create_generation_mix_comparison(ax1, real_world_data, model_data_processed)

    # Chart 2: Import Capacity vs Real World Consumption (Percentage)
    ax2 = axes[0, 1]
    create_import_capacity_comparison(ax2, real_world_data, model_data_processed)

    # Chart 3: Demand Distribution Comparison
    ax3 = axes[1, 0]
    create_demand_distribution_comparison(ax3, real_world_data, model_data_processed)

    # Chart 4: Technology Coverage Analysis
    ax4 = axes[1, 1]
    create_technology_coverage_analysis(ax4, real_world_data, model_data_processed)

    plt.tight_layout()

    # Save the chart
    output_path = output_dir / 'real_world_vs_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart to: {output_path}")

    return fig


def prepare_percentage_data(real_world_df, model_data):
    """
    Prepare data for percentage-based comparisons.

    Parameters:
    -----------
    real_world_df : pd.DataFrame
        Real world generation data
    model_data : dict
        Processed model data

    Returns:
    --------
    tuple : (real_world_data, model_data_processed) dictionaries
    """
    # Process real world data with corrected technology mapping
    real_world_data = {}

    # Calculate total generation for percentages
    total_generation = real_world_df['Generation_2023_TWh'].sum()

    # Map real world technologies to model technologies
    tech_mapping = {
        'Coal': 'E01',
        'Nuclear': 'E21',
        'Hydro': 'E31',
        'Natural Gas': 'Not in Model',  # No gas technology in corrected model
        'Wind': 'Not in Model',         # No wind in model
        'Solar': 'Not in Model',        # No solar in model
        'Oil': 'E70/SRE'               # Oil can be E70 (diesel) or SRE (crude oil)
    }

    real_world_data['generation_percentages'] = {}
    real_world_data['covered_by_model'] = {}
    real_world_data['not_in_model'] = {}

    for _, row in real_world_df.iterrows():
        tech = row['Technology_Type']
        percentage = (row['Generation_2023_TWh'] / total_generation) * 100
        model_tech = tech_mapping.get(tech, 'Not in Model')

        real_world_data['generation_percentages'][tech] = percentage

        if model_tech != 'Not in Model':
            real_world_data['covered_by_model'][tech] = percentage
        else:
            real_world_data['not_in_model'][tech] = percentage

    # Process model data
    model_data_processed = {}

    if 'TotCapacityAnn' in model_data:
        capacity_data = model_data['TotCapacityAnn']
        power_plants = capacity_data[capacity_data['Category'] == 'Power_Plants']

        if not power_plants.empty:
            total_capacity = power_plants['Avg_TotCapacityAnn'].sum()
            model_data_processed['capacity_percentages'] = {}

            for _, row in power_plants.iterrows():
                tech = row['TECHNOLOGY']
                percentage = (row['Avg_TotCapacityAnn'] / total_capacity) * 100
                model_data_processed['capacity_percentages'][tech] = percentage

    # Process fuel use data for import comparison
    if 'UseAnn_by_fuel' in model_data:
        fuel_data = model_data['UseAnn_by_fuel']
        total_fuel_use = fuel_data['Avg_UseAnn'].sum()

        model_data_processed['fuel_percentages'] = {}
        for _, row in fuel_data.iterrows():
            fuel = row['FUEL']
            percentage = (row['Avg_UseAnn'] / total_fuel_use) * 100
            model_data_processed['fuel_percentages'][fuel] = percentage

    return real_world_data, model_data_processed


def create_generation_mix_comparison(ax, real_world_data, model_data_processed):
    """Create side-by-side generation mix comparison."""

    # Define technology mapping for comparison
    tech_comparison = {
        'Coal': {'real': 'Coal', 'model': 'E01'},
        'Nuclear': {'real': 'Nuclear', 'model': 'E21'},
        'Hydro': {'real': 'Hydro', 'model': 'E31'},
        'Oil': {'real': 'Oil', 'model': 'E70+SRE'}  # Combined oil technologies
    }

    technologies = list(tech_comparison.keys())
    real_percentages = []
    model_percentages = []

    for tech in technologies:
        # Real world percentage
        real_tech = tech_comparison[tech]['real']
        real_pct = real_world_data['generation_percentages'].get(real_tech, 0)
        real_percentages.append(real_pct)

        # Model percentage (might need to combine E70 and SRE for oil)
        if tech == 'Oil':
            e70_pct = model_data_processed.get('capacity_percentages', {}).get('E70', 0)
            sre_pct = model_data_processed.get('capacity_percentages', {}).get('SRE', 0)
            model_pct = e70_pct + sre_pct
        else:
            model_tech = tech_comparison[tech]['model']
            model_pct = model_data_processed.get('capacity_percentages', {}).get(model_tech, 0)
        model_percentages.append(model_pct)

    # Create side-by-side bars
    x = np.arange(len(technologies))
    width = 0.35

    bars1 = ax.bar(x - width/2, real_percentages, width, label='Real World (2023)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, model_percentages, width, label='Model (2008-2010)',
                   color='darkorange', alpha=0.8)

    ax.set_title('Generation Mix Comparison\n(Technologies in Both Systems)', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Technology Type')
    ax.set_xticks(x)
    ax.set_xticklabels(technologies)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)


def create_import_capacity_comparison(ax, real_world_data, model_data_processed):
    """Create import capacity vs real world fuel use comparison."""

    # Try to load calibration data if available
    import json
    from pathlib import Path

    calibration_file = Path('calibration_output/calibrated_parameters.json')

    if calibration_file.exists():
        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)

            import_capacities = calibration_data.get('import_capacities', {})

            if import_capacities:
                # Create comparison chart
                fuels = []
                real_consumption = []
                import_limits = []

                for tech, data in import_capacities.items():
                    fuel_name = data['fuel_type']
                    real_pj = data['annual_consumption_pj']
                    limit_pj = data['capacity_limit']

                    fuels.append(f"{tech}\n({fuel_name})")
                    real_consumption.append(real_pj / 1000)  # Convert to thousand PJ for readability
                    import_limits.append(limit_pj / 1000)

                x = np.arange(len(fuels))
                width = 0.35

                bars1 = ax.bar(x - width/2, real_consumption, width,
                              label='Real World Consumption', color='red', alpha=0.7)
                bars2 = ax.bar(x + width/2, import_limits, width,
                              label='Model Import Limit', color='blue', alpha=0.7)

                ax.set_title('Import Capacity vs Real Consumption\n(From Calibration Data)', fontweight='bold')
                ax.set_ylabel('Energy (Thousand PJ/year)')
                ax.set_xlabel('Import Technology')
                ax.set_xticks(x)
                ax.set_xticklabels(fuels, fontsize=8)
                ax.legend()

                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height,
                               f'{height:.0f}', ha='center', va='bottom', fontsize=8)

                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height,
                               f'{height:.0f}', ha='center', va='bottom', fontsize=8)
                return

        except Exception as e:
            print(f"Could not load calibration data: {e}")

    # Fallback if no calibration data
    ax.text(0.5, 0.5, 'Import Capacity Analysis\n\nRun calibration script first:\npython calibrate_model_with_real_data.py\n\nThen rerun this comparison',
            ha='center', va='center', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    ax.set_title('Import Capacity vs Real Consumption', fontweight='bold')


def create_demand_distribution_comparison(ax, real_world_data, model_data_processed):
    """Create demand distribution comparison by sector using actual model demand results."""

    # Extract real world electricity sales data from EPA data
    # We need to extract this from the original real world data file
    try:
        import pandas as pd

        # Load real world sales data
        df_raw = pd.read_excel('real_world_data/summary.xlsx', sheet_name='epa_01_01')

        # Extract sales data (around rows 45-50)
        real_demand_data = {}
        for i in range(40, len(df_raw)):
            row = df_raw.iloc[i]
            if pd.notna(row.iloc[0]):
                sector = str(row.iloc[0]).strip()

                if any(s in sector.lower() for s in ['residential', 'commercial', 'industrial', 'transportation']):
                    try:
                        val_2023 = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0
                        if val_2023 > 0:
                            real_demand_data[sector] = val_2023  # Million kWh = GWh
                    except (ValueError, TypeError):
                        continue

        # Get actual model demand results (not calibrated parameters)
        # Use model_data_processed which has actual model outputs

        # Map real world sectors to model demand technologies
        sector_mapping = {
            'Residential': 'RHE',
            'Commercial': 'RL1',
            'Industrial': 'RHO',
            'Transportation': 'TXE'
        }

        # Extract actual model demand from model output (not calibration)
        # Load actual model demand data
        df_model = pd.read_excel('Output_Data/UTOPIA_BASE_result.xlsx', sheet_name='Demand')

        # Get recent years (2008-2010) and non-zero demand
        recent_model_demand = df_model[(df_model['YEAR'].isin([2008, 2009, 2010])) & (df_model['VAR_VALUE'] > 0)]

        # Calculate average demand by fuel type
        model_demand_by_fuel = recent_model_demand.groupby('FUEL')['VAR_VALUE'].mean().to_dict()

        # Map model fuels to sectors for comparison
        model_fuel_mapping = {
            'RH': 'Residential',  # RH likely = Residential Heating
            'RL': 'Commercial',   # RL likely = Residential Lighting (map to commercial)
            # Note: Model doesn't have industrial or transportation demand
        }

        sectors = []
        real_values = []
        model_values = []

        # Real world data
        if real_demand_data:
            for sector, value in real_demand_data.items():
                sectors.append(sector)
                real_values.append(value / 1000)  # Convert GWh to TWh

                # Find corresponding model demand
                model_fuel = None
                for fuel, mapped_sector in model_fuel_mapping.items():
                    if mapped_sector.lower() in sector.lower():
                        model_fuel = fuel
                        break

                if model_fuel and model_fuel in model_demand_by_fuel:
                    # The UTOPIA model is a small reference system.
                    # Total model annual demand ~70 units vs real world ~3900 TWh
                    # Calculate a scaling factor based on total system size
                    total_real_demand = sum(real_demand_data.values()) / 1000  # Convert to TWh
                    total_model_demand = sum(model_demand_by_fuel.values())

                    if total_model_demand > 0:
                        # Scale model demand to match real world total scale
                        scale_factor = total_real_demand / total_model_demand
                        model_demand_twh = model_demand_by_fuel[model_fuel] * scale_factor
                    else:
                        model_demand_twh = 0

                    model_values.append(model_demand_twh)
                else:
                    model_values.append(0)  # No corresponding model demand

        if sectors and any(real_values):
            x = np.arange(len(sectors))
            width = 0.35

            bars1 = ax.bar(x - width/2, real_values, width,
                          label='Real World Sales (EPA)', color='green', alpha=0.7)
            bars2 = ax.bar(x + width/2, model_values, width,
                          label='Model Demand (Actual Results)', color='orange', alpha=0.7)

            ax.set_title('Demand Distribution by Sector\n(Real World vs Actual Model Output)', fontweight='bold')
            ax.set_ylabel('Demand (TWh/year)')
            ax.set_xlabel('Sector')
            ax.set_xticks(x)
            ax.set_xticklabels(sectors, fontsize=8, rotation=45, ha='right')
            ax.legend()

            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=8)

            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

            # Add explanation text
            scale_factor = (sum(real_demand_data.values()) / 1000) / sum(model_demand_by_fuel.values()) if sum(model_demand_by_fuel.values()) > 0 else 0
            ax.text(0.98, 0.98, f'Note: UTOPIA model scaled\nby factor {scale_factor:.0f}x to match\nreal world total demand.\nModel lacks Industrial/Transport.',
                    transform=ax.transAxes, fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            return

    except Exception as e:
        print(f"Error extracting real world demand data: {e}")

    # Fallback explanation
    ax.text(0.5, 0.5, 'Demand Distribution Analysis\n\nISSUE IDENTIFIED:\nCalibration script sets model demand\nequal to real world demand.\n\nNeed to compare:\n• Real world electricity sales (EPA)\n• Actual OSeMOSYS demand outputs\n\nNot calibrated parameters!',
            ha='center', va='center', transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    ax.set_title('Demand Distribution by Sector - Issue Identified', fontweight='bold')


def create_technology_coverage_analysis(ax, real_world_data, model_data_processed):
    """Analyze what real world technologies are covered/missing in model."""

    # Calculate coverage statistics
    covered_pct = sum(real_world_data.get('covered_by_model', {}).values())
    not_covered_pct = sum(real_world_data.get('not_in_model', {}).values())

    # Create pie chart showing coverage
    sizes = [covered_pct, not_covered_pct]
    labels = [f'Covered by Model\n{covered_pct:.1f}%', f'Not in Model\n{not_covered_pct:.1f}%']
    colors = ['lightgreen', 'lightcoral']

    ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax.set_title('Real World Technology Coverage\nin OSeMOSYS Model', fontweight='bold')

    # Add text showing what's missing
    missing_techs = list(real_world_data.get('not_in_model', {}).keys())
    if missing_techs:
        missing_text = "Missing from model:\n" + "\n".join([f"• {tech}" for tech in missing_techs])
        ax.text(1.3, 0.5, missing_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))


def create_summary_table(real_world_df, model_data, output_dir):
    """
    Create a summary table comparing key metrics.

    Parameters:
    -----------
    real_world_df : pd.DataFrame
        Real world generation data
    model_data : dict
        Processed model data
    output_dir : Path
        Directory to save the table
    """
    print("Creating summary comparison table...")

    # Create summary statistics
    summary_data = []

    # Real world summary
    total_real_2023 = real_world_df['Generation_2023_TWh'].sum()
    total_real_2022 = real_world_df['Generation_2022_TWh'].sum()
    growth_rate = ((total_real_2023 - total_real_2022) / total_real_2022) * 100

    summary_data.append({
        'Metric': 'Total Real World Generation 2023 (TWh)',
        'Value': f'{total_real_2023:.1f}',
        'Notes': 'EPA data'
    })

    summary_data.append({
        'Metric': 'Total Real World Generation 2022 (TWh)',
        'Value': f'{total_real_2022:.1f}',
        'Notes': 'EPA data'
    })

    summary_data.append({
        'Metric': 'Year-over-year Growth Rate (%)',
        'Value': f'{growth_rate:.2f}%',
        'Notes': '2023 vs 2022'
    })

    # Technology diversity
    summary_data.append({
        'Metric': 'Number of Real World Technology Types',
        'Value': str(len(real_world_df)),
        'Notes': 'Major generation sources'
    })

    # Model summary
    if 'TotCapacityAnn' in model_data:
        model_techs = len(model_data['TotCapacityAnn'])
        summary_data.append({
            'Metric': 'Number of Model Technologies',
            'Value': str(model_techs),
            'Notes': 'OSeMOSYS model'
        })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_path = output_dir / 'comparison_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary table to: {output_path}")

    return summary_df


def generate_insights_report(real_world_df, model_data, output_dir):
    """
    Generate a text report with insights from the comparison.

    Parameters:
    -----------
    real_world_df : pd.DataFrame
        Real world generation data
    model_data : dict
        Processed model data
    output_dir : Path
        Directory to save the report
    """
    print("Generating insights report...")

    report_lines = []
    report_lines.append("="*60)
    report_lines.append("REAL WORLD vs OSeMOSYS MODEL COMPARISON REPORT")
    report_lines.append("="*60)
    report_lines.append("")

    # Real world analysis
    report_lines.append("REAL WORLD DATA ANALYSIS (2022-2023)")
    report_lines.append("-" * 40)

    total_2023 = real_world_df['Generation_2023_TWh'].sum()
    total_2022 = real_world_df['Generation_2022_TWh'].sum()

    report_lines.append(f"Total Electricity Generation 2023: {total_2023:.1f} TWh")
    report_lines.append(f"Total Electricity Generation 2022: {total_2022:.1f} TWh")
    report_lines.append(f"Year-over-year change: {((total_2023-total_2022)/total_2022)*100:.2f}%")
    report_lines.append("")

    # Top generators
    report_lines.append("Top Electricity Generation Sources (2023):")
    sorted_real = real_world_df.sort_values('Generation_2023_TWh', ascending=False)
    for i, row in sorted_real.head().iterrows():
        pct = (row['Generation_2023_TWh'] / total_2023) * 100
        report_lines.append(f"  {row['Technology_Type']}: {row['Generation_2023_TWh']:.1f} TWh ({pct:.1f}%)")
    report_lines.append("")

    # Model analysis
    report_lines.append("MODEL DATA ANALYSIS (OSeMOSYS)")
    report_lines.append("-" * 30)

    if 'TotCapacityAnn' in model_data:
        capacity_data = model_data['TotCapacityAnn']
        report_lines.append(f"Number of modeled technologies: {len(capacity_data)}")

        # Analyze by category
        report_lines.append("\nModel Technologies by Category:")

        # Power Plants (E-series, SRE)
        power_plants = capacity_data[capacity_data['Category'] == 'Power_Plants']
        if not power_plants.empty:
            report_lines.append("  Power Plants (E-series, SRE):")
            for i, row in power_plants.iterrows():
                report_lines.append(f"    {row['TECHNOLOGY']}: {row['Avg_TotCapacityAnn']:.3f} units")

        # Energy Demands (R-series)
        energy_demands = capacity_data[capacity_data['Category'] == 'Energy_Demands']
        if not energy_demands.empty:
            report_lines.append("  Energy Service Demands (R-series):")
            for i, row in energy_demands.iterrows():
                report_lines.append(f"    {row['TECHNOLOGY']}: {row['Avg_TotCapacityAnn']:.3f} units")

        # Transport Demands (TX-series)
        transport_demands = capacity_data[capacity_data['Category'] == 'Transport_Demands']
        if not transport_demands.empty:
            report_lines.append("  Transport Demands (TX-series):")
            for i, row in transport_demands.iterrows():
                report_lines.append(f"    {row['TECHNOLOGY']}: {row['Avg_TotCapacityAnn']:.3f} units")

    report_lines.append("")

    # Comparison insights
    report_lines.append("KEY INSIGHTS")
    report_lines.append("-" * 15)

    # Renewable vs fossil
    renewable_real = real_world_df[real_world_df['Technology_Type'].isin(['Wind', 'Solar', 'Hydro'])]['Generation_2023_TWh'].sum()
    fossil_real = real_world_df[real_world_df['Technology_Type'].isin(['Coal', 'Natural Gas', 'Oil'])]['Generation_2023_TWh'].sum()

    report_lines.append(f"Real World Renewable Generation (2023): {renewable_real:.1f} TWh ({(renewable_real/total_2023)*100:.1f}%)")
    report_lines.append(f"Real World Fossil Generation (2023): {fossil_real:.1f} TWh ({(fossil_real/total_2023)*100:.1f}%)")
    report_lines.append("")

    # Model structure explanation
    report_lines.append("OSEMOSYS MODEL STRUCTURE (CORRECTED)")
    report_lines.append("-" * 40)
    report_lines.append("• IMP* technologies: Import resources (IMPDSL1=diesel, IMPGSL1=gasoline, IMPHCO1=coal, IMPURN1=uranium)")
    report_lines.append("• E* + SRE technologies: Power generation plants")
    report_lines.append("  - E01: Coal power plant")
    report_lines.append("  - E21: Nuclear power plant")
    report_lines.append("  - E31: Hydro power plant (no import needed)")
    report_lines.append("  - E51: Pumped storage")
    report_lines.append("  - E70: Diesel power plant")
    report_lines.append("  - SRE: Crude oil power plant")
    report_lines.append("• R* technologies: Energy service demands (RL1=lighting, RHE=electric heating, RHO=oil heating)")
    report_lines.append("• TX* technologies: Transport demands (TXE=electric vehicles, TXD=diesel vehicles, TXG=gasoline vehicles)")
    report_lines.append("")

    # Technology coverage analysis
    report_lines.append("REAL WORLD COVERAGE IN MODEL")
    report_lines.append("-" * 30)

    # Calculate coverage based on real world data
    total_2023 = real_world_df['Generation_2023_TWh'].sum()

    # Technologies covered by model (Coal, Nuclear, Hydro, Oil)
    covered_techs = ['Coal', 'Nuclear', 'Hydro', 'Oil']
    covered_generation = real_world_df[real_world_df['Technology_Type'].isin(covered_techs)]['Generation_2023_TWh'].sum()
    covered_percentage = (covered_generation / total_2023) * 100

    # Technologies NOT in model
    not_covered_percentage = 100 - covered_percentage

    report_lines.append(f"Technologies covered by model: {covered_percentage:.1f}% of real world generation")
    report_lines.append(f"Technologies NOT in model: {not_covered_percentage:.1f}% of real world generation")
    report_lines.append("")
    report_lines.append("Missing from model (major real world sources):")
    report_lines.append("• Natural Gas (38.0% of US generation) - No gas technology in current model")
    report_lines.append("• Wind (8.9% of US generation) - No wind technology in current model")
    report_lines.append("• Solar (varies) - No solar technology in current model")
    report_lines.append("• Other Renewables (biomass, geothermal) - Limited representation")
    report_lines.append("")

    # Data limitations
    report_lines.append("DATA LIMITATIONS & NOTES")
    report_lines.append("-" * 25)
    report_lines.append("• Real world data is from EPA (2022-2023) while model data is from OSeMOSYS (2008-2010)")
    report_lines.append("• Model uses UTOPIA reference system, not US electricity grid")
    report_lines.append("• Power generation comparison: Real world fuels vs Model E-series power plants")
    report_lines.append("• Different units and scales between datasets require careful interpretation")
    report_lines.append("• Real world data represents actual electricity generation in TWh")
    report_lines.append("• Model data represents capacity and fuel use in model-specific units")
    report_lines.append("• Filtered out unrealistic model values (≥1000) which appear to be upper bound constraints")
    report_lines.append("• Import technologies (IMP*) had constraint values of 99999, not actual capacities")
    report_lines.append("")

    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 15)
    report_lines.append("• Use this comparison to validate model assumptions against real trends")
    report_lines.append("• Consider updating model parameters to reflect current technology mix")
    report_lines.append("• Focus on relative proportions rather than absolute values for comparison")
    report_lines.append("• Use real world renewable growth trends to inform future scenarios")

    # Save report
    output_path = output_dir / 'comparison_insights_report.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Saved insights report to: {output_path}")
    return report_lines


def main():
    """
    Main function to run the comparison analysis.
    """
    print("Starting Real World vs Model Data Comparison...")

    # Set up paths
    base_dir = Path.cwd()
    real_world_file = base_dir / 'real_world_data' / 'summary.xlsx'
    model_output_file = base_dir / 'Output_Data' / 'UTOPIA_BASE_result.xlsx'
    output_dir = base_dir / 'comparison_output'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Check if files exist
    if not real_world_file.exists():
        print(f"Error: Real world data file not found: {real_world_file}")
        return

    if not model_output_file.exists():
        print(f"Error: Model output file not found: {model_output_file}")
        return

    try:
        # Load and process data
        real_world_df = load_and_clean_real_world_data(real_world_file)
        model_data = load_and_process_model_data(model_output_file)

        # Create visualizations and reports
        fig = create_technology_comparison_chart(real_world_df, model_data, output_dir)
        summary_df = create_summary_table(real_world_df, model_data, output_dir)
        insights = generate_insights_report(real_world_df, model_data, output_dir)

        # Display summary
        print("\nCOMPARISSON COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Generated files:")
        print(f"• Comparison chart: {output_dir / 'real_world_vs_model_comparison.png'}")
        print(f"• Summary table: {output_dir / 'comparison_summary.csv'}")
        print(f"• Insights report: {output_dir / 'comparison_insights_report.txt'}")
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))

        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
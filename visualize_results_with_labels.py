#!/usr/bin/env python3
"""
OSeMOSYS-PuLP Results Visualization with Descriptive Technology Labels
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Technology name mapping with descriptions
TECHNOLOGY_LABELS = {
    # Power Generation Technologies
    'E01': 'E01 - Coal Power Plant',
    'E21': 'E21 - Gas Power Plant', 
    'E31': 'E31 - Oil Power Plant',
    'E51': 'E51 - Nuclear Power Plant',
    'E70': 'E70 - Hydro Power Plant',
    
    # Renewable Energy Technologies
    'RHE': 'RHE - Renewable Electricity',
    'RHO': 'RHO - Renewable Heating',
    'SRE': 'SRE - Solar Electricity',
    
    # Lighting and Residential
    'RL1': 'RL1 - Renewable Lighting',
    
    # Transmission and Distribution
    'TXD': 'TXD - Distribution Network',
    'TXE': 'TXE - Electricity Transmission',
    'TXG': 'TXG - Gas Transmission',
    
    # Import Technologies (usually unlimited capacity)
    'IMPDSL1': 'IMPDSL1 - Diesel Import',
    'IMPGSL1': 'IMPGSL1 - Gasoline Import',
    'IMPHCO1': 'IMPHCO1 - Heavy Oil Import',
    'IMPOIL1': 'IMPOIL1 - Crude Oil Import',
    'IMPURN1': 'IMPURN1 - Uranium Import',
    
    # Resource Technologies (usually unlimited capacity)
    'RIV': 'RIV - River/Hydro Resource',
    'Rhu': 'Rhu - Heating Resource',
    'Rlu': 'Rlu - Lighting Resource',
    'Txu': 'Txu - Transmission Utility',
}

# Fuel name mapping
FUEL_LABELS = {
    'DSL': 'Diesel',
    'ELC': 'Electricity', 
    'GSL': 'Gasoline',
    'HCO': 'Heavy Oil',
    'HYD': 'Hydrogen',
    'OIL': 'Crude Oil',
    'RH': 'Residential Heating',
    'RL': 'Residential Lighting',
    'TX': 'Transmission',
    'URN': 'Uranium'
}

# Emission labels
EMISSION_LABELS = {
    'CO2': 'Carbon Dioxide (CO2)',
    'NOx': 'Nitrogen Oxides (NOx)',
    'SO2': 'Sulfur Dioxide (SO2)',
}

def get_technology_label(tech_code):
    """Get descriptive label for technology code"""
    return TECHNOLOGY_LABELS.get(tech_code, f'{tech_code} - Unknown Technology')

def get_fuel_label(fuel_code):
    """Get descriptive label for fuel code"""
    return FUEL_LABELS.get(fuel_code, f'{fuel_code} - Unknown Fuel')

def get_emission_label(emission_code):
    """Get descriptive label for emission code"""
    return EMISSION_LABELS.get(emission_code, f'{emission_code} - Unknown Emission')

def load_results(file_path):
    """Load all sheets from the Excel results file"""
    print(f"Loading results from: {file_path}")
    
    xl_file = pd.ExcelFile(file_path)
    sheet_names = xl_file.sheet_names
    print(f"Available sheets: {sheet_names}")
    
    results = {}
    for sheet in sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            results[sheet] = df
            print(f"  - {sheet}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error loading {sheet}: {e}")
    
    return results

def filter_meaningful_technologies(df):
    """Filter out technologies with artificially high capacities (99999)"""
    unlimited_techs = ['IMPDSL1', 'IMPGSL1', 'IMPHCO1', 'IMPOIL1', 'IMPURN1', 
                      'RIV', 'Rhu', 'Rlu', 'Txu']
    
    if 'TECHNOLOGY' in df.columns:
        meaningful_df = df[~df['TECHNOLOGY'].isin(unlimited_techs)].copy()
        print(f"Filtered from {len(df)} to {len(meaningful_df)} rows (removed unlimited capacity techs)")
        return meaningful_df
    return df

def add_technology_labels(df):
    """Add descriptive labels for technologies"""
    if 'TECHNOLOGY' in df.columns:
        df = df.copy()
        df['TECH_LABEL'] = df['TECHNOLOGY'].apply(get_technology_label)
        return df
    return df

def add_fuel_labels(df):
    """Add descriptive labels for fuels"""
    if 'FUEL' in df.columns:
        df = df.copy()
        df['FUEL_LABEL'] = df['FUEL'].apply(get_fuel_label)
        return df
    return df

def plot_meaningful_technology_capacity(results, output_dir):
    """Plot new capacity for meaningful technologies with descriptive labels"""
    if 'NewCapacity' not in results:
        print("NewCapacity data not found")
        return
    
    df = results['NewCapacity']
    if df.empty:
        print("NewCapacity data is empty")
        return
    
    # Filter meaningful technologies and non-zero values
    df_filtered = filter_meaningful_technologies(df)
    df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0].copy()
    df_filtered = add_technology_labels(df_filtered)
    
    if df_filtered.empty:
        print("No meaningful new capacity investments found")
        return
    
    plt.figure(figsize=(16, 10))
    
    # Group by technology label and year
    capacity_by_tech = df_filtered.groupby(['TECH_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
    
    # Pivot for plotting
    pivot_data = capacity_by_tech.pivot(index='YEAR', columns='TECH_LABEL', values='VAR_VALUE').fillna(0)
    
    # Create stacked area plot
    ax = pivot_data.plot(kind='area', stacked=True, figsize=(16, 10), alpha=0.7)
    
    plt.title('New Capacity Investment Over Time\n(Meaningful Technologies Only)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('New Capacity (GW)', fontsize=12)
    plt.legend(title='Technology', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'meaningful_technology_capacity_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_meaningful_total_capacity(results, output_dir):
    """Plot total capacity for meaningful technologies with descriptive labels"""
    if 'TotCapacityAnn' not in results:
        print("TotCapacityAnn data not found")
        return
    
    df = results['TotCapacityAnn']
    if df.empty:
        print("TotCapacityAnn data is empty")
        return
    
    # Filter meaningful technologies and non-zero values
    df_filtered = filter_meaningful_technologies(df)
    df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0].copy()
    df_filtered = add_technology_labels(df_filtered)
    
    if df_filtered.empty:
        print("No meaningful total capacity found")
        return
    
    plt.figure(figsize=(16, 10))
    
    # Group by technology label and year
    capacity_by_tech = df_filtered.groupby(['TECH_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
    
    # Pivot for plotting
    pivot_data = capacity_by_tech.pivot(index='YEAR', columns='TECH_LABEL', values='VAR_VALUE').fillna(0)
    
    # Create line plot
    ax = pivot_data.plot(kind='line', figsize=(16, 10), marker='o', linewidth=2, markersize=4)
    
    plt.title('Total Technology Capacity Over Time\n(Meaningful Technologies Only)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Capacity (GW)', fontsize=12)
    plt.legend(title='Technology', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'meaningful_total_capacity_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_capacity_by_category_labeled(results, output_dir):
    """Plot capacity by technology category with better labels"""
    if 'TotCapacityAnn' not in results:
        return
    
    df = results['TotCapacityAnn']
    if df.empty:
        return
    
    # Filter meaningful technologies
    df_filtered = filter_meaningful_technologies(df)
    df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0].copy()
    
    if df_filtered.empty:
        return
    
    # Enhanced categorization with better names
    tech_categories = {
        'Fossil Power Plants': ['E01', 'E21', 'E31'],  # Coal, Gas, Oil
        'Clean Power Plants': ['E51', 'E70'],         # Nuclear, Hydro
        'Renewable Energy': ['RHE', 'SRE'],           # Renewable electricity, Solar
        'Renewable Heating': ['RHO'],                 # Renewable heating
        'Renewable Lighting': ['RL1'],                # Renewable lighting
        'Transmission & Distribution': ['TXD', 'TXE', 'TXG']  # Distribution, transmission
    }
    
    # Add category column
    def categorize_tech(tech):
        for category, techs in tech_categories.items():
            if tech in techs:
                return category
        return 'Other Technologies'
    
    df_filtered['CATEGORY'] = df_filtered['TECHNOLOGY'].apply(categorize_tech)
    
    # Group by category and year
    capacity_by_cat = df_filtered.groupby(['CATEGORY', 'YEAR'])['VAR_VALUE'].sum().reset_index()
    
    # Pivot for plotting
    pivot_data = capacity_by_cat.pivot(index='YEAR', columns='CATEGORY', values='VAR_VALUE').fillna(0)
    
    if not pivot_data.empty:
        plt.figure(figsize=(14, 8))
        ax = pivot_data.plot(kind='area', stacked=True, figsize=(14, 8), alpha=0.7)
        
        plt.title('Total Capacity by Technology Category Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Capacity (GW)', fontsize=12)
        plt.legend(title='Technology Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_dir / 'capacity_by_category_labeled.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_demand_and_use_labeled(results, output_dir):
    """Plot demand vs usage by fuel type with descriptive labels"""
    demand_data = results.get('Demand')
    use_data = results.get('UseAnn')
    
    if demand_data is None or use_data is None:
        print("Demand or UseAnn data not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Energy Demand and Use Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Total demand by fuel over time
    if not demand_data.empty:
        demand_filtered = demand_data[demand_data['VAR_VALUE'] > 0].copy()
        demand_filtered = add_fuel_labels(demand_filtered)
        
        if not demand_filtered.empty:
            demand_by_fuel = demand_filtered.groupby(['FUEL_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
            demand_pivot = demand_by_fuel.pivot(index='YEAR', columns='FUEL_LABEL', values='VAR_VALUE').fillna(0)
            
            if not demand_pivot.empty:
                demand_pivot.plot(kind='area', stacked=True, ax=axes[0,0], alpha=0.7)
                axes[0,0].set_title('Energy Demand by Fuel Type Over Time')
                axes[0,0].set_ylabel('Demand (PJ)')
                axes[0,0].legend(title='Fuel Type', bbox_to_anchor=(1.05, 1), fontsize=9)
                axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Total use by fuel over time
    if not use_data.empty:
        use_filtered = use_data[use_data['VAR_VALUE'] > 0].copy()
        use_filtered = add_fuel_labels(use_filtered)
        
        if not use_filtered.empty:
            use_by_fuel = use_filtered.groupby(['FUEL_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
            use_pivot = use_by_fuel.pivot(index='YEAR', columns='FUEL_LABEL', values='VAR_VALUE').fillna(0)
            
            if not use_pivot.empty:
                use_pivot.plot(kind='area', stacked=True, ax=axes[0,1], alpha=0.7)
                axes[0,1].set_title('Energy Use by Fuel Type Over Time')
                axes[0,1].set_ylabel('Use (PJ)')
                axes[0,1].legend(title='Fuel Type', bbox_to_anchor=(1.05, 1), fontsize=9)
                axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3 & 4: Demand vs Use comparison for key fuels
    if not demand_data.empty and not use_data.empty:
        demand_total = demand_data[demand_data['VAR_VALUE'] > 0].groupby(['FUEL', 'YEAR'])['VAR_VALUE'].sum()
        use_total = use_data[use_data['VAR_VALUE'] > 0].groupby(['FUEL', 'YEAR'])['VAR_VALUE'].sum()
        
        # Plot for Electricity
        if 'ELC' in demand_total.index.get_level_values(0):
            elc_demand = demand_total.loc['ELC'] if 'ELC' in demand_total.index.get_level_values(0) else pd.Series()
            elc_use = use_total.loc['ELC'] if 'ELC' in use_total.index.get_level_values(0) else pd.Series()
            
            if not elc_demand.empty or not elc_use.empty:
                years = sorted(set(elc_demand.index) | set(elc_use.index))
                demand_vals = [elc_demand.get(year, 0) for year in years]
                use_vals = [elc_use.get(year, 0) for year in years]
                
                axes[1,0].plot(years, demand_vals, marker='o', label='Demand', linewidth=2, markersize=5)
                axes[1,0].plot(years, use_vals, marker='s', label='Use', linewidth=2, markersize=5)
                axes[1,0].set_title('Electricity - Demand vs Use')
                axes[1,0].set_ylabel('Energy (PJ)')
                axes[1,0].set_xlabel('Year')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
        
        # Plot for Residential Heating
        if 'RH' in demand_total.index.get_level_values(0):
            rh_demand = demand_total.loc['RH'] if 'RH' in demand_total.index.get_level_values(0) else pd.Series()
            rh_use = use_total.loc['RH'] if 'RH' in use_total.index.get_level_values(0) else pd.Series()
            
            if not rh_demand.empty or not rh_use.empty:
                years = sorted(set(rh_demand.index) | set(rh_use.index))
                demand_vals = [rh_demand.get(year, 0) for year in years]
                use_vals = [rh_use.get(year, 0) for year in years]
                
                axes[1,1].plot(years, demand_vals, marker='o', label='Demand', linewidth=2, markersize=5)
                axes[1,1].plot(years, use_vals, marker='s', label='Use', linewidth=2, markersize=5)
                axes[1,1].set_title('Residential Heating - Demand vs Use')
                axes[1,1].set_ylabel('Energy (PJ)')
                axes[1,1].set_xlabel('Year')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demand_and_use_analysis_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_capital_investment_labeled(results, output_dir):
    """Plot capital investment with descriptive technology labels"""
    if 'CapitalInvestment' not in results:
        print("CapitalInvestment data not found")
        return
    
    df = results['CapitalInvestment']
    if df.empty:
        print("CapitalInvestment data is empty")
        return
    
    # Filter meaningful technologies and non-zero values
    df_filtered = filter_meaningful_technologies(df)
    df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0].copy()
    df_filtered = add_technology_labels(df_filtered)
    
    if df_filtered.empty:
        print("No meaningful capital investment found")
        return
    
    plt.figure(figsize=(16, 10))
    
    # Group by technology label and year
    investment_by_tech = df_filtered.groupby(['TECH_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
    
    # Pivot for plotting
    pivot_data = investment_by_tech.pivot(index='YEAR', columns='TECH_LABEL', values='VAR_VALUE').fillna(0)
    
    # Create stacked bar plot
    ax = pivot_data.plot(kind='bar', stacked=True, figsize=(16, 10))
    
    plt.title('Capital Investment by Technology and Year', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Capital Investment (Million $)', fontsize=12)
    plt.legend(title='Technology', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'capital_investment_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard_labeled(results, output_dir):
    """Create a comprehensive dashboard with descriptive labels"""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    
    # 1. Technology capacity evolution
    if 'TotCapacityAnn' in results:
        ax1 = fig.add_subplot(gs[0, :2])
        df = results['TotCapacityAnn']
        if not df.empty:
            df_filtered = filter_meaningful_technologies(df)
            df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0].copy()
            df_filtered = add_technology_labels(df_filtered)
            
            if not df_filtered.empty:
                capacity_by_tech = df_filtered.groupby(['TECH_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
                pivot_data = capacity_by_tech.pivot(index='YEAR', columns='TECH_LABEL', values='VAR_VALUE').fillna(0)
                
                if not pivot_data.empty:
                    pivot_data.plot(kind='line', ax=ax1, marker='o', linewidth=2, markersize=3)
                    ax1.set_title('Total Capacity Evolution by Technology', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Capacity (GW)')
                    ax1.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
                    ax1.grid(True, alpha=0.3)
    
    # 2. Technology categories
    if 'TotCapacityAnn' in results:
        ax2 = fig.add_subplot(gs[0, 2])
        df = results['TotCapacityAnn']
        if not df.empty:
            df_filtered = filter_meaningful_technologies(df)
            df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0]
            
            # Final year capacity by category
            tech_categories = {
                'Fossil Power': ['E01', 'E21', 'E31'],
                'Clean Power': ['E51', 'E70'],
                'Renewables': ['RHE', 'SRE', 'RHO', 'RL1'],
                'Transmission': ['TXD', 'TXE', 'TXG']
            }
            
            def categorize_tech(tech):
                for category, techs in tech_categories.items():
                    if tech in techs:
                        return category
                return 'Other'
            
            df_filtered['CATEGORY'] = df_filtered['TECHNOLOGY'].apply(categorize_tech)
            final_year = df_filtered['YEAR'].max()
            final_capacity = df_filtered[df_filtered['YEAR'] == final_year].groupby('CATEGORY')['VAR_VALUE'].sum()
            
            if not final_capacity.empty:
                final_capacity.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
                ax2.set_title(f'Capacity Mix in {final_year}', fontsize=12, fontweight='bold')
                ax2.set_ylabel('')
    
    # 3. Energy demand trends
    if 'Demand' in results:
        ax3 = fig.add_subplot(gs[1, :])
        df = results['Demand']
        if not df.empty:
            demand_filtered = df[df['VAR_VALUE'] > 0].copy()
            demand_filtered = add_fuel_labels(demand_filtered)
            
            if not demand_filtered.empty:
                demand_by_fuel = demand_filtered.groupby(['FUEL_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
                demand_pivot = demand_by_fuel.pivot(index='YEAR', columns='FUEL_LABEL', values='VAR_VALUE').fillna(0)
                
                if not demand_pivot.empty:
                    demand_pivot.plot(kind='area', stacked=True, ax=ax3, alpha=0.7)
                    ax3.set_title('Energy Demand by Fuel Type', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Demand (PJ)')
                    ax3.legend(bbox_to_anchor=(1.05, 1), fontsize=9)
                    ax3.grid(True, alpha=0.3)
    
    # 4. Emissions
    if 'AnnEmissions' in results:
        ax4 = fig.add_subplot(gs[2, :2])
        df = results['AnnEmissions']
        if not df.empty:
            emissions_filtered = df[df['VAR_VALUE'] > 0]
            if not emissions_filtered.empty:
                emissions_by_year = emissions_filtered.groupby('YEAR')['VAR_VALUE'].sum()
                if not emissions_by_year.empty:
                    emissions_by_year.plot(kind='line', ax=ax4, marker='o', linewidth=3, markersize=6, color='red')
                    ax4.set_title('Annual CO₂ Emissions', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Emissions (Mt CO₂)')
                    ax4.grid(True, alpha=0.3)
    
    # 5. System cost
    # if 'cost' in results:
    #     ax5 = fig.add_subplot(gs[2, 2])
    #     df = results['cost']
    #     if not df.empty:
    #         total_cost = df['VAR_VALUE'].iloc[0]
    #         ax5.text(0.5, 0.5, f'Total System Cost:\n${total_cost:,.0f} Million', 
    #                 transform=ax5.transAxes, ha='center', va='center', 
    #                 fontsize=12, fontweight='bold',
    #                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    #         ax5.set_xlim(0, 1)
    #         ax5.set_ylim(0, 1)
    #         ax5.set_xticks([])
    #         ax5.set_yticks([])
    #         ax5.set_title('Economic Analysis', fontsize=12, fontweight='bold')
    
    # 6. Investment trends
    if 'CapitalInvestment' in results:
        ax6 = fig.add_subplot(gs[3, :])
        df = results['CapitalInvestment']
        if not df.empty:
            df_filtered = filter_meaningful_technologies(df)
            df_filtered = df_filtered[df_filtered['VAR_VALUE'] > 0].copy()
            df_filtered = add_technology_labels(df_filtered)
            
            if not df_filtered.empty:
                investment_by_tech = df_filtered.groupby(['TECH_LABEL', 'YEAR'])['VAR_VALUE'].sum().reset_index()
                investment_pivot = investment_by_tech.pivot(index='YEAR', columns='TECH_LABEL', values='VAR_VALUE').fillna(0)
                
                if not investment_pivot.empty:
                    investment_pivot.plot(kind='bar', stacked=True, ax=ax6)
                    ax6.set_title('Capital Investment by Technology and Year', fontsize=12, fontweight='bold')
                    ax6.set_ylabel('Investment (Million $)')
                    ax6.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
                    ax6.tick_params(axis='x', rotation=45)
    
    plt.suptitle('OSeMOSYS-PuLP UTOPIA Energy System Analysis Dashboard', fontsize=18, fontweight='bold')
    plt.savefig(output_dir / 'comprehensive_dashboard_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the labeled visualization"""
    base_dir = Path(__file__).parent
    results_file = base_dir / 'Output_Data' / 'UTOPIA_BASE_result.xlsx'
    output_dir = base_dir / 'visualizations'
    
    output_dir.mkdir(exist_ok=True)
    
    print("OSeMOSYS-PuLP Results Visualization with Descriptive Labels")
    print("=" * 65)
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        return
    
    try:
        # Load results
        results = load_results(results_file)
        
        if not results:
            print("No data loaded from results file")
            return
        
        print(f"\n=== GENERATING LABELED VISUALIZATIONS ===")
        print(f"Saving plots to: {output_dir}")
        
        # Generate visualizations with descriptive labels
        plot_meaningful_technology_capacity(results, output_dir)
        plot_meaningful_total_capacity(results, output_dir)
        plot_capacity_by_category_labeled(results, output_dir)
        plot_demand_and_use_labeled(results, output_dir)
        plot_capital_investment_labeled(results, output_dir)
        create_comprehensive_dashboard_labeled(results, output_dir)
        
        print(f"\nLabeled visualization complete! Check the '{output_dir}' folder for new plots.")
        print("\nNew files created with '_labeled' suffix contain descriptive technology names.")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
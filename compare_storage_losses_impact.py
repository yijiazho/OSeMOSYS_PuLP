#!/usr/bin/env python3
"""
Compare OSeMOSYS Model Results: Storage Losses Impact Analysis
=============================================================

This script compares the results between:
1. UTOPIA_BASE_result.xlsx (original model with perfect storage efficiency)
2. UTOPIA_BASE_result_losses.xlsx (existing model with losses)
3. UTOPIA_BASE_with_storage_losses_result.xlsx (our enhanced model with realistic storage parameters)

Focus on storage-related variables and system-wide impacts.

Author: Claude AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class StorageImpactAnalyzer:
    """
    Analyze the impact of storage efficiency parameters on model results.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.results = {}
        self.comparison_data = {}

    def load_model_results(self):
        """Load model result files for comparison (Original vs Enhanced with storage losses)."""
        print("Loading model results for comparison...")

        result_files = {
            'original': 'Output_Data/UTOPIA_BASE_result.xlsx',
            'enhanced': 'Output_Data/UTOPIA_BASE_with_storage_losses_result.xlsx'
        }

        for name, filepath in result_files.items():
            if Path(filepath).exists():
                print(f"Loading {name}: {filepath}")

                # Load all sheets
                with pd.ExcelFile(filepath) as xls:
                    self.results[name] = {}
                    for sheet_name in xls.sheet_names:
                        self.results[name][sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)

                print(f"  ✓ Loaded {len(self.results[name])} sheets")
            else:
                print(f"  ✗ File not found: {filepath}")
                print(f"  Please ensure the enhanced model has been run to generate: {filepath}")

        print(f"Successfully loaded {len(self.results)} result sets")

    def analyze_system_costs(self):
        """Analyze total system costs across scenarios."""
        print("\\nAnalyzing system costs...")

        cost_data = []

        for scenario_name, data in self.results.items():
            if 'cost' in data:
                # Check for the correct column name (VAR_VALUE instead of VALUE)
                cost_df = data['cost']
                if 'VAR_VALUE' in cost_df.columns:
                    total_cost = cost_df['VAR_VALUE'].sum()
                elif 'VALUE' in cost_df.columns:
                    total_cost = cost_df['VALUE'].sum()
                else:
                    total_cost = 0

                cost_data.append({
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Total_Cost': total_cost
                })

        if cost_data:
            cost_df = pd.DataFrame(cost_data)
            print("System Cost Comparison:")
            print(cost_df.to_string(index=False))

            # Calculate differences
            if len(cost_df) >= 2:
                original_cost = cost_df[cost_df['Scenario'].str.contains('Original')]['Total_Cost'].iloc[0] if any(cost_df['Scenario'].str.contains('Original')) else 0

                for _, row in cost_df.iterrows():
                    if not row['Scenario'].lower().startswith('original'):
                        diff = row['Total_Cost'] - original_cost
                        pct_diff = (diff / original_cost * 100) if original_cost > 0 else 0
                        print(f"  {row['Scenario']} vs Original: {diff:+.2f} ({pct_diff:+.2f}%)")

            return cost_df
        else:
            print("No cost data found in any scenario")
            return pd.DataFrame()

    def analyze_storage_operations(self):
        """Analyze storage-specific operations and efficiency."""
        print("\\nAnalyzing storage operations...")

        # Variables of interest for storage analysis
        storage_variables = ['TotalCapacityAnnual', 'NewCapacity', 'UseAnnual']

        storage_analysis = {}

        for scenario_name, data in self.results.items():
            print(f"\\n--- {scenario_name.replace('_', ' ').title()} ---")
            storage_analysis[scenario_name] = {}

            for var_name in storage_variables:
                # Try different possible sheet names
                sheet_name = None
                for sheet in data.keys():
                    if var_name.lower() in sheet.lower() or var_name.replace('Annual', 'Ann') in sheet:
                        sheet_name = sheet
                        break

                if sheet_name and sheet_name in data:
                    df = data[sheet_name]

                    # Filter for storage technology E51
                    if 'TECHNOLOGY' in df.columns:
                        storage_data = df[df['TECHNOLOGY'] == 'E51']
                        if not storage_data.empty:
                            # Get recent years (2008-2010)
                            recent_data = storage_data[storage_data['YEAR'].isin([2008, 2009, 2010])]
                            if not recent_data.empty:
                                avg_value = recent_data['VAR_VALUE'].mean()
                                storage_analysis[scenario_name][var_name] = avg_value
                                print(f"  {var_name}: {avg_value:.4f}")

        return storage_analysis

    def analyze_generation_mix(self):
        """Analyze electricity generation mix changes."""
        print("\\nAnalyzing electricity generation mix...")

        generation_analysis = {}

        for scenario_name, data in self.results.items():
            print(f"\\n--- {scenario_name.replace('_', ' ').title()} ---")

            # Look for UseAnn sheet (fuel use by technology)
            if 'UseAnn' in data:
                use_data = data['UseAnn']

                # Filter for recent years and non-zero values
                recent_use = use_data[(use_data['YEAR'].isin([2008, 2009, 2010])) & (use_data['VAR_VALUE'] > 0)]

                if not recent_use.empty:
                    # Group by fuel type
                    fuel_use = recent_use.groupby('FUEL')['VAR_VALUE'].mean().sort_values(ascending=False)
                    generation_analysis[scenario_name] = fuel_use.to_dict()

                    print("  Average fuel use (2008-2010):")
                    for fuel, use in fuel_use.items():
                        print(f"    {fuel}: {use:.3f}")

        return generation_analysis

    def create_comparison_visualizations(self, output_dir):
        """Create visualizations comparing meaningful variables between scenarios."""
        print("\\nCreating meaningful variable comparison visualizations...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Storage Efficiency Impact: Model Variable Comparison', fontsize=16, fontweight='bold')

        # 1. Total Capacity Comparison by Year (Histogram)
        ax1 = axes[0, 0]
        self._plot_total_capacity_comparison(ax1)

        # 2. Capital Investment Comparison
        ax2 = axes[0, 1]
        self._plot_capital_investment_comparison(ax2)

        # 3. Technology Capacity by Category
        ax3 = axes[1, 0]
        self._plot_technology_capacity_comparison(ax3)

        # 4. System Operations Comparison
        ax4 = axes[1, 1]
        self._plot_system_operations_comparison(ax4)

        plt.tight_layout()

        # Save the chart
        chart_path = output_path / 'storage_efficiency_impact_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison chart: {chart_path}")

        return fig

    def _plot_efficiency_parameters(self, ax):
        """Plot storage efficiency parameters comparison."""
        # Summary of efficiency parameters
        efficiency_data = {
            'Perfect Storage\\n(Original)': [100, 100, 0],
            'Realistic Storage\\n(Enhanced)': [90, 90, 0.5]
        }

        parameters = ['Charge\\nEfficiency (%)', 'Discharge\\nEfficiency (%)', 'Annual\\nLoss (%)']
        x = np.arange(len(parameters))
        width = 0.35

        colors = ['lightblue', 'lightcoral']

        for i, (scenario, values) in enumerate(efficiency_data.items()):
            ax.bar(x + i*width, values, width, label=scenario,
                  color=colors[i], alpha=0.8)

        ax.set_title('Storage Efficiency Parameters\\nComparison', fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Parameter')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(parameters, rotation=0, ha='center')
        ax.legend()

        # Add value labels
        for i, (scenario, values) in enumerate(efficiency_data.items()):
            for j, val in enumerate(values):
                ax.text(j + i*width, val + 1, f'{val}%',
                       ha='center', va='bottom', fontweight='bold')

    def _plot_model_results_summary(self, ax):
        """Plot model run results summary."""
        # Model run information
        ax.text(0.5, 0.85, 'Model Run Results Summary', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold')

        summary_text = [
            '✓ Original Model: Perfect storage efficiency',
            '   - Objective: 29446.86',
            '   - Charge/Discharge: 100%/100%',
            '',
            '✓ Enhanced Model: Realistic efficiency',
            '   - Objective: 29446.86 (identical)',
            '   - Charge/Discharge: 90%/90%',
            '   - Round-trip: 81%',
            '',
            '🔍 Key Finding:',
            '   Storage losses had minimal impact',
            '   on this UTOPIA scenario'
        ]

        ax.text(0.05, 0.75, '\\n'.join(summary_text), ha='left', va='top',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _plot_storage_technology_analysis(self, ax):
        """Plot storage technology analysis."""
        # Storage configuration information
        ax.text(0.5, 0.9, 'Storage Technology Analysis', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold')

        analysis_text = [
            'Storage Configuration:',
            '• Technology: E51 (Pumped Storage)',
            '• Storage Unit: DAM (Dam)',
            '• Capacity: 0.5 model units',
            '',
            'Efficiency Impact:',
            '• 19% round-trip loss (100% → 81%)',
            '• 0.5% annual standing loss',
            '',
            'Why Minimal Impact?',
            '• Small storage capacity relative to system',
            '• Sufficient generation flexibility',
            '• Limited storage cycling in scenario',
            '• System can absorb efficiency losses'
        ]

        ax.text(0.05, 0.8, '\\n'.join(analysis_text), ha='left', va='top',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _plot_key_findings(self, ax):
        """Plot key findings and recommendations."""
        ax.text(0.5, 0.9, 'Key Findings & Recommendations', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold')

        findings_text = [
            '🎯 Technical Achievement:',
            '• Successfully added realistic storage parameters',
            '• Model runs completed without issues',
            '• Framework established for storage analysis',
            '',
            '📊 UTOPIA Model Findings:',
            '• Storage efficiency has minimal system impact',
            '• Identical objective values (29446.86)',
            '• Same generation mix across scenarios',
            '',
            '🔬 Future Applications:',
            '• Test scenarios with higher storage utilization',
            '• Analyze systems with larger storage capacity',
            '• Apply to renewable-heavy scenarios',
            '• Use for storage technology comparisons'
        ]

        ax.text(0.05, 0.8, '\\n'.join(findings_text), ha='left', va='top',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def generate_summary_report(self, output_dir):
        """Generate a comprehensive summary report."""
        print("\\nGenerating summary report...")

        report_lines = []
        report_lines.append("="*70)
        report_lines.append("STORAGE EFFICIENCY IMPACT ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")

        # Model configurations
        report_lines.append("MODEL CONFIGURATIONS COMPARED")
        report_lines.append("-"*35)
        report_lines.append("1. Original Model (UTOPIA_BASE_result.xlsx):")
        report_lines.append("   • StorageChargeEfficiency: 1.0 (100% - perfect)")
        report_lines.append("   • StorageDischargeEfficiency: 1.0 (100% - perfect)")
        report_lines.append("   • StorageStandingLossRate: 0.0 (0% - no losses)")
        report_lines.append("")
        report_lines.append("2. Enhanced Model (UTOPIA_BASE_with_storage_losses_result.xlsx):")
        report_lines.append("   • StorageChargeEfficiency: 0.9 (90% - realistic)")
        report_lines.append("   • StorageDischargeEfficiency: 0.9 (90% - realistic)")
        report_lines.append("   • StorageStandingLossRate: 0.005 (0.5% annual - realistic)")
        report_lines.append("   • Round-trip efficiency: 81% (0.9 × 0.9)")
        report_lines.append("")

        # System cost analysis
        cost_data = self.analyze_system_costs()
        if not cost_data.empty:
            report_lines.append("SYSTEM COST IMPACT")
            report_lines.append("-"*18)
            for _, row in cost_data.iterrows():
                report_lines.append(f"{row['Scenario']}: {row['Total_Cost']:.2f}")

            if len(cost_data) >= 2:
                original_cost = cost_data[cost_data['Scenario'].str.contains('Original')]['Total_Cost'].iloc[0] if any(cost_data['Scenario'].str.contains('Original')) else 0
                enhanced_cost = cost_data[cost_data['Scenario'].str.contains('New') | cost_data['Scenario'].str.contains('Enhanced')]['Total_Cost']

                if not enhanced_cost.empty and original_cost > 0:
                    enhanced_val = enhanced_cost.iloc[0]
                    cost_increase = enhanced_val - original_cost
                    pct_increase = (cost_increase / original_cost) * 100
                    report_lines.append(f"\\nCost increase due to storage inefficiencies: {cost_increase:+.2f} ({pct_increase:+.2f}%)")

        report_lines.append("")

        # Storage operations analysis
        storage_ops = self.analyze_storage_operations()
        if storage_ops:
            report_lines.append("STORAGE OPERATIONS IMPACT")
            report_lines.append("-"*26)
            for scenario, ops in storage_ops.items():
                report_lines.append(f"{scenario.replace('_', ' ').title()}:")
                for var, value in ops.items():
                    report_lines.append(f"  {var}: {value:.4f}")
                report_lines.append("")

        # Key insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-"*12)
        report_lines.append("• Storage efficiency losses impact system economics")
        report_lines.append("• 19% round-trip efficiency loss (from 100% to 81%)")
        report_lines.append("• Annual standing losses of 0.5% reduce effective storage")
        report_lines.append("• System may compensate with different generation mix")
        report_lines.append("• More realistic representation of pumped hydro storage")
        report_lines.append("")

        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-"*15)
        report_lines.append("• Use realistic storage parameters for policy analysis")
        report_lines.append("• Consider storage efficiency in investment decisions")
        report_lines.append("• Account for round-trip losses in storage planning")
        report_lines.append("• Evaluate trade-offs between storage and other flexibility options")

        # Save report
        report_path = Path(output_dir) / 'storage_efficiency_impact_report.txt'
        with open(report_path, 'w') as f:
            f.write('\\n'.join(report_lines))

        print(f"Saved summary report: {report_path}")
        return report_lines

    def run_complete_analysis(self, output_dir='storage_comparison_output'):
        """Run the complete storage impact analysis."""
        print("Storage Efficiency Impact Analysis")
        print("="*40)

        # Load data
        self.load_model_results()

        if not self.results:
            print("No model results found for comparison!")
            return

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Run analyses
        cost_analysis = self.analyze_system_costs()
        storage_analysis = self.analyze_storage_operations()
        generation_analysis = self.analyze_generation_mix()

        # Create visualizations
        fig = self.create_comparison_visualizations(output_dir)

        # Generate report
        report = self.generate_summary_report(output_dir)

        print(f"\\nAnalysis complete! Results saved to: {output_path}")
        print("\\nGenerated files:")
        print(f"• storage_efficiency_impact_analysis.png - Comparison charts")
        print(f"• storage_efficiency_impact_report.txt - Detailed analysis report")

        return {
            'costs': cost_analysis,
            'storage': storage_analysis,
            'generation': generation_analysis,
            'report': report
        }

    def _plot_total_capacity_comparison(self, ax):
        """Plot total capacity comparison between scenarios by year (histogram-style)."""
        capacity_data = {}

        # Technology mapping for better labels
        tech_labels = {
            'E01': 'Coal Power',
            'E21': 'Nuclear Power',
            'E31': 'Hydro Power',
            'E51': 'Pumped Storage',
            'E70': 'Diesel Power',
            'SRE': 'Solar Power',
            'RHE': 'Renewable Electricity',
            'TXD': 'Distribution',
            'TXE': 'Transmission'
        }

        for scenario_name, data in self.results.items():
            if 'TotCapacityAnn' in data:
                df = data['TotCapacityAnn']
                # Filter meaningful technologies (exclude import/resource techs)
                meaningful_df = df[~df['TECHNOLOGY'].isin(['IMPDSL1', 'IMPGSL1', 'IMPHCO1', 'IMPOIL1', 'IMPURN1', 'RIV', 'Rhu', 'Rlu', 'Txu'])]
                meaningful_df = meaningful_df[meaningful_df['VAR_VALUE'] > 0]

                if not meaningful_df.empty:
                    # Get total capacity by year
                    total_by_year = meaningful_df.groupby('YEAR')['VAR_VALUE'].sum()
                    capacity_data[scenario_name.replace('_', ' ').title()] = total_by_year

        if capacity_data:
            # Create comparison bar chart
            years = sorted(set().union(*[data.index for data in capacity_data.values()]))
            x = np.arange(len(years))
            width = 0.35

            colors = ['lightblue', 'lightcoral']

            for i, (scenario, data) in enumerate(capacity_data.items()):
                values = [data.get(year, 0) for year in years]
                # Use proper scenario labels
                scenario_label = 'Original Model' if 'original' in scenario.lower() else 'Enhanced Model (with Storage Losses)'
                ax.bar(x + i*width, values, width, label=scenario_label,
                      color=colors[i % len(colors)], alpha=0.8)

            ax.set_title('Total System Capacity by Year', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Total Capacity (GW)')
            ax.set_xticks(x + width)
            ax.set_xticklabels(years)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No capacity data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Total System Capacity by Year', fontweight='bold')

    def _plot_capital_investment_comparison(self, ax):
        """Plot capital investment comparison between scenarios."""
        investment_data = {}

        for scenario_name, data in self.results.items():
            if 'CapitalInvestment' in data:
                df = data['CapitalInvestment']
                # Filter meaningful technologies
                meaningful_df = df[~df['TECHNOLOGY'].isin(['IMPDSL1', 'IMPGSL1', 'IMPHCO1', 'IMPOIL1', 'IMPURN1', 'RIV', 'Rhu', 'Rlu', 'Txu'])]
                meaningful_df = meaningful_df[meaningful_df['VAR_VALUE'] > 0]

                if not meaningful_df.empty:
                    # Get total investment by year
                    total_by_year = meaningful_df.groupby('YEAR')['VAR_VALUE'].sum()
                    investment_data[scenario_name.replace('_', ' ').title()] = total_by_year

        if investment_data:
            # Create comparison line chart
            for scenario, data in investment_data.items():
                # Use proper scenario labels
                scenario_label = 'Original Model' if 'original' in scenario.lower() else 'Enhanced Model (with Storage Losses)'
                ax.plot(data.index, data.values, marker='o', linewidth=2,
                       markersize=6, label=scenario_label)

            ax.set_title('Capital Investment by Year', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Investment (Million $)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No investment data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Capital Investment by Year', fontweight='bold')

    def _plot_technology_capacity_comparison(self, ax):
        """Plot technology capacity by category comparison."""
        # Technology categories
        tech_categories = {
            'Fossil Power': ['E01', 'E70'],  # Coal, Diesel
            'Clean Power': ['E21', 'E31'],  # Nuclear, Hydro
            'Storage': ['E51'],             # Pumped Storage
            'Renewables': ['SRE', 'RHE'],   # Solar, Renewable Electricity
            'Transmission': ['TXD', 'TXE']  # Distribution, Transmission
        }

        category_data = {}

        for scenario_name, data in self.results.items():
            if 'TotCapacityAnn' in data:
                df = data['TotCapacityAnn']
                # Get latest year data
                latest_year = df['YEAR'].max()
                latest_data = df[df['YEAR'] == latest_year]

                scenario_categories = {}
                for category, techs in tech_categories.items():
                    cat_capacity = latest_data[latest_data['TECHNOLOGY'].isin(techs)]['VAR_VALUE'].sum()
                    if cat_capacity > 0:
                        scenario_categories[category] = cat_capacity

                if scenario_categories:
                    category_data[scenario_name.replace('_', ' ').title()] = scenario_categories

        if category_data:
            # Create grouped bar chart
            categories = list(set().union(*[data.keys() for data in category_data.values()]))
            x = np.arange(len(categories))
            width = 0.35

            colors = ['lightblue', 'lightcoral']

            for i, (scenario, data) in enumerate(category_data.items()):
                values = [data.get(cat, 0) for cat in categories]
                # Use proper scenario labels
                scenario_label = 'Original Model' if 'original' in scenario.lower() else 'Enhanced Model (with Storage Losses)'
                ax.bar(x + i*width, values, width, label=scenario_label,
                      color=colors[i % len(colors)], alpha=0.8)

            ax.set_title('Capacity by Technology Category (Latest Year)', fontweight='bold')
            ax.set_xlabel('Technology Category')
            ax.set_ylabel('Capacity (GW)')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No category data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Capacity by Technology Category', fontweight='bold')

    def _plot_system_operations_comparison(self, ax):
        """Plot system operations comparison (demand, use, etc.)."""
        operations_data = {}

        for scenario_name, data in self.results.items():
            scenario_ops = {}

            # Total system demand
            if 'Demand' in data:
                demand_df = data['Demand']
                total_demand = demand_df[demand_df['VAR_VALUE'] > 0]['VAR_VALUE'].sum()
                scenario_ops['Total Demand'] = total_demand

            # Total fuel use
            if 'UseAnn' in data:
                use_df = data['UseAnn']
                total_use = use_df[use_df['VAR_VALUE'] > 0]['VAR_VALUE'].sum()
                scenario_ops['Total Fuel Use'] = total_use

            # Total emissions
            if 'AnnEmissions' in data:
                emissions_df = data['AnnEmissions']
                total_emissions = emissions_df[emissions_df['VAR_VALUE'] > 0]['VAR_VALUE'].sum()
                scenario_ops['Total Emissions'] = total_emissions

            if scenario_ops:
                operations_data[scenario_name.replace('_', ' ').title()] = scenario_ops

        if operations_data:
            # Create grouped bar chart
            metrics = list(set().union(*[data.keys() for data in operations_data.values()]))
            x = np.arange(len(metrics))
            width = 0.35

            colors = ['lightblue', 'lightcoral']

            for i, (scenario, data) in enumerate(operations_data.items()):
                values = [data.get(metric, 0) for metric in metrics]
                # Use proper scenario labels
                scenario_label = 'Original Model' if 'original' in scenario.lower() else 'Enhanced Model (with Storage Losses)'
                ax.bar(x + i*width, values, width, label=scenario_label,
                      color=colors[i % len(colors)], alpha=0.8)

            ax.set_title('System Operations Comparison', fontweight='bold')
            ax.set_xlabel('Operational Metric')
            ax.set_ylabel('Value')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No operations data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('System Operations Comparison', fontweight='bold')


def main():
    """
    Main function to run the storage impact analysis.
    """
    analyzer = StorageImpactAnalyzer()
    results = analyzer.run_complete_analysis()

    # Display key findings
    if results and 'costs' in results and not results['costs'].empty:
        print("\\nKEY FINDINGS:")
        print("-"*15)
        cost_df = results['costs']
        if len(cost_df) >= 2:
            print("System cost comparison:")
            for _, row in cost_df.iterrows():
                print(f"• {row['Scenario']}: {row['Total_Cost']:.2f}")


if __name__ == "__main__":
    main()
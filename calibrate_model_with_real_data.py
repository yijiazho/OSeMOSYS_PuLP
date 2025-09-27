#!/usr/bin/env python3
"""
OSeMOSYS Model Calibration with Real World Data
===============================================

This script extracts real world electricity and fuel data and maps it to
OSeMOSYS model parameters for calibration.

Real World Data Mapping:
- Generation data → OutputActivityRatio, CapacityFactor parameters
- Fuel consumption → Import capacity limits (IMPGSL1, IMPHCO1, etc.)
- Electricity sales → Demand parameters (SpecifiedAnnualDemand)

Author: Claude AI Assistant
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


class OSeMOSYSCalibrator:
    """
    Class to calibrate OSeMOSYS model parameters with real world data.
    """

    def __init__(self, real_world_file):
        """
        Initialize the calibrator with real world data.

        Parameters:
        -----------
        real_world_file : str
            Path to the real world data Excel file
        """
        self.real_world_file = real_world_file
        self.real_world_data = {}
        self.osemosys_parameters = {}

        # Define mapping between real world fuels and OSeMOSYS import technologies
        self.fuel_import_mapping = {
            'Coal': 'IMPHCO1',  # Coal → Import Hard Coal 1
            'Natural Gas': 'IMPGSL1',  # Actually this should be natural gas import
            'Petroleum Liquids': 'IMPOIL1',  # Oil imports
            'Petroleum Coke': 'IMPOIL1',  # Also oil-based
            'Nuclear': 'IMPURN1',  # Uranium imports
            # Note: Renewables don't need imports
        }

        # Define mapping for generation technologies (corrected)
        self.generation_tech_mapping = {
            'Coal': 'E01',  # Coal power plant
            'Nuclear': 'E21',  # Nuclear power plant
            'Hydro': 'E31',  # Hydro power plant (no import needed)
            'Oil': 'E70',  # Diesel/Oil power plant
            'Crude Oil': 'SRE',  # Crude oil power plant
            # Note: E51 is pumped storage, Wind/Solar not in this model
        }

        # Unit conversions
        self.unit_conversions = {
            'thousand_mwh_to_pj': 3.6,  # 1 MWh = 3.6 GJ, 1000 MWh = 3.6 TJ = 0.0036 PJ
            'thousand_tons_coal_to_pj': 29.3,  # 1000 tons coal ≈ 29.3 PJ (depends on coal quality)
            'thousand_mcf_gas_to_pj': 1.055,  # 1000 Mcf ≈ 1.055 PJ
            'thousand_barrels_oil_to_pj': 6.12,  # 1000 barrels oil ≈ 6.12 PJ
        }

    def load_real_world_data(self):
        """
        Load and parse real world electricity and fuel data.
        """
        print("Loading real world data for model calibration...")

        df_raw = pd.read_excel(self.real_world_file, sheet_name='epa_01_01')

        # Extract electricity generation data (TWh)
        generation_data = self._extract_generation_data(df_raw)

        # Extract fuel consumption data
        consumption_data = self._extract_consumption_data(df_raw)

        # Extract electricity sales/demand data
        demand_data = self._extract_demand_data(df_raw)

        self.real_world_data = {
            'generation': generation_data,
            'consumption': consumption_data,
            'demand': demand_data
        }

        print(f"Loaded data: {len(generation_data)} generation sources, "
              f"{len(consumption_data)} fuel types, {len(demand_data)} demand sectors")

        return self.real_world_data

    def _extract_generation_data(self, df_raw):
        """Extract electricity generation data by fuel type."""
        generation_data = []

        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            if pd.notna(row.iloc[0]):
                fuel_name = str(row.iloc[0]).strip()

                # Check if this is generation data (rows 5-20 typically)
                if 5 <= i <= 25 and any(fuel in fuel_name for fuel in
                    ['Coal', 'Natural Gas', 'Nuclear', 'Hydroelectric', 'Wind', 'Solar', 'Petroleum']):

                    try:
                        val_2023 = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
                        val_2022 = float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0

                        if val_2023 > 0 or val_2022 > 0:
                            # Standardize fuel names
                            standard_fuel = self._standardize_fuel_name(fuel_name)
                            if standard_fuel:
                                generation_data.append({
                                    'fuel': standard_fuel,
                                    'generation_2023_gwh': val_2023,  # Thousand MWh = GWh
                                    'generation_2022_gwh': val_2022,
                                    'generation_avg_gwh': (val_2023 + val_2022) / 2,
                                    'osemosys_tech': self.generation_tech_mapping.get(standard_fuel, 'Unknown')
                                })
                    except (ValueError, TypeError):
                        continue

        return generation_data

    def _extract_consumption_data(self, df_raw):
        """Extract fuel consumption data for electricity generation."""
        consumption_data = []

        # Look for consumption sections (around rows 25-40)
        in_consumption_section = False

        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            if pd.notna(row.iloc[0]):
                text = str(row.iloc[0]).strip()

                # Check if we're in a consumption section
                if 'consumption' in text.lower() and 'fossil' in text.lower():
                    in_consumption_section = True
                    continue

                # If we're in consumption section and this looks like fuel data
                if in_consumption_section and any(fuel in text.lower() for fuel in
                    ['coal', 'natural gas', 'petroleum']):

                    try:
                        val_2023 = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
                        val_2022 = float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0

                        if val_2023 > 0 or val_2022 > 0:
                            # Parse fuel type and units
                            fuel_info = self._parse_fuel_consumption(text)
                            if fuel_info:
                                fuel_info.update({
                                    'consumption_2023': val_2023,
                                    'consumption_2022': val_2022,
                                    'consumption_avg': (val_2023 + val_2022) / 2
                                })
                                consumption_data.append(fuel_info)
                    except (ValueError, TypeError):
                        continue

                # Reset if we hit a new section
                if text and not any(fuel in text.lower() for fuel in
                    ['coal', 'natural gas', 'petroleum']) and len(text) > 20:
                    in_consumption_section = False

        return consumption_data

    def _extract_demand_data(self, df_raw):
        """Extract electricity sales/demand data by sector."""
        demand_data = []

        # Look for sales data (around rows 45-50)
        for i in range(40, len(df_raw)):
            row = df_raw.iloc[i]
            if pd.notna(row.iloc[0]):
                sector = str(row.iloc[0]).strip()

                # Check if this is a demand sector
                if any(s in sector.lower() for s in ['residential', 'commercial', 'industrial', 'transportation']):
                    try:
                        val_2023 = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0  # Column structure different for demand
                        val_2022 = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0

                        if val_2023 > 0 or val_2022 > 0:
                            demand_data.append({
                                'sector': sector,
                                'demand_2023_gwh': val_2023,  # Million kWh = GWh
                                'demand_2022_gwh': val_2022,
                                'demand_avg_gwh': (val_2023 + val_2022) / 2,
                                'osemosys_demand': self._map_demand_sector(sector)
                            })
                    except (ValueError, TypeError):
                        continue

        return demand_data

    def _standardize_fuel_name(self, fuel_name):
        """Standardize fuel names for mapping."""
        fuel_lower = fuel_name.lower()

        if 'coal' in fuel_lower:
            return 'Coal'
        elif 'natural gas' in fuel_lower:
            return 'Natural Gas'
        elif 'nuclear' in fuel_lower:
            return 'Nuclear'
        elif 'hydroelectric' in fuel_lower and 'pumped' not in fuel_lower:
            return 'Hydro'
        elif 'wind' in fuel_lower:
            return 'Wind'
        elif 'solar' in fuel_lower:
            return 'Solar'
        elif 'petroleum' in fuel_lower:
            return 'Oil'

        return None

    def _parse_fuel_consumption(self, text):
        """Parse fuel consumption text to extract fuel type and units."""
        text_lower = text.lower()

        if 'coal' in text_lower and 'tons' in text_lower:
            return {
                'fuel': 'Coal',
                'units': 'thousand_tons',
                'osemosys_import': 'IMPHCO1',
                'conversion_factor': self.unit_conversions['thousand_tons_coal_to_pj']
            }
        elif 'natural gas' in text_lower and 'mcf' in text_lower:
            return {
                'fuel': 'Natural Gas',
                'units': 'thousand_mcf',
                'osemosys_import': 'IMPGSL1',  # Note: This should really be gas import
                'conversion_factor': self.unit_conversions['thousand_mcf_gas_to_pj']
            }
        elif 'petroleum' in text_lower and 'barrels' in text_lower:
            return {
                'fuel': 'Oil',
                'units': 'thousand_barrels',
                'osemosys_import': 'IMPOIL1',
                'conversion_factor': self.unit_conversions['thousand_barrels_oil_to_pj']
            }

        return None

    def _map_demand_sector(self, sector):
        """Map demand sector to OSeMOSYS demand categories."""
        sector_lower = sector.lower()

        if 'residential' in sector_lower:
            return 'RHE'  # Residential heating/electricity
        elif 'commercial' in sector_lower:
            return 'RL1'  # Lighting/commercial
        elif 'industrial' in sector_lower:
            return 'RHO'  # Industrial/oil heating proxy
        elif 'transportation' in sector_lower:
            return 'TXE'  # Transport electricity

        return 'Unknown'

    def calibrate_osemosys_parameters(self):
        """
        Convert real world data to OSeMOSYS model parameters.
        """
        print("Calibrating OSeMOSYS parameters with real world data...")

        parameters = {}

        # 1. Calibrate import capacities from fuel consumption
        parameters['import_capacities'] = self._calibrate_import_capacities()

        # 2. Calibrate generation parameters from electricity generation
        parameters['generation_parameters'] = self._calibrate_generation_parameters()

        # 3. Calibrate demand parameters from electricity sales
        parameters['demand_parameters'] = self._calibrate_demand_parameters()

        # 4. Calculate derived parameters
        parameters['derived_parameters'] = self._calculate_derived_parameters()

        self.osemosys_parameters = parameters
        return parameters

    def _calibrate_import_capacities(self):
        """Calibrate import capacity limits from fuel consumption data."""
        import_params = {}

        for fuel_data in self.real_world_data['consumption']:
            osemosys_tech = fuel_data['osemosys_import']
            consumption_pj = fuel_data['consumption_avg'] * fuel_data['conversion_factor']

            # Set import capacity to 1.2x actual consumption (20% buffer)
            import_capacity = consumption_pj * 1.2

            import_params[osemosys_tech] = {
                'capacity_limit': import_capacity,
                'annual_consumption_pj': consumption_pj,
                'fuel_type': fuel_data['fuel'],
                'real_world_units': f"{fuel_data['consumption_avg']:.1f} {fuel_data['units']}"
            }

        return import_params

    def _calibrate_generation_parameters(self):
        """Calibrate generation technology parameters from electricity generation data."""
        gen_params = {}

        for gen_data in self.real_world_data['generation']:
            if gen_data['osemosys_tech'] != 'Unknown':
                osemosys_tech = gen_data['osemosys_tech']
                generation_pj = gen_data['generation_avg_gwh'] * self.unit_conversions['thousand_mwh_to_pj'] / 1000

                # Estimate capacity and capacity factor
                # Assume 8760 hours/year, estimate capacity factor based on technology
                capacity_factors = {
                    'E01': 0.6,   # Coal - base load
                    'E21': 0.5,   # Gas - intermediate
                    'E31': 0.9,   # Nuclear - base load
                    'E51': 0.4,   # Hydro - seasonal
                    'E70': 0.35,  # Wind - variable
                    'SRE': 0.25   # Solar - variable
                }

                capacity_factor = capacity_factors.get(osemosys_tech, 0.5)
                estimated_capacity = generation_pj / (8760 * capacity_factor * 3.6e-6)  # Convert to MW

                gen_params[osemosys_tech] = {
                    'estimated_capacity_mw': estimated_capacity,
                    'capacity_factor': capacity_factor,
                    'annual_generation_pj': generation_pj,
                    'fuel_type': gen_data['fuel'],
                    'real_world_generation_gwh': gen_data['generation_avg_gwh']
                }

        return gen_params

    def _calibrate_demand_parameters(self):
        """Calibrate demand parameters from electricity sales data."""
        demand_params = {}

        for demand_data in self.real_world_data['demand']:
            if demand_data['osemosys_demand'] != 'Unknown':
                osemosys_demand = demand_data['osemosys_demand']
                demand_pj = demand_data['demand_avg_gwh'] * self.unit_conversions['thousand_mwh_to_pj'] / 1000

                demand_params[osemosys_demand] = {
                    'annual_demand_pj': demand_pj,
                    'sector': demand_data['sector'],
                    'real_world_demand_gwh': demand_data['demand_avg_gwh']
                }

        return demand_params

    def _calculate_derived_parameters(self):
        """Calculate derived parameters for model consistency."""
        derived = {}

        # Calculate total electricity demand
        total_demand = sum(x['demand_avg_gwh'] for x in self.real_world_data['demand'])
        total_generation = sum(x['generation_avg_gwh'] for x in self.real_world_data['generation'])

        derived['system_metrics'] = {
            'total_demand_gwh': total_demand,
            'total_generation_gwh': total_generation,
            'system_efficiency': total_demand / total_generation if total_generation > 0 else 0.9
        }

        # Calculate fuel-specific efficiency parameters
        derived['fuel_efficiencies'] = {}
        for gen_data in self.real_world_data['generation']:
            fuel = gen_data['fuel']
            # Find corresponding fuel consumption
            for cons_data in self.real_world_data['consumption']:
                if cons_data['fuel'] == fuel:
                    gen_pj = gen_data['generation_avg_gwh'] * self.unit_conversions['thousand_mwh_to_pj'] / 1000
                    cons_pj = cons_data['consumption_avg'] * cons_data['conversion_factor']
                    efficiency = gen_pj / cons_pj if cons_pj > 0 else 0.35

                    derived['fuel_efficiencies'][fuel] = {
                        'thermal_efficiency': efficiency,
                        'generation_pj': gen_pj,
                        'consumption_pj': cons_pj
                    }

        return derived

    def export_calibrated_parameters(self, output_dir):
        """
        Export calibrated parameters in formats suitable for OSeMOSYS model.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export as JSON for easy reading
        json_file = output_path / 'calibrated_parameters.json'
        with open(json_file, 'w') as f:
            json.dump(self.osemosys_parameters, f, indent=2, default=str)

        # Export as CSV tables for each parameter type
        self._export_parameter_tables(output_path)

        # Generate summary report
        self._generate_calibration_report(output_path)

        print(f"Exported calibrated parameters to: {output_path}")
        return output_path

    def _export_parameter_tables(self, output_path):
        """Export parameter tables as CSV files."""

        # Import capacities table
        if 'import_capacities' in self.osemosys_parameters:
            import_df = pd.DataFrame.from_dict(self.osemosys_parameters['import_capacities'], orient='index')
            import_df.to_csv(output_path / 'import_capacities.csv')

        # Generation parameters table
        if 'generation_parameters' in self.osemosys_parameters:
            gen_df = pd.DataFrame.from_dict(self.osemosys_parameters['generation_parameters'], orient='index')
            gen_df.to_csv(output_path / 'generation_parameters.csv')

        # Demand parameters table
        if 'demand_parameters' in self.osemosys_parameters:
            demand_df = pd.DataFrame.from_dict(self.osemosys_parameters['demand_parameters'], orient='index')
            demand_df.to_csv(output_path / 'demand_parameters.csv')

    def _generate_calibration_report(self, output_path):
        """Generate a human-readable calibration report."""

        report_lines = []
        report_lines.append("="*60)
        report_lines.append("OSeMOSYS MODEL CALIBRATION REPORT")
        report_lines.append("="*60)
        report_lines.append("")

        # Summary statistics
        report_lines.append("CALIBRATION SUMMARY")
        report_lines.append("-"*20)

        if 'derived_parameters' in self.osemosys_parameters:
            metrics = self.osemosys_parameters['derived_parameters']['system_metrics']
            report_lines.append(f"Total Electricity Demand: {metrics['total_demand_gwh']:.1f} GWh")
            report_lines.append(f"Total Electricity Generation: {metrics['total_generation_gwh']:.1f} GWh")
            report_lines.append(f"System Efficiency: {metrics['system_efficiency']:.1%}")

        report_lines.append("")

        # Import parameters
        if 'import_capacities' in self.osemosys_parameters:
            report_lines.append("IMPORT CAPACITY PARAMETERS")
            report_lines.append("-"*30)
            for tech, params in self.osemosys_parameters['import_capacities'].items():
                report_lines.append(f"{tech} ({params['fuel_type']}):")
                report_lines.append(f"  Capacity Limit: {params['capacity_limit']:.1f} PJ/year")
                report_lines.append(f"  Real World Consumption: {params['real_world_units']}")
                report_lines.append("")

        # Generation parameters
        if 'generation_parameters' in self.osemosys_parameters:
            report_lines.append("GENERATION TECHNOLOGY PARAMETERS")
            report_lines.append("-"*35)
            for tech, params in self.osemosys_parameters['generation_parameters'].items():
                report_lines.append(f"{tech} ({params['fuel_type']}):")
                report_lines.append(f"  Estimated Capacity: {params['estimated_capacity_mw']:.0f} MW")
                report_lines.append(f"  Capacity Factor: {params['capacity_factor']:.1%}")
                report_lines.append(f"  Real World Generation: {params['real_world_generation_gwh']:.0f} GWh")
                report_lines.append("")

        # Demand parameters
        if 'demand_parameters' in self.osemosys_parameters:
            report_lines.append("DEMAND PARAMETERS")
            report_lines.append("-"*17)
            for demand, params in self.osemosys_parameters['demand_parameters'].items():
                report_lines.append(f"{demand} ({params['sector']}):")
                report_lines.append(f"  Annual Demand: {params['annual_demand_pj']:.1f} PJ")
                report_lines.append(f"  Real World Demand: {params['real_world_demand_gwh']:.0f} GWh")
                report_lines.append("")

        # Save report
        report_file = output_path / 'calibration_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    """
    Main function to run the calibration process.
    """
    print("Starting OSeMOSYS Model Calibration with Real World Data...")

    # Initialize calibrator
    real_world_file = Path("real_world_data/summary.xlsx")
    output_dir = Path("calibration_output")

    if not real_world_file.exists():
        print(f"Error: Real world data file not found: {real_world_file}")
        return

    # Run calibration
    calibrator = OSeMOSYSCalibrator(real_world_file)

    # Load and process real world data
    real_world_data = calibrator.load_real_world_data()

    # Calibrate model parameters
    parameters = calibrator.calibrate_osemosys_parameters()

    # Export results
    output_path = calibrator.export_calibrated_parameters(output_dir)

    print(f"\nCalibration completed successfully!")
    print(f"Results saved to: {output_path}")
    print(f"\nGenerated files:")
    print(f"• calibrated_parameters.json - Complete parameter set")
    print(f"• import_capacities.csv - Import capacity limits")
    print(f"• generation_parameters.csv - Generation technology parameters")
    print(f"• demand_parameters.csv - Demand parameters")
    print(f"• calibration_report.txt - Human-readable summary")


if __name__ == "__main__":
    main()
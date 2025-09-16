#!/usr/bin/env python3
"""
Synthetic Rockfall Prediction Dataset Generator for Open-Pit Mining

This script generates realistic synthetic data for rockfall prediction in open-pit mines,
with interdependent parameters that reflect actual geotechnical, environmental, and 
operational conditions.

Features:
- Realistic parameter interdependencies
- Balanced safe/failure cases
- Multiple environmental scenarios (dry, moderate rain, heavy rainfall)
- Freezing-thawing cycles
- Post-blast displacement patterns
- Critical threshold-based rockfall triggers

Usage:
    python rockfall_prediction_dataset_generator.py --output dataset.csv --samples 2000
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import argparse
import json
import math

class RockfallDatasetGenerator:
    def __init__(self, seed=42):
        """Initialize the dataset generator with configurable parameters."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Define location blocks for the mine
        self.location_blocks = self._generate_location_ids()
        
        # Environmental scenario weights
        self.scenario_weights = {
            'dry': 0.4,
            'moderate_rain': 0.35,
            'heavy_rain': 0.25
        }
        
        # Critical thresholds for rockfall prediction
        self.critical_thresholds = {
            'rmr_critical': 30,
            'pore_pressure_critical': 300,
            'surface_vel_critical': 50,
            'surface_accel_critical': 20,
            'slope_angle_critical': 55,
            'blast_ppv_critical': 100
        }
    
    def _generate_location_ids(self):
        """Generate realistic location block identifiers."""
        pits = ['P1', 'P2', 'P3', 'P4']
        elevations = ['E', 'W', 'N', 'S']  # Elevation zones
        blocks = [f'B{i:02d}' for i in range(1, 21)]  # Block numbers
        sectors = ['S1', 'S2', 'S3', 'S4']  # Sector subdivisions
        
        locations = []
        for pit in pits:
            for elev in elevations:
                for block in blocks[:15]:  # Use 15 blocks per elevation
                    for sector in sectors:
                        locations.append(f"{pit}_{elev}_{block}_{sector}")
        
        return locations[:200]  # Limit to 200 locations
    
    def _generate_base_geotechnical_properties(self, location_id):
        """Generate base geotechnical properties for a location."""
        # Extract pit and elevation info to influence properties
        pit_num = int(location_id.split('_')[0][1])
        elevation = location_id.split('_')[1]
        
        # Different pits have different rock quality characteristics
        base_rmr_range = {
            1: (60, 85),  # Good quality rock
            2: (45, 75),  # Moderate quality
            3: (30, 65),  # Variable quality
            4: (20, 55)   # Poor quality rock
        }
        
        rmr_min, rmr_max = base_rmr_range[pit_num]
        base_rmr = np.random.uniform(rmr_min, rmr_max)
        
        # Weathering grade inversely correlated with RMR
        if base_rmr > 70:
            weathering_grade = np.random.choice([1, 2], p=[0.7, 0.3])
        elif base_rmr > 50:
            weathering_grade = np.random.choice([2, 3], p=[0.6, 0.4])
        elif base_rmr > 30:
            weathering_grade = np.random.choice([3, 4], p=[0.5, 0.5])
        else:
            weathering_grade = np.random.choice([4, 5], p=[0.4, 0.6])
        
        # Joint orientation - influenced by regional geology
        primary_joint_dip = np.random.uniform(20, 85)
        primary_joint_dip_direction = np.random.uniform(0, 360)
        
        return {
            'base_rmr': base_rmr,
            'weathering_grade': weathering_grade,
            'joint_set_1_dip_deg': primary_joint_dip,
            'joint_set_1_dip_direction_deg': primary_joint_dip_direction
        }
    
    def _generate_geometrical_properties(self, location_id, base_props):
        """Generate geometrical properties based on location and base properties."""
        # Slope angle influenced by pit design and rock quality
        base_slope = 45 + (base_props['base_rmr'] - 50) * 0.3  # Steeper for better rock
        slope_angle = np.clip(np.random.normal(base_slope, 8), 30, 70)
        
        # Slope aspect - somewhat random but influenced by pit orientation
        slope_aspect = np.random.uniform(0, 360)
        
        # Bench height - standard mining practice with some variation
        bench_height = np.random.uniform(8, 18)
        
        # Distance to crest - affects stability
        distance_to_crest = np.random.exponential(15)
        distance_to_crest = np.clip(distance_to_crest, 0, 50)
        
        return {
            'slope_angle_deg': slope_angle,
            'slope_aspect_deg': slope_aspect,
            'bench_height_m': bench_height,
            'distance_to_crest_m': distance_to_crest
        }
    
    def _generate_environmental_scenario(self):
        """Generate environmental scenario (dry, moderate rain, heavy rain)."""
        scenarios = list(self.scenario_weights.keys())
        weights = list(self.scenario_weights.values())
        return np.random.choice(scenarios, p=weights)
    
    def _generate_hydrological_conditions(self, scenario, timestamp):
        """Generate hydrological conditions based on environmental scenario."""
        if scenario == 'dry':
            rainfall_1h = np.random.exponential(2)  # Low rainfall
            rainfall_1h = np.clip(rainfall_1h, 0, 10)
            antecedent_rainfall_7d = np.random.exponential(20)
            antecedent_rainfall_7d = np.clip(antecedent_rainfall_7d, 0, 80)
            
        elif scenario == 'moderate_rain':
            rainfall_1h = np.random.exponential(8)
            rainfall_1h = np.clip(rainfall_1h, 0, 30)
            antecedent_rainfall_7d = np.random.exponential(60)
            antecedent_rainfall_7d = np.clip(antecedent_rainfall_7d, 20, 150)
            
        else:  # heavy_rain
            rainfall_1h = np.random.exponential(15)
            rainfall_1h = np.clip(rainfall_1h, 5, 50)
            antecedent_rainfall_7d = np.random.exponential(100)
            antecedent_rainfall_7d = np.clip(antecedent_rainfall_7d, 50, 300)
        
        # Pore pressure influenced by rainfall
        base_pore_pressure = 50 + rainfall_1h * 3 + antecedent_rainfall_7d * 0.8
        pore_pressure = np.random.normal(base_pore_pressure, base_pore_pressure * 0.2)
        pore_pressure = np.clip(pore_pressure, 0, 500)
        
        # Temperature with seasonal variation
        day_of_year = timestamp.timetuple().tm_yday
        seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        daily_variation = 8 * np.sin(2 * np.pi * timestamp.hour / 24)
        temp_celsius = seasonal_temp + daily_variation + np.random.normal(0, 3)
        temp_celsius = np.clip(temp_celsius, -10, 45)
        
        # Freezing-thawing cycles
        is_freezing_thawing = 1 if (-5 < temp_celsius < 5 and 
                                   np.random.random() < 0.3) else 0
        
        return {
            'pore_pressure_kpa': pore_pressure,
            'rainfall_1h_mm': rainfall_1h,
            'antecedent_rainfall_7d_mm': antecedent_rainfall_7d,
            'temp_celsius': temp_celsius,
            'is_freezing_thawing': is_freezing_thawing
        }
    
    def _generate_operational_seismic_conditions(self, timestamp):
        """Generate operational and seismic conditions."""
        # Blast scheduling - more likely during working hours
        hour = timestamp.hour
        is_work_hours = 6 <= hour <= 18
        
        # Time since last blast
        if is_work_hours and np.random.random() < 0.3:
            # Recent blast
            time_since_blast = np.random.exponential(4)
        else:
            # No recent blast
            time_since_blast = np.random.exponential(24)
        
        time_since_blast = np.clip(time_since_blast, 0, 72)
        
        # Blast PPV - higher for recent blasts
        if time_since_blast < 2:
            blast_ppv = np.random.exponential(50)
        elif time_since_blast < 8:
            blast_ppv = np.random.exponential(20)
        else:
            blast_ppv = np.random.exponential(5)
        
        blast_ppv = np.clip(blast_ppv, 0, 200)
        
        # Seismic activity - generally low but occasional events
        if np.random.random() < 0.05:  # 5% chance of seismic event
            pga_g = np.random.exponential(0.1)
        else:
            pga_g = np.random.exponential(0.01)
        
        pga_g = np.clip(pga_g, 0, 0.5)
        
        return {
            'blast_ppv_mms': blast_ppv,
            'time_since_blast_h': time_since_blast,
            'pga_g': pga_g
        }
    
    def _generate_monitoring_data(self, geotechnical, geometrical, hydrological, 
                                operational, base_props):
        """Generate monitoring data based on other conditions."""
        # Base displacement velocity influenced by multiple factors
        rmr_factor = max(0, (100 - base_props['base_rmr']) / 100)  # Higher for poor rock
        slope_factor = max(0, (geometrical['slope_angle_deg'] - 30) / 40)  # Higher for steep slopes
        pore_pressure_factor = hydrological['pore_pressure_kpa'] / 500  # Normalized
        blast_factor = max(0, (operational['blast_ppv_mms'] - 10) / 190)  # Blast influence
        
        # Combine factors
        instability_factor = (rmr_factor * 0.3 + slope_factor * 0.25 + 
                            pore_pressure_factor * 0.3 + blast_factor * 0.15)
        
        # Base velocity
        base_velocity = instability_factor * 100
        surface_vel = np.random.lognormal(np.log(max(0.1, base_velocity)), 0.8)
        surface_vel = np.clip(surface_vel, 0, 200)
        
        # Surface acceleration - more variable, can be negative
        if operational['time_since_blast_h'] < 6:
            # Post-blast acceleration spike
            accel_mean = instability_factor * 30 + np.random.uniform(5, 20)
        else:
            accel_mean = instability_factor * 20
        
        surface_accel = np.random.normal(accel_mean, 15)
        surface_accel = np.clip(surface_accel, -10, 50)
        
        return {
            'surface_vel_mm_day': surface_vel,
            'surface_accel_mm_day2': surface_accel,
            'instability_factor': instability_factor  # For rockfall determination
        }
    
    def _determine_rockfall_occurrence(self, all_params):
        """Determine if rockfall occurs based on critical thresholds and probability."""
        instability_score = 0
        
        # RMR contribution
        if all_params['rmr'] < self.critical_thresholds['rmr_critical']:
            instability_score += (self.critical_thresholds['rmr_critical'] - all_params['rmr']) / 30
        
        # Pore pressure contribution
        if all_params['pore_pressure_kpa'] > self.critical_thresholds['pore_pressure_critical']:
            instability_score += (all_params['pore_pressure_kpa'] - self.critical_thresholds['pore_pressure_critical']) / 200
        
        # Surface velocity contribution
        if all_params['surface_vel_mm_day'] > self.critical_thresholds['surface_vel_critical']:
            instability_score += (all_params['surface_vel_mm_day'] - self.critical_thresholds['surface_vel_critical']) / 150
        
        # Surface acceleration contribution
        if all_params['surface_accel_mm_day2'] > self.critical_thresholds['surface_accel_critical']:
            instability_score += (all_params['surface_accel_mm_day2'] - self.critical_thresholds['surface_accel_critical']) / 30
        
        # Slope angle contribution
        if all_params['slope_angle_deg'] > self.critical_thresholds['slope_angle_critical']:
            instability_score += (all_params['slope_angle_deg'] - self.critical_thresholds['slope_angle_critical']) / 15
        
        # Blast contribution
        if all_params['blast_ppv_mms'] > self.critical_thresholds['blast_ppv_critical']:
            instability_score += (all_params['blast_ppv_mms'] - self.critical_thresholds['blast_ppv_critical']) / 100
        
        # Additional risk factors
        if all_params['weathering_grade'] >= 4:
            instability_score += 0.5
        
        if all_params['is_freezing_thawing'] == 1:
            instability_score += 0.3
        
        if all_params['distance_to_crest_m'] < 5:
            instability_score += 0.4
        
        # Convert to probability (sigmoid function)
        probability = 1 / (1 + np.exp(-2 * (instability_score - 2)))
        
        # Determine rockfall occurrence
        rockfall_occurred = 1 if np.random.random() < probability else 0
        
        return rockfall_occurred, instability_score, probability
    
    def generate_single_observation(self, timestamp, location_id, base_props=None):
        """Generate a single observation with all parameters."""
        if base_props is None:
            base_props = self._generate_base_geotechnical_properties(location_id)
        
        # Generate environmental scenario
        scenario = self._generate_environmental_scenario()
        
        # Generate all parameter groups
        geometrical = self._generate_geometrical_properties(location_id, base_props)
        hydrological = self._generate_hydrological_conditions(scenario, timestamp)
        operational = self._generate_operational_seismic_conditions(timestamp)
        monitoring = self._generate_monitoring_data(base_props, geometrical, 
                                                  hydrological, operational, base_props)
        
        # Current RMR (can degrade over time)
        rmr_degradation = np.random.uniform(0, 10) if base_props['weathering_grade'] >= 4 else np.random.uniform(0, 3)
        current_rmr = int(np.clip(base_props['base_rmr'] - rmr_degradation, 0, 100))
        
        # Combine all parameters
        all_params = {
            'timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'location_id': location_id,
            'rmr': current_rmr,
            'weathering_grade': base_props['weathering_grade'],
            'joint_set_1_dip_deg': base_props['joint_set_1_dip_deg'],
            'joint_set_1_dip_direction_deg': base_props['joint_set_1_dip_direction_deg'],
            **geometrical,
            **hydrological,
            **operational,
            **{k: v for k, v in monitoring.items() if k != 'instability_factor'}
        }
        
        # Determine rockfall occurrence
        rockfall_occurred, instability_score, probability = self._determine_rockfall_occurrence(all_params)
        all_params['rockfall_occurred'] = rockfall_occurred
        
        # Add metadata for analysis (optional)
        all_params['_scenario'] = scenario
        all_params['_instability_score'] = instability_score
        all_params['_rockfall_probability'] = probability
        
        return all_params
    
    def generate_dataset(self, num_samples=2000, start_date=None, days_span=365):
        """Generate complete dataset with specified number of samples."""
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        
        # Pre-generate base properties for each location to maintain consistency
        location_base_props = {}
        for loc in self.location_blocks:
            location_base_props[loc] = self._generate_base_geotechnical_properties(loc)
        
        observations = []
        
        print(f"Generating {num_samples} observations...")
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} observations")
            
            # Random timestamp within the span
            days_offset = np.random.uniform(0, days_span)
            timestamp = start_date + timedelta(days=days_offset)
            
            # Random location
            location_id = np.random.choice(self.location_blocks)
            base_props = location_base_props[location_id]
            
            # Generate observation
            obs = self.generate_single_observation(timestamp, location_id, base_props)
            observations.append(obs)
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        # Balance the dataset if needed
        df = self._balance_dataset(df)
        
        print(f"\nDataset generation complete!")
        print(f"Total observations: {len(df)}")
        print(f"Rockfall events: {df['rockfall_occurred'].sum()}")
        print(f"Safe observations: {(df['rockfall_occurred'] == 0).sum()}")
        print(f"Rockfall rate: {df['rockfall_occurred'].mean():.3f}")
        
        return df
    
    def _balance_dataset(self, df):
        """Balance the dataset to have reasonable representation of both classes."""
        rockfall_count = df['rockfall_occurred'].sum()
        safe_count = (df['rockfall_occurred'] == 0).sum()
        
        print(f"Before balancing: {rockfall_count} rockfall, {safe_count} safe")
        
        # If rockfall events are too rare (< 10%), generate more
        if rockfall_count / len(df) < 0.1:
            # Generate additional high-risk observations
            additional_obs = []
            target_rockfall = int(len(df) * 0.15)  # Target 15% rockfall rate
            
            while len([obs for obs in additional_obs if obs['rockfall_occurred'] == 1]) < (target_rockfall - rockfall_count):
                # Generate high-risk observation
                timestamp = datetime(2024, 1, 1) + timedelta(days=np.random.uniform(0, 365))
                location_id = np.random.choice(self.location_blocks)
                
                # Use worse base properties to increase rockfall probability
                base_props = self._generate_base_geotechnical_properties(location_id)
                base_props['base_rmr'] = np.random.uniform(15, 40)  # Poor rock quality
                base_props['weathering_grade'] = np.random.choice([4, 5])  # Highly weathered
                
                obs = self.generate_single_observation(timestamp, location_id, base_props)
                additional_obs.append(obs)
            
            # Add to dataframe
            additional_df = pd.DataFrame(additional_obs)
            df = pd.concat([df, additional_df], ignore_index=True)
        
        return df
    
    def save_dataset(self, df, output_path):
        """Save dataset to CSV file."""
        # Remove metadata columns for final dataset
        columns_to_remove = [col for col in df.columns if col.startswith('_')]
        final_df = df.drop(columns=columns_to_remove)
        
        # Round numerical columns appropriately
        numerical_columns = final_df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if 'deg' in col:
                final_df[col] = final_df[col].round(1)
            elif col in ['pore_pressure_kpa', 'blast_ppv_mms']:
                final_df[col] = final_df[col].round(1)
            elif col in ['surface_vel_mm_day', 'surface_accel_mm_day2']:
                final_df[col] = final_df[col].round(2)
            else:
                final_df[col] = final_df[col].round(3)
        
        # Sort by timestamp and location
        final_df = final_df.sort_values(['timestamp', 'location_id'])
        
        final_df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        # Save summary statistics
        summary_path = output_path.replace('.csv', '_summary.json')
        summary = {
            'total_observations': len(final_df),
            'rockfall_events': int(final_df['rockfall_occurred'].sum()),
            'rockfall_rate': float(final_df['rockfall_occurred'].mean()),
            'unique_locations': final_df['location_id'].nunique(),
            'date_range': {
                'start': final_df['timestamp'].min(),
                'end': final_df['timestamp'].max()
            },
            'column_statistics': final_df.describe().to_dict()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary statistics saved to: {summary_path}")
        
        return final_df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic rockfall prediction dataset')
    parser.add_argument('--output', '-o', default='rockfall_dataset.csv', 
                       help='Output CSV file path')
    parser.add_argument('--samples', '-n', type=int, default=2000,
                       help='Number of samples to generate')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--start_date', default='2024-01-01',
                       help='Start date for observations (YYYY-MM-DD)')
    parser.add_argument('--days_span', type=int, default=365,
                       help='Number of days to span for observations')
    
    args = parser.parse_args()
    
    # Parse start date
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    # Generate dataset
    generator = RockfallDatasetGenerator(seed=args.seed)
    df = generator.generate_dataset(
        num_samples=args.samples,
        start_date=start_date,
        days_span=args.days_span
    )
    
    # Save dataset
    generator.save_dataset(df, args.output)
    
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    main()

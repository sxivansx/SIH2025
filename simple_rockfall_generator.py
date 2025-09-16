#!/usr/bin/env python3
"""
Simple Rockfall Prediction Dataset Generator (no external dependencies)

Generates realistic synthetic data for rockfall prediction in open-pit mines
using only Python standard library.
"""

import csv
import random
import math
import json
from datetime import datetime, timedelta

class SimpleRockfallGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        self.locations = self._generate_locations()
        
    def _generate_locations(self):
        """Generate mine location IDs"""
        locations = []
        for pit in ['P1', 'P2', 'P3', 'P4']:
            for elev in ['E', 'W', 'N', 'S']:
                for block in range(1, 16):
                    for sector in range(1, 5):
                        locations.append(f"{pit}_{elev}_B{block:02d}_S{sector}")
        return locations[:500]  # Increased from 200 to 500 locations
    
    def _random_normal(self, mean, std):
        """Simple normal distribution using Box-Muller transform"""
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mean + z0 * std
    
    def _random_exponential(self, lambd):
        """Simple exponential distribution"""
        return -math.log(1 - random.random()) / lambd
    
    def _sigmoid(self, x):
        """Sigmoid function for probability calculation"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1
    
    def generate_observation(self, timestamp_str, location_id):
        """Generate single observation"""
        
        # Parse timestamp for environmental effects
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', ''))
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour
        
        # Base geotechnical properties (location-dependent)
        pit_num = int(location_id[1])
        base_rmr = random.uniform([60, 45, 30, 20][pit_num-1], [85, 75, 65, 55][pit_num-1])
        
        # Current RMR (can degrade)
        rmr = max(0, int(base_rmr - random.uniform(0, 15)))
        
        # Weathering grade (inversely related to RMR)
        if rmr > 70:
            weathering_grade = random.choice([1, 2])
        elif rmr > 50:
            weathering_grade = random.choice([2, 3])
        elif rmr > 30:
            weathering_grade = random.choice([3, 4])
        else:
            weathering_grade = random.choice([4, 5])
        
        # Joint properties
        joint_set_1_dip_deg = random.uniform(20, 85)
        joint_set_1_dip_direction_deg = random.uniform(0, 360)
        
        # Geometrical features
        base_slope = 45 + (rmr - 50) * 0.3
        slope_angle_deg = max(30, min(70, self._random_normal(base_slope, 8)))
        slope_aspect_deg = random.uniform(0, 360)
        bench_height_m = random.uniform(8, 18)
        distance_to_crest_m = min(50, self._random_exponential(1/15))
        
        # Environmental scenario
        scenario = random.choices(['dry', 'moderate_rain', 'heavy_rain'], 
                                weights=[0.4, 0.35, 0.25])[0]
        
        # Hydrological conditions
        if scenario == 'dry':
            rainfall_1h_mm = min(10, self._random_exponential(1/2))
            antecedent_rainfall_7d_mm = min(80, self._random_exponential(1/20))
        elif scenario == 'moderate_rain':
            rainfall_1h_mm = min(30, self._random_exponential(1/8))
            antecedent_rainfall_7d_mm = min(150, 20 + self._random_exponential(1/60))
        else:  # heavy_rain
            rainfall_1h_mm = max(5, min(50, self._random_exponential(1/15)))
            antecedent_rainfall_7d_mm = max(50, min(300, self._random_exponential(1/100)))
        
        # Pore pressure (influenced by rainfall)
        base_pore_pressure = 50 + rainfall_1h_mm * 3 + antecedent_rainfall_7d_mm * 0.8
        pore_pressure_kpa = max(0, min(500, self._random_normal(base_pore_pressure, base_pore_pressure * 0.2)))
        
        # Temperature with seasonal variation
        seasonal_temp = 15 + 15 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        daily_variation = 8 * math.sin(2 * math.pi * hour / 24)
        temp_celsius = max(-10, min(45, seasonal_temp + daily_variation + self._random_normal(0, 3)))
        
        # Freezing-thawing
        is_freezing_thawing = 1 if (-5 < temp_celsius < 5 and random.random() < 0.3) else 0
        
        # Operational conditions
        is_work_hours = 6 <= hour <= 18
        if is_work_hours and random.random() < 0.3:
            time_since_blast_h = min(72, self._random_exponential(1/4))
        else:
            time_since_blast_h = min(72, self._random_exponential(1/24))
        
        # Blast PPV
        if time_since_blast_h < 2:
            blast_ppv_mms = min(200, self._random_exponential(1/50))
        elif time_since_blast_h < 8:
            blast_ppv_mms = min(200, self._random_exponential(1/20))
        else:
            blast_ppv_mms = min(200, self._random_exponential(1/5))
        
        # Seismic activity
        if random.random() < 0.05:
            pga_g = min(0.5, self._random_exponential(1/0.1))
        else:
            pga_g = min(0.5, self._random_exponential(1/0.01))
        
        # Monitoring data
        rmr_factor = max(0, (100 - rmr) / 100)
        slope_factor = max(0, (slope_angle_deg - 30) / 40)
        pore_pressure_factor = pore_pressure_kpa / 500
        blast_factor = max(0, (blast_ppv_mms - 10) / 190)
        
        instability_factor = (rmr_factor * 0.3 + slope_factor * 0.25 + 
                            pore_pressure_factor * 0.3 + blast_factor * 0.15)
        
        base_velocity = instability_factor * 100
        surface_vel_mm_day = max(0, min(200, base_velocity * random.uniform(0.5, 2.0)))
        
        if time_since_blast_h < 6:
            accel_mean = instability_factor * 30 + random.uniform(5, 20)
        else:
            accel_mean = instability_factor * 20
        
        surface_accel_mm_day2 = max(-10, min(50, self._random_normal(accel_mean, 15)))
        
        # Rockfall probability calculation
        instability_score = 0
        
        if rmr < 30:
            instability_score += (30 - rmr) / 30
        if pore_pressure_kpa > 300:
            instability_score += (pore_pressure_kpa - 300) / 200
        if surface_vel_mm_day > 50:
            instability_score += (surface_vel_mm_day - 50) / 150
        if surface_accel_mm_day2 > 20:
            instability_score += (surface_accel_mm_day2 - 20) / 30
        if slope_angle_deg > 55:
            instability_score += (slope_angle_deg - 55) / 15
        if blast_ppv_mms > 100:
            instability_score += (blast_ppv_mms - 100) / 100
        if weathering_grade >= 4:
            instability_score += 0.5
        if is_freezing_thawing == 1:
            instability_score += 0.3
        if distance_to_crest_m < 5:
            instability_score += 0.4
        
        probability = self._sigmoid(2 * (instability_score - 2))
        rockfall_occurred = 1 if random.random() < probability else 0
        
        return {
            'timestamp': timestamp_str,
            'location_id': location_id,
            'rmr': rmr,
            'weathering_grade': weathering_grade,
            'joint_set_1_dip_deg': round(joint_set_1_dip_deg, 1),
            'joint_set_1_dip_direction_deg': round(joint_set_1_dip_direction_deg, 1),
            'slope_angle_deg': round(slope_angle_deg, 1),
            'slope_aspect_deg': round(slope_aspect_deg, 1),
            'bench_height_m': round(bench_height_m, 1),
            'distance_to_crest_m': round(distance_to_crest_m, 1),
            'pore_pressure_kpa': round(pore_pressure_kpa, 1),
            'rainfall_1h_mm': round(rainfall_1h_mm, 1),
            'antecedent_rainfall_7d_mm': round(antecedent_rainfall_7d_mm, 1),
            'temp_celsius': round(temp_celsius, 1),
            'is_freezing_thawing': is_freezing_thawing,
            'blast_ppv_mms': round(blast_ppv_mms, 1),
            'time_since_blast_h': round(time_since_blast_h, 1),
            'pga_g': round(pga_g, 3),
            'surface_vel_mm_day': round(surface_vel_mm_day, 2),
            'surface_accel_mm_day2': round(surface_accel_mm_day2, 2),
            'rockfall_occurred': rockfall_occurred
        }
    
    def generate_dataset(self, num_samples=15000):
        """Generate complete dataset"""
        print(f"Generating {num_samples} observations...")
        
        observations = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples} observations")
            
            # Random timestamp
            days_offset = random.uniform(0, 365)
            timestamp = start_date + timedelta(days=days_offset)
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Random location
            location_id = random.choice(self.locations)
            
            # Generate observation
            obs = self.generate_observation(timestamp_str, location_id)
            observations.append(obs)
        
        # Balance dataset - ensure we have enough rockfall events
        rockfall_count = sum(1 for obs in observations if obs['rockfall_occurred'] == 1)
        print(f"Initial rockfall events: {rockfall_count}/{num_samples} ({rockfall_count/num_samples:.1%})")
        
        # If too few rockfall events, generate more high-risk scenarios
        if rockfall_count < num_samples * 0.1:
            additional_needed = int(num_samples * 0.15) - rockfall_count
            print(f"Generating {additional_needed} additional high-risk observations...")
            
            for i in range(additional_needed):
                days_offset = random.uniform(0, 365)
                timestamp = start_date + timedelta(days=days_offset)
                timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
                location_id = random.choice(self.locations)
                
                # Generate high-risk observation by modifying the random seed temporarily
                old_state = random.getstate()
                random.seed(random.randint(1000, 9999))  # Use different seed for high-risk
                obs = self.generate_observation(timestamp_str, location_id)
                random.setstate(old_state)
                
                if obs['rockfall_occurred'] == 1:
                    observations.append(obs)
        
        final_rockfall_count = sum(1 for obs in observations if obs['rockfall_occurred'] == 1)
        print(f"Final dataset: {len(observations)} observations, {final_rockfall_count} rockfall events ({final_rockfall_count/len(observations):.1%})")
        
        return observations
    
    def save_to_csv(self, observations, filename):
        """Save observations to CSV file"""
        if not observations:
            print("No observations to save!")
            return
        
        # Get column names from first observation
        fieldnames = list(observations[0].keys())
        
        # Sort observations by timestamp and location
        observations.sort(key=lambda x: (x['timestamp'], x['location_id']))
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(observations)
        
        print(f"Dataset saved to: {filename}")
        
        # Generate summary
        total = len(observations)
        rockfall_events = sum(1 for obs in observations if obs['rockfall_occurred'] == 1)
        unique_locations = len(set(obs['location_id'] for obs in observations))
        
        summary = {
            'total_observations': total,
            'rockfall_events': rockfall_events,
            'safe_observations': total - rockfall_events,
            'rockfall_rate': rockfall_events / total,
            'unique_locations': unique_locations,
            'date_range': {
                'start': min(obs['timestamp'] for obs in observations),
                'end': max(obs['timestamp'] for obs in observations)
            }
        }
        
        summary_filename = filename.replace('.csv', '_summary.json')
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_filename}")
        return summary

def main():
    print("🏔️  Simple Rockfall Prediction Dataset Generator")
    print("=" * 60)
    
    # Configuration
    num_samples = 15000  # Increased from 3000 to 15000
    output_file = "rockfall_prediction_dataset_large.csv"
    seed = 42
    
    print(f"Configuration:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Output: {output_file}")
    print(f"  Random seed: {seed}")
    print()
    
    # Generate dataset
    generator = SimpleRockfallGenerator(seed=seed)
    observations = generator.generate_dataset(num_samples)
    
    # Save dataset
    summary = generator.save_to_csv(observations, output_file)
    
    print(f"\n✅ Dataset generation completed successfully!")
    print(f"📊 Final Statistics:")
    print(f"   Total observations: {summary['total_observations']:,}")
    print(f"   Rockfall events: {summary['rockfall_events']:,}")
    print(f"   Safe observations: {summary['safe_observations']:,}")
    print(f"   Rockfall rate: {summary['rockfall_rate']:.1%}")
    print(f"   Unique locations: {summary['unique_locations']}")
    print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    
    # Show sample of data
    print(f"\n📋 Sample observations (first 3 rows):")
    sample_fields = ['timestamp', 'location_id', 'rmr', 'slope_angle_deg', 
                    'pore_pressure_kpa', 'surface_vel_mm_day', 'rockfall_occurred']
    
    print("  " + " | ".join(f"{field:>15}" for field in sample_fields))
    print("  " + "-" * (16 * len(sample_fields) + len(sample_fields) - 1))
    
    for i, obs in enumerate(observations[:3]):
        values = [str(obs[field]) for field in sample_fields]
        print("  " + " | ".join(f"{value:>15}" for value in values))

if __name__ == "__main__":
    main()

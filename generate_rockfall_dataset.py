#!/usr/bin/env python3
"""
Simple script to generate rockfall prediction dataset with predefined settings
"""

from rockfall_prediction_dataset_generator import RockfallDatasetGenerator
from datetime import datetime

def main():
    print("🏔️  Synthetic Rockfall Prediction Dataset Generator")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_samples': 3000,
        'seed': 42,
        'start_date': datetime(2024, 1, 1),
        'days_span': 365,
        'output_file': 'rockfall_prediction_dataset.csv'
    }
    
    print(f"Configuration:")
    print(f"  Samples: {config['num_samples']}")
    print(f"  Date range: {config['start_date'].date()} to {(config['start_date']).replace(year=config['start_date'].year + 1).date()}")
    print(f"  Output: {config['output_file']}")
    print(f"  Random seed: {config['seed']}")
    print()
    
    # Generate dataset
    generator = RockfallDatasetGenerator(seed=config['seed'])
    
    print("Generating synthetic dataset...")
    df = generator.generate_dataset(
        num_samples=config['num_samples'],
        start_date=config['start_date'],
        days_span=config['days_span']
    )
    
    print("\nDataset Statistics:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Rockfall events: {df['rockfall_occurred'].sum():,}")
    print(f"  Safe observations: {(df['rockfall_occurred'] == 0).sum():,}")
    print(f"  Rockfall rate: {df['rockfall_occurred'].mean():.1%}")
    print(f"  Unique locations: {df['location_id'].nunique()}")
    
    # Show environmental scenarios distribution
    if '_scenario' in df.columns:
        scenario_dist = df['_scenario'].value_counts()
        print(f"\nEnvironmental Scenarios:")
        for scenario, count in scenario_dist.items():
            print(f"  {scenario}: {count:,} ({count/len(df):.1%})")
    
    # Show parameter ranges
    print(f"\nParameter Ranges:")
    key_params = ['rmr', 'slope_angle_deg', 'pore_pressure_kpa', 'surface_vel_mm_day', 'rainfall_1h_mm']
    for param in key_params:
        if param in df.columns:
            print(f"  {param}: {df[param].min():.1f} - {df[param].max():.1f}")
    
    # Save dataset
    print(f"\nSaving dataset...")
    final_df = generator.save_dataset(df, config['output_file'])
    
    print(f"\n✅ Dataset generation completed successfully!")
    print(f"📁 Files created:")
    print(f"   - {config['output_file']}")
    print(f"   - {config['output_file'].replace('.csv', '_summary.json')}")
    
    # Show sample data
    print(f"\n📊 Sample data (first 3 rows):")
    sample_cols = ['timestamp', 'location_id', 'rmr', 'slope_angle_deg', 
                   'pore_pressure_kpa', 'surface_vel_mm_day', 'rockfall_occurred']
    if all(col in final_df.columns for col in sample_cols):
        print(final_df[sample_cols].head(3).to_string(index=False))

if __name__ == "__main__":
    main()

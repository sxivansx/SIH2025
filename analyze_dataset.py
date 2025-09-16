#!/usr/bin/env python3
"""
Dataset Analysis Script for Rockfall Prediction Dataset
"""

import csv
import json
from collections import defaultdict

def analyze_dataset(csv_file):
    """Analyze the rockfall prediction dataset"""
    
    print("🔍 Rockfall Prediction Dataset Analysis")
    print("=" * 50)
    
    # Read data
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert numeric columns
    numeric_cols = ['rmr', 'slope_angle_deg', 'pore_pressure_kpa', 'surface_vel_mm_day', 
                   'surface_accel_mm_day2', 'rainfall_1h_mm', 'antecedent_rainfall_7d_mm',
                   'temp_celsius', 'blast_ppv_mms', 'weathering_grade']
    
    for row in data:
        for col in numeric_cols:
            if col in row:
                row[col] = float(row[col])
        row['rockfall_occurred'] = int(row['rockfall_occurred'])
    
    # Basic statistics
    total_obs = len(data)
    rockfall_events = sum(1 for row in data if row['rockfall_occurred'] == 1)
    safe_events = total_obs - rockfall_events
    
    print(f"📊 Basic Statistics:")
    print(f"   Total observations: {total_obs:,}")
    print(f"   Rockfall events: {rockfall_events:,} ({rockfall_events/total_obs:.1%})")
    print(f"   Safe observations: {safe_events:,} ({safe_events/total_obs:.1%})")
    
    # Separate rockfall and safe observations
    rockfall_data = [row for row in data if row['rockfall_occurred'] == 1]
    safe_data = [row for row in data if row['rockfall_occurred'] == 0]
    
    print(f"\n🎯 Parameter Comparison (Rockfall vs Safe):")
    
    # Analyze key parameters
    key_params = [
        ('rmr', 'Rock Mass Rating'),
        ('slope_angle_deg', 'Slope Angle (°)'),
        ('pore_pressure_kpa', 'Pore Pressure (kPa)'),
        ('surface_vel_mm_day', 'Surface Velocity (mm/day)'),
        ('surface_accel_mm_day2', 'Surface Acceleration (mm/day²)'),
        ('weathering_grade', 'Weathering Grade'),
        ('rainfall_1h_mm', 'Rainfall 1h (mm)')
    ]
    
    for param, name in key_params:
        rockfall_values = [row[param] for row in rockfall_data if param in row]
        safe_values = [row[param] for row in safe_data if param in row]
        
        if rockfall_values and safe_values:
            rf_mean = sum(rockfall_values) / len(rockfall_values)
            safe_mean = sum(safe_values) / len(safe_values)
            rf_min, rf_max = min(rockfall_values), max(rockfall_values)
            safe_min, safe_max = min(safe_values), max(safe_values)
            
            print(f"\n   {name}:")
            print(f"     Rockfall: {rf_mean:.1f} (range: {rf_min:.1f}-{rf_max:.1f})")
            print(f"     Safe:     {safe_mean:.1f} (range: {safe_min:.1f}-{safe_max:.1f})")
    
    # Environmental scenarios analysis
    print(f"\n🌦️  Environmental Scenarios:")
    
    # Analyze by rainfall levels
    dry_events = [row for row in data if row['rainfall_1h_mm'] < 5]
    moderate_rain = [row for row in data if 5 <= row['rainfall_1h_mm'] < 20]
    heavy_rain = [row for row in data if row['rainfall_1h_mm'] >= 20]
    
    print(f"   Dry conditions (< 5mm/h): {len(dry_events):,} observations")
    print(f"     Rockfall rate: {sum(1 for r in dry_events if r['rockfall_occurred'])/len(dry_events):.1%}")
    
    print(f"   Moderate rain (5-20mm/h): {len(moderate_rain):,} observations")
    if moderate_rain:
        print(f"     Rockfall rate: {sum(1 for r in moderate_rain if r['rockfall_occurred'])/len(moderate_rain):.1%}")
    
    print(f"   Heavy rain (>20mm/h): {len(heavy_rain):,} observations")
    if heavy_rain:
        print(f"     Rockfall rate: {sum(1 for r in heavy_rain if r['rockfall_occurred'])/len(heavy_rain):.1%}")
    
    # Critical threshold analysis
    print(f"\n⚠️  Critical Threshold Analysis:")
    
    thresholds = [
        ('rmr', 30, 'below'),
        ('slope_angle_deg', 55, 'above'),
        ('pore_pressure_kpa', 300, 'above'),
        ('surface_vel_mm_day', 50, 'above'),
        ('surface_accel_mm_day2', 20, 'above')
    ]
    
    for param, threshold, direction in thresholds:
        if direction == 'below':
            critical_obs = [row for row in data if row[param] < threshold]
        else:
            critical_obs = [row for row in data if row[param] > threshold]
        
        if critical_obs:
            critical_rockfall_rate = sum(1 for r in critical_obs if r['rockfall_occurred']) / len(critical_obs)
            print(f"   {param} {direction} {threshold}: {len(critical_obs)} obs, {critical_rockfall_rate:.1%} rockfall rate")
    
    # Location analysis
    print(f"\n🏗️  Location Analysis:")
    location_stats = defaultdict(lambda: {'total': 0, 'rockfall': 0})
    
    for row in data:
        pit = row['location_id'].split('_')[0]
        location_stats[pit]['total'] += 1
        if row['rockfall_occurred']:
            location_stats[pit]['rockfall'] += 1
    
    for pit in sorted(location_stats.keys()):
        stats = location_stats[pit]
        rate = stats['rockfall'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {pit}: {stats['total']:,} observations, {stats['rockfall']} rockfalls ({rate:.1%})")
    
    # Extreme events analysis
    print(f"\n🚨 Extreme Events Analysis:")
    
    # Find observations with multiple risk factors
    high_risk_obs = []
    for row in data:
        risk_factors = 0
        if row['rmr'] < 30:
            risk_factors += 1
        if row['slope_angle_deg'] > 55:
            risk_factors += 1
        if row['pore_pressure_kpa'] > 300:
            risk_factors += 1
        if row['surface_vel_mm_day'] > 50:
            risk_factors += 1
        if row['weathering_grade'] >= 4:
            risk_factors += 1
        
        if risk_factors >= 3:
            high_risk_obs.append(row)
    
    if high_risk_obs:
        high_risk_rockfall_rate = sum(1 for r in high_risk_obs if r['rockfall_occurred']) / len(high_risk_obs)
        print(f"   High-risk observations (3+ risk factors): {len(high_risk_obs)}")
        print(f"   Rockfall rate in high-risk conditions: {high_risk_rockfall_rate:.1%}")
    
    # Show most dangerous events
    print(f"\n💥 Most Critical Rockfall Events:")
    
    # Sort rockfall events by severity (combination of factors)
    def calculate_severity(row):
        severity = 0
        severity += max(0, 30 - row['rmr']) / 30  # RMR factor
        severity += max(0, row['slope_angle_deg'] - 30) / 40  # Slope factor
        severity += row['pore_pressure_kpa'] / 500  # Pore pressure factor
        severity += row['surface_vel_mm_day'] / 200  # Velocity factor
        severity += max(0, row['surface_accel_mm_day2']) / 50  # Acceleration factor
        return severity
    
    rockfall_data_with_severity = [(row, calculate_severity(row)) for row in rockfall_data]
    rockfall_data_with_severity.sort(key=lambda x: x[1], reverse=True)
    
    for i, (row, severity) in enumerate(rockfall_data_with_severity[:3]):
        print(f"\n   Event {i+1} (Severity: {severity:.2f}):")
        print(f"     Location: {row['location_id']}")
        print(f"     RMR: {row['rmr']}, Slope: {row['slope_angle_deg']:.1f}°")
        print(f"     Pore Pressure: {row['pore_pressure_kpa']:.1f} kPa")
        print(f"     Surface Vel: {row['surface_vel_mm_day']:.1f} mm/day")
        print(f"     Weathering: Grade {row['weathering_grade']}")

def main():
    csv_file = "rockfall_prediction_dataset.csv"
    analyze_dataset(csv_file)

if __name__ == "__main__":
    main()

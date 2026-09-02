import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from datetime import datetime

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the indoor CSV file
indoor_input_file = os.path.join(script_dir, 'Yavatmal Indoor Data.csv')
indoor_df = pd.read_csv(indoor_input_file)

# Read the outdoor CSV file
outdoor_input_file = os.path.join(script_dir, 'Yavatmal AWS Data.csv')
outdoor_df = pd.read_csv(outdoor_input_file)

# Combine Date and Time columns into a single datetime column for indoor data
indoor_df['Datetime'] = pd.to_datetime(indoor_df['DD/MM/YYYY'] + ' ' + indoor_df['Time'], format='%d-%m-%Y %I:%M:%S %p')

# Combine Date and Time columns into a single datetime column for outdoor data
outdoor_df['Datetime'] = pd.to_datetime(outdoor_df['DD/MM/YYYY'] + ' ' + outdoor_df['Time'], format='%d-%m-%y %H:%M:%S')

# Remove duplicate datetime entries
indoor_df = indoor_df.drop_duplicates(subset=['Datetime'])
outdoor_df = outdoor_df.drop_duplicates(subset=['Datetime'])

# Find all columns ending with (RH) in indoor data
indoor_rh_columns = [col for col in indoor_df.columns if col.endswith('(RH)')]

print(f"Found {len(indoor_rh_columns)} indoor humidity columns: {indoor_rh_columns}\n")

# Resample to daily averages instead of hourly
indoor_df_daily = indoor_df.set_index('Datetime').resample('1D').mean(numeric_only=True)
outdoor_df_daily = outdoor_df.set_index('Datetime').resample('1D').interpolate(method='linear')

# Apply 7-day rolling average for smoothing
indoor_df_resampled = indoor_df_daily.select_dtypes(include=['number']).rolling(window=7, center=True).mean()
outdoor_df_resampled = outdoor_df_daily.select_dtypes(include=['number']).rolling(window=7, center=True).mean()

# ============================================================================
# Plot 1: Comparative Time Series Plot
# ============================================================================
print("Generating: Indoor vs Outdoor Humidity Time Series Plot...")

plt.figure(figsize=(20, 10))

# Plot indoor RH columns
for rh_col in indoor_rh_columns:
    plt.plot(indoor_df_resampled.index, indoor_df_resampled[rh_col], 
             label=f'Indoor - {rh_col}', linestyle='-', linewidth=1.5)

# Plot outdoor humidity
plt.plot(outdoor_df_resampled.index, outdoor_df_resampled['Humidity'], 
         label='Outdoor', color='black', linestyle='--', linewidth=1.5)

plt.title('Indoor vs Outdoor Relative Humidity Over Time')
plt.xlabel('Datetime')
plt.ylabel('Relative Humidity (%)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'indoor_vs_outdoor_humidity.png'), dpi=300)
plt.close()

print("✓ Saved: indoor_vs_outdoor_humidity.png\n")

# ============================================================================
# Resampling and Alignment for Correlation Analysis
# ============================================================================
outdoor_df_resampled_10min = outdoor_df.set_index('Datetime').resample('10T').interpolate()

# ============================================================================
# Comparative Analysis Function with Hexbin Plot
# ============================================================================
def compare_humidity(indoor_col):
    """
    Compare indoor and outdoor humidity with hexbin density plot
    """
    # Prepare indoor data series with unique index
    indoor_series = indoor_df.set_index('Datetime')[indoor_col]
    indoor_series = indoor_series[~indoor_series.index.duplicated(keep='first')]
    
    # Prepare outdoor data
    outdoor_series = outdoor_df_resampled_10min['Humidity']
    
    # Align the dataframes
    aligned_data = pd.DataFrame({
        'Indoor': indoor_series,
        'Outdoor': outdoor_series
    }).dropna()
    
    # Calculate correlation
    correlation = aligned_data['Indoor'].corr(aligned_data['Outdoor'])
    
    # Calculate R-squared
    z = np.polyfit(aligned_data['Outdoor'], aligned_data['Indoor'], 1)
    p = np.poly1d(z)
    y_pred = p(aligned_data['Outdoor'])
    ss_res = np.sum((aligned_data['Indoor'] - y_pred) ** 2)
    ss_tot = np.sum((aligned_data['Indoor'] - aligned_data['Indoor'].mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create hexbin plot
    plt.figure(figsize=(10, 6))
    hexbin = plt.hexbin(aligned_data['Outdoor'], aligned_data['Indoor'], 
                        gridsize=25, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2)
    
    # Add colorbar
    cbar = plt.colorbar(hexbin, label='Data Point Count')
    
    # Add regression line
    if len(aligned_data) > 2:
        plt.plot(aligned_data['Outdoor'], p(aligned_data['Outdoor']), 
                 "b-", linewidth=2.5, label=f'Linear Fit (R² = {r_squared:.4f})')
    
    plt.title(f'Indoor vs Outdoor Humidity Correlation - {indoor_col}\n(Correlation: {correlation:.4f}, n={len(aligned_data)})', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Outdoor Humidity (%)', fontsize=11)
    plt.ylabel(f'Indoor Humidity - {indoor_col} (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save figure
    safe_col_name = indoor_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(script_dir, f'humidity_correlation_{safe_col_name}.png'), dpi=300)
    plt.close()
    
    return {
        'Column': indoor_col,
        'Correlation': correlation,
        'R_Squared': r_squared,
        'Data_Points': len(aligned_data)
    }

# ============================================================================
# Perform Correlation Analysis for Each Indoor RH Column
# ============================================================================
print("Generating correlation plots for each humidity column...\n")

correlations_list = []
for rh_col in indoor_rh_columns:
    print(f"  Processing: {rh_col}...")
    result = compare_humidity(rh_col)
    correlations_list.append(result)
    print(f"    ✓ Saved: humidity_correlation_{rh_col.replace('/', '_').replace(' ', '_')}.png")

print("\n" + "="*80)
print("HUMIDITY CORRELATION ANALYSIS RESULTS")
print("="*80)

# Create a summary dataframe
correlations_df = pd.DataFrame(correlations_list)
print("\n" + correlations_df.to_string(index=False))

print("\n" + "="*80)
print("OUTDOOR HUMIDITY STATISTICS")
print("="*80)
print(outdoor_df['Humidity'].describe())

print("\n" + "="*80)
print("DATETIME RANGES")
print("="*80)
print(f"Indoor Data  - Start: {indoor_df['Datetime'].min()}, End: {indoor_df['Datetime'].max()}")
print(f"Outdoor Data - Start: {outdoor_df['Datetime'].min()}, End: {outdoor_df['Datetime'].max()}")

print("\n" + "="*80)
print("✓ Analysis complete! All plots have been saved.")
print("="*80)

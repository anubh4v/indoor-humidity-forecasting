import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from datetime import datetime

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the indoor CSV file
indoor_input_file = os.path.join(script_dir, 'Delhi Indoor Data.csv')
indoor_df = pd.read_csv(indoor_input_file)

# Read the outdoor CSV file
outdoor_input_file = os.path.join(script_dir, 'Delhi AWS Data.csv')
outdoor_df = pd.read_csv(outdoor_input_file)

# ============================================================================
# INDOOR DATA DATETIME HANDLING
# ============================================================================
# Create masks for different date formats in indoor data
indoor_mask1 = indoor_df['DD/MM/YYYY'].str.contains(r'^\d{1,2}/\d{2}/\d{4}$', regex=True, na=False)  # M/D/YYYY format
indoor_mask2 = indoor_df['DD/MM/YYYY'].str.contains(r'^\d{2}-\d{2}-\d{4}$', regex=True, na=False)    # MM-DD-YYYY format

# Initialize the Datetime column
indoor_df['Datetime'] = pd.NaT

# Apply the appropriate format to each group for indoor data
indoor_df.loc[indoor_mask1, 'Datetime'] = pd.to_datetime(
    indoor_df.loc[indoor_mask1, 'DD/MM/YYYY'] + ' ' + indoor_df.loc[indoor_mask1, 'Time'], 
    format='%m/%d/%Y %I:%M:%S %p',
    errors='coerce'
)

indoor_df.loc[indoor_mask2, 'Datetime'] = pd.to_datetime(
    indoor_df.loc[indoor_mask2, 'DD/MM/YYYY'] + ' ' + indoor_df.loc[indoor_mask2, 'Time'], 
    format='%m-%d-%Y %I:%M:%S %p',
    errors='coerce'
)

# Check for unmatched indoor rows
indoor_unmatched = ~(indoor_mask1 | indoor_mask2)
if indoor_unmatched.any():
    print(f"Warning: {indoor_unmatched.sum()} indoor rows have unrecognized date formats")
    print("Unrecognized formats:", indoor_df.loc[indoor_unmatched, 'DD/MM/YYYY'].unique())

# Remove rows with NaT datetime values
indoor_df = indoor_df.dropna(subset=['Datetime'])

# ============================================================================
# OUTDOOR DATA DATETIME HANDLING (ORIGINAL)
# ============================================================================
# Combine Date and Time columns into a single datetime column for outdoor data
outdoor_df['Datetime'] = pd.to_datetime(
    outdoor_df['DD/MM/YYYY'] + ' ' + outdoor_df['Time'], 
    format='mixed',
    dayfirst=True
)

# Remove duplicate datetime entries
indoor_df = indoor_df.drop_duplicates(subset=['Datetime'])
outdoor_df = outdoor_df.drop_duplicates(subset=['Datetime'])

# ============================================================================
# FILTER OUTDOOR DATA TO MATCH INDOOR DATA RANGE
# ============================================================================
indoor_start = indoor_df['Datetime'].min()
indoor_end = indoor_df['Datetime'].max()

print(f"\nIndoor Data Range: {indoor_start} to {indoor_end}")
print(f"Outdoor Data Range (before filtering): {outdoor_df['Datetime'].min()} to {outdoor_df['Datetime'].max()}")

# Filter outdoor data to match indoor data's date range
outdoor_df_filtered = outdoor_df[(outdoor_df['Datetime'] >= indoor_start) & 
                                  (outdoor_df['Datetime'] <= indoor_end)].copy()

print(f"Outdoor Data Range (after filtering): {outdoor_df_filtered['Datetime'].min()} to {outdoor_df_filtered['Datetime'].max()}")

# Find all columns ending with (RH) in indoor data
indoor_rh_columns = [col for col in indoor_df.columns if col.endswith('(RH)')]

# Resample to daily averages instead of hourly
indoor_df_daily = indoor_df.set_index('Datetime').resample('1D').mean(numeric_only=True)
outdoor_df_daily = outdoor_df_filtered.set_index('Datetime').resample('1D').interpolate(method='linear')

# Apply 7-day rolling average for smoothing
indoor_df_resampled = indoor_df_daily.select_dtypes(include=['number']).rolling(window=7, center=True).mean()
outdoor_df_resampled = outdoor_df_daily.select_dtypes(include=['number']).rolling(window=7, center=True).mean()

# Comparative Time Series Plot
plt.figure(figsize=(20, 10))

# Plot indoor RH columns
for rh_col in indoor_rh_columns:
    plt.plot(indoor_df_resampled.index, indoor_df_resampled[rh_col], 
             label=f'Indoor - {rh_col}', linestyle='-', linewidth=1.5)

# Plot outdoor humidity
plt.plot(outdoor_df_resampled.index, outdoor_df_resampled['Out Hum'], 
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

# Resampling and Alignment
# Use the filtered outdoor data resampled to 10 minutes
outdoor_df_resampled_10min = outdoor_df_filtered.set_index('Datetime').resample('10T').interpolate()

# Comparative Analysis Function with Hexbin Plot
def compare_humidity(indoor_col):
    # Prepare indoor data series with unique index
    indoor_series = indoor_df.set_index('Datetime')[indoor_col]
    indoor_series = indoor_series[~indoor_series.index.duplicated(keep='first')]
    
    # Prepare outdoor data
    outdoor_series = outdoor_df_resampled_10min['Out Hum']
    
    # Align the dataframes
    aligned_data = pd.DataFrame({
        'Indoor': indoor_series,
        'Outdoor': outdoor_series
    }).dropna()
    
    # Correlation
    correlation = aligned_data['Indoor'].corr(aligned_data['Outdoor'])
    
    # Hexbin Plot
    plt.figure(figsize=(10, 8))
    hexbin = plt.hexbin(aligned_data['Outdoor'], aligned_data['Indoor'], 
                        gridsize=25, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2)
    
    plt.title(f'Indoor vs Outdoor Humidity Correlation - {indoor_col}\n(Correlation: {correlation:.4f})')
    plt.xlabel('Outdoor Humidity (%)')
    plt.ylabel(f'Indoor Humidity - {indoor_col} (%)')
    plt.colorbar(hexbin, label='Count')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add correlation line if enough data points
    if len(aligned_data) > 2:
        z = np.polyfit(aligned_data['Outdoor'], aligned_data['Indoor'], 1)
        p = np.poly1d(z)
        plt.plot(aligned_data['Outdoor'], p(aligned_data['Outdoor']), "b--", 
                 linewidth=2, label=f'Trend Line')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'humidity_correlation_{indoor_col}.png'), dpi=300)
    plt.close()
    
    return correlation

# Perform correlation analysis for each indoor RH column
correlations = {}
for rh_col in indoor_rh_columns:
    correlations[rh_col] = compare_humidity(rh_col)

# Print correlation results
print("\nHumidity Correlations:")
for rh_col, correlation in correlations.items():
    print(f"{rh_col}: {correlation:.4f}")

# Basic statistics
print("\nOutdoor Humidity Statistics (filtered to indoor data range):")
print(outdoor_df_filtered['Out Hum'].describe())

# Datetime ranges
print("\nDatetime Ranges:")
print(f"Indoor Data - Start: {indoor_df['Datetime'].min()}, End: {indoor_df['Datetime'].max()}")
print(f"Outdoor Data (filtered) - Start: {outdoor_df_filtered['Datetime'].min()}, End: {outdoor_df_filtered['Datetime'].max()}")

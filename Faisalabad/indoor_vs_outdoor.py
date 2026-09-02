import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from datetime import datetime

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the indoor CSV file
indoor_input_file = os.path.join(script_dir, 'Faisalabad Indoor Data.csv')
indoor_df = pd.read_csv(indoor_input_file)

# Read the outdoor CSV file
outdoor_input_file = os.path.join(script_dir, 'Faisalabad AWS Data.csv')
outdoor_df = pd.read_csv(outdoor_input_file)

# ============================================================================
# INDOOR DATA DATETIME HANDLING (UNCHANGED)
# ============================================================================
indoor_df['Datetime'] = pd.to_datetime(indoor_df['DD/MM/YYYY'] + ' ' + indoor_df['Time'], format='%d-%m-%Y %I:%M:%S %p')

# ============================================================================
# OUTDOOR DATA DATETIME HANDLING (ROBUST FORMAT DETECTION)
# ============================================================================
# Create masks for the two date formats in outdoor data
outdoor_mask1 = outdoor_df['DD/MM/YYYY'].str.contains(r'^\d{1,2}/\d{1,2}/\d{4}$', regex=True, na=False)  # M/D/YYYY format
outdoor_mask2 = outdoor_df['DD/MM/YYYY'].str.contains(r'^\d{2}-\d{2}-\d{2}$', regex=True, na=False)       # MM-DD-YY format

# Initialize the Datetime column
outdoor_df['Datetime'] = pd.NaT

# Apply the appropriate format to each group for outdoor data
outdoor_df.loc[outdoor_mask1, 'Datetime'] = pd.to_datetime(
    outdoor_df.loc[outdoor_mask1, 'DD/MM/YYYY'] + ' ' + outdoor_df.loc[outdoor_mask1, 'Time'], 
    format='%m/%d/%Y %I:%M:%S %p',
    errors='coerce'
)

outdoor_df.loc[outdoor_mask2, 'Datetime'] = pd.to_datetime(
    outdoor_df.loc[outdoor_mask2, 'DD/MM/YYYY'] + ' ' + outdoor_df.loc[outdoor_mask2, 'Time'], 
    format='%m-%d-%y %I:%M:%S %p',
    errors='coerce'
)

# Check for unmatched outdoor rows
outdoor_unmatched = ~(outdoor_mask1 | outdoor_mask2)
if outdoor_unmatched.any():
    print(f"Warning: {outdoor_unmatched.sum()} outdoor rows have unrecognized date formats")
    print("Unrecognized formats:", outdoor_df.loc[outdoor_unmatched, 'DD/MM/YYYY'].unique())

# Remove rows with NaT datetime values
outdoor_df = outdoor_df.dropna(subset=['Datetime'])

# Remove duplicate datetime entries
indoor_df = indoor_df.drop_duplicates(subset=['Datetime'])
outdoor_df = outdoor_df.drop_duplicates(subset=['Datetime'])
# ============================================================================
# FILTER OUTDOOR DATA TO MATCH INDOOR DATA RANGE
# ============================================================================
indoor_start = indoor_df['Datetime'].min()
indoor_end = indoor_df['Datetime'].max()

outdoor_df_filtered = outdoor_df[(outdoor_df['Datetime'] >= indoor_start) & 
                                  (outdoor_df['Datetime'] <= indoor_end)].copy()

print(f"Indoor Data Range:  {indoor_start} to {indoor_end}")
print(f"Outdoor Data Range (filtered): {outdoor_df_filtered['Datetime'].min()} to {outdoor_df_filtered['Datetime'].max()}\n")

# Find all columns ending with (RH) in indoor data
# indoor_rh_columns = [col for col in indoor_df.columns if col.endswith('(RH)')]
indoor_rh_columns = ["10692997 (RH)", "10692999 (RH)", "10699999 (RH)"]

print(f"Found {len(indoor_rh_columns)} indoor humidity columns: {indoor_rh_columns}\n")

# Resample to daily averages instead of hourly
indoor_df_daily = indoor_df.set_index('Datetime').resample('1D').mean(numeric_only=True)
outdoor_df_daily = outdoor_df_filtered.set_index('Datetime').resample('1D').interpolate(method='linear')

# Apply 7-day rolling average for smoothing
indoor_df_resampled = indoor_df_daily.select_dtypes(include=['number']).rolling(window=7, center=True).mean()
outdoor_df_resampled = outdoor_df_daily.select_dtypes(include=['number']).rolling(window=7, center=True).mean()

# ============================================================================
# Plot 1: Comparative Time Series Plot
# ============================================================================
print("Generating: Indoor vs Outdoor Humidity Time Series Plot...")

# Find the overlapping date range
start_date = max(indoor_df_resampled.index.min(), outdoor_df_resampled.index.min())
end_date = min(indoor_df_resampled.index.max(), outdoor_df_resampled.index.max())

# Filter both dataframes to the overlapping period
indoor_filtered = indoor_df_resampled.loc[start_date:end_date]
outdoor_filtered = outdoor_df_resampled.loc[start_date:end_date]

plt.figure(figsize=(20, 10))

# Plot indoor RH columns
for rh_col in indoor_rh_columns:
    plt.plot(indoor_filtered.index, indoor_filtered[rh_col], 
             label=f'Indoor - {rh_col}', linestyle='-', linewidth=1.5)

# Plot outdoor humidity
plt.plot(outdoor_filtered.index, outdoor_filtered['Out Hum'], 
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
outdoor_df_resampled_10min = outdoor_df_filtered.set_index('Datetime').resample('10T').interpolate()

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
    outdoor_series = outdoor_df_resampled_10min['Out Hum']
    
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
print("OUTDOOR HUMIDITY STATISTICS (FILTERED TO INDOOR DATA RANGE)")
print("="*80)
print(outdoor_df_filtered['Out Hum'].describe())

print("\n" + "="*80)
print("DATETIME RANGES")
print("="*80)
print(f"Indoor Data  - Start: {indoor_df['Datetime'].min()}, End: {indoor_df['Datetime'].max()}")
print(f"Outdoor Data (filtered) - Start: {outdoor_df_filtered['Datetime'].min()}, End: {outdoor_df_filtered['Datetime'].max()}")

print("\n" + "="*80)
print("✓ Analysis complete! All plots have been saved.")
print("="*80)

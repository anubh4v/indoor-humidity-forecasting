# Save this as humidity_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV file
input_file = os.path.join(script_dir, 'Delhi Indoor Data.csv')

# Read the CSV file
df = pd.read_csv(input_file)

# Combine Date and Time columns into a single datetime column
# df['Datetime'] = pd.to_datetime(df['DD/MM/YYYY'] + ' ' + df['Time'], format='%d-%m-%Y %I:%M:%S %p')

# Create a mask for each format
mask1 = df['DD/MM/YYYY'].str.contains(r'^\d{1,2}/\d{2}/\d{4}$', regex=True)  # 3/31/2016 format
mask2 = df['DD/MM/YYYY'].str.contains(r'^\d{2}-\d{2}-\d{4}$', regex=True)    # 04-13-2016 format

# Initialize the Datetime column
df['Datetime'] = pd.NaT

# Apply the appropriate format to each group
df.loc[mask1, 'Datetime'] = pd.to_datetime(
    df.loc[mask1, 'DD/MM/YYYY'] + ' ' + df.loc[mask1, 'Time'], 
    format='%m/%d/%Y %I:%M:%S %p'
)

df.loc[mask2, 'Datetime'] = pd.to_datetime(
    df.loc[mask2, 'DD/MM/YYYY'] + ' ' + df.loc[mask2, 'Time'], 
    format='%m-%d-%Y %I:%M:%S %p'
)

# Find all columns ending with (RH)
rh_columns = [col for col in df.columns if col.endswith('(RH)')]

# Create a figure with subplots for each RH column
plt.figure(figsize=(20, 5 * len(rh_columns)))

# Plot each RH column
for i, rh_col in enumerate(rh_columns, 1):
    plt.subplot(len(rh_columns), 1, i)
    
    # Line plot of humidity over time
    plt.plot(df['Datetime'], df[rh_col], label=rh_col, marker='o', linestyle='-', markersize=4)
    
    # Add horizontal lines for humidity thresholds
    plt.axhline(y=50, color='r', linestyle='--', label='Less Favourable Threshold')
    plt.axhline(y=65, color='g', linestyle='--', label='Highly Favourable Threshold')
    
    plt.title(f'Relative Humidity Over Time - {rh_col}')
    plt.xlabel('Datetime')
    plt.ylabel('Relative Humidity (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'humidity_over_time.png'), dpi=300)

# Optional: Monthly boxplot for each RH column
plt.figure(figsize=(20, 5 * len(rh_columns)))

for i, rh_col in enumerate(rh_columns, 1):
    plt.subplot(len(rh_columns), 1, i)
    
    # Create a month column
    df['Month'] = df['Datetime'].dt.to_period('M')
    
    # Boxplot of humidity by month
    sns.boxplot(x=df['Month'].astype(str), y=df[rh_col])
    
    plt.title(f'Monthly Relative Humidity Distribution - {rh_col}')
    plt.xlabel('Month')
    plt.ylabel('Relative Humidity (%)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'monthly_humidity_boxplot.png'), dpi=300)

# Optional: Hourly average humidity
plt.figure(figsize=(20, 5 * len(rh_columns)))

for i, rh_col in enumerate(rh_columns, 1):
    plt.subplot(len(rh_columns), 1, i)
    
    # Group by hour and calculate mean humidity
    hourly_humidity = df.groupby(df['Datetime'].dt.hour)[rh_col].mean()
    
    # Plot hourly average
    hourly_humidity.plot(kind='bar')
    
    plt.title(f'Average Hourly Humidity - {rh_col}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Relative Humidity (%)')
    plt.xticks(rotation=0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'hourly_average_humidity.png'), dpi=300)

print("\n=== Daily Diurnal Humidity Range (All Locations) ===\n")

all_daily_ranges = []

for rh_col in rh_columns:
    daily_range = df.groupby(df['Datetime'].dt.date)[rh_col].apply(lambda x: x.max() - x.min())
    all_daily_ranges.extend(daily_range.values)

all_daily_ranges = pd.Series(all_daily_ranges)

print(f"Average daily range: {all_daily_ranges.mean():.2f}%")
print(f"Max daily range: {all_daily_ranges.max():.2f}%")
print(f"Min daily range: {all_daily_ranges.min():.2f}%")

# Print basic statistics
print("\nHumidity Statistics:")
for rh_col in rh_columns:
    print(f"\nColumn: {rh_col}")
    print(df[rh_col].describe())

# Additional insights
print("\nDatetime Range:")
print(f"Start: {df['Datetime'].min()}")
print(f"End: {df['Datetime'].max()}")

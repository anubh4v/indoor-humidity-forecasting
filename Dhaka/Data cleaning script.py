import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV file
input_file = os.path.join(script_dir, 'Dhaka AWS Data.csv')

df = pd.read_csv(input_file, dtype=str)

# Function to standardize dates
def standardize_date(date_str):
    try:
        # Try MM-DD-YYYY format (with dashes)
        return pd.to_datetime(date_str, format='%m-%d-%Y').strftime('%d/%m/%Y')
    except:
        try:
            # Try M/D/YYYY format (with slashes)
            return pd.to_datetime(date_str, format='%m/%d/%Y').strftime('%d/%m/%Y')
        except:
            return date_str  # Return original if both fail

# Function to standardize times
def standardize_time(time_str):
    try:
        # Try 12-hour format with AM/PM (e.g., "2:00:00 PM")
        return pd.to_datetime(time_str, format='%I:%M:%S %p').strftime('%H:%M:%S')
    except:
        try:
            # Try 24-hour format (e.g., "8:00:00" or "08:00:00")
            return pd.to_datetime(time_str, format='%H:%M:%S').strftime('%H:%M:%S')
        except:
            return time_str  # Return original if both fail

# Apply the standardization functions
df['DD/MM/YYYY'] = df['DD/MM/YYYY'].apply(standardize_date)
df['Time'] = df['Time'].apply(standardize_time)

# Save the output
df.to_csv(os.path.join(script_dir, "output.csv"), index=False)

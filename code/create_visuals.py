import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---
# Set the current working directory to the project root for reliable pathing
# This assumes the script is run from the project root (D:\FlightDelayProject)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

# The path where Spark wrote the distributed results
# We use the explicit path from the root
RESULTS_BASE_PATH = os.path.join(PROJECT_ROOT, "data", "results")
VISUAL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "visuals")

# Ensure the output directory exists
os.makedirs(VISUAL_OUTPUT_PATH, exist_ok=True)
print(f"Saving visualizations to: {VISUAL_OUTPUT_PATH}\n")


def load_spark_output(folder_name):
    """Loads the scattered 'part-files' written by Spark into a single Pandas DataFrame."""
    
    # Construct the full path using the pre-defined base path
    full_path = os.path.join(RESULTS_BASE_PATH, folder_name)
    
    try:
        # Check if the folder exists first
        if not os.path.isdir(full_path):
            print(f"Error: Output directory not found at {full_path}")
            return None # Return None if the folder doesn't exist
            
        # The glob pattern finds all 'part-00xxx.csv' files inside the directory
        # Starts with os.listdir and then gets every f file which starts with part- and then feeds it as a list to csv_files variable but this only contains the whole path of the file and not the actual data 
        csv_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.startswith('part-')]

        if not csv_files:
            print(f"Error: No part-files found in {full_path}")
            return None # Return None if no files are found

        # Read all parts into a list of DataFrames and concatenate them
        df_list = [pd.read_csv(f) for f in csv_files]
        return pd.concat(df_list, ignore_index=True)

    except Exception as e:
        print(f"An unexpected error occurred while loading data from {full_path}: {e}")
        return None # Return None on any other exception


# --- Visualization 1: Top 10 Delayed Carriers (Bar Chart) ---
def plot_top_carriers():
    print("Creating Visualization 1: Top Carriers...")
    df = load_spark_output("top_carriers")
    
    # Check if df is None before attempting to use it
    if df is None:
        print("Skipping Visualization 1 due to missing data.")
        return
        
    df = df.sort_values('AVG_DEP_DELAY_MIN', ascending=False)
    
    # Plotting
    ## Creates a dummy figure
    plt.figure(figsize=(12, 6))
    ## creates a bar plot and gives it x and y as carriers and avg departure time
    sns.barplot(x='op_unique_carrier', y='AVG_DEP_DELAY_MIN', data=df, palette='viridis')
    ## Just sets the title of the graph
    plt.title('Top 10 Airlines by Average Departure Delay Time (Minutes)', fontsize=16)
    ## gives the x lable and y lable
    plt.xlabel('Airline Carrier (OP_CARRIER)', fontsize=12)
    plt.ylabel('Average Delay (Minutes)', fontsize=12)
    ## rotates the labels on the x axis to 45 degres
    plt.xticks(rotation=45, ha='right')
    ## Sets the compact flow and compacts the graph 
    plt.tight_layout()
    ## Saves the file to the root folder and then names it xxx
    plt.savefig(os.path.join(VISUAL_OUTPUT_PATH, '1_Top_Delayed_Carriers.png'))
    ## Closes the plot
    plt.close()
    print("Visualization 1 saved.")


# --- Visualization 2: Delay Count by Hour of Day (Line Chart) ---
def plot_hourly_delays():
    print("Creating Visualization 2: Hourly Delay Trend...")
    df = load_spark_output("hourly_delays")
    df = df.sort_values('DEP_HOUR')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='DEP_HOUR', y='DELAY_COUNT', data=df, marker='o', color='red')
    plt.title('Total Delays by Scheduled Departure Hour (24h)', fontsize=16)
    plt.xlabel('Departure Hour', fontsize=12)
    plt.ylabel('Total Delayed Flights (Count)', fontsize=12)
    ## This is done so that the x hours will be not 0 5 10 etc but constant from 0 to 23 to represent actual hours
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_OUTPUT_PATH, '2_Hourly_Delay_Trend.png'))
    plt.close()
    print("Visualization 2 saved.")


# --- Visualization 3: Delay Rate by Distance Group (Bar Chart with Rate) ---
def plot_distance_rate():
    print("Creating Visualization 3: Delay Rate by Distance...")
    df = load_spark_output("distance_rate")
    df = df.sort_values('DELAY_RATE', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DISTANCE_GROUP', y='DELAY_RATE', data=df, palette='cividis')
    plt.title('Flight Delay Rate by Distance Category', fontsize=16)
    plt.xlabel('Distance Category', fontsize=12)
    plt.ylabel('Delay Rate (%)', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_OUTPUT_PATH, '3_Distance_Delay_Rate.png'))
    plt.close()
    print("Visualization 3 saved.")


# --- Visualization 4: Primary Delay Cause Breakdown (Pie Chart) ---
def plot_delay_causes():
    print("Creating Visualization 4: Delay Cause Breakdown (Pie Chart)...")
    df = load_spark_output("delay_cause_breakdown")

    if df is None:
        print("Skipping Visualization 4 due to missing data.")
        return

    # Calculate percentage for the pie chart
    total_minutes = df['TOTAL_DELAY_MINUTES'].sum()
    ## The percentage column is created on the data frame and is not stored like anywhere not on the file at all
    df['PERCENTAGE'] = (df['TOTAL_DELAY_MINUTES'] / total_minutes) * 100
    df = df.sort_values('TOTAL_DELAY_MINUTES', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    # Use Pandas plot for a simple pie chart
    plt.pie(
        df['PERCENTAGE'], 
        labels=df['PRIMARY_DELAY_CAUSE'], 
        ## Minimum width is 1 from %1 and .1f is have accuracy of one integer after point 
        autopct='%1.1f%%', 
        startangle=90, 
        textprops={'fontsize': 12}, 
        colors=sns.color_palette('pastel')
    )
    plt.title('Distribution of Total Delay Minutes by Primary Cause', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_OUTPUT_PATH, '4_Primary_Delay_Causes.png'))
    plt.close()
    print("Visualization 4 saved.")


if __name__ == "__main__":
    plot_top_carriers()
    plot_hourly_delays()
    plot_distance_rate()
    plot_delay_causes()
    print("\n--- All visualizations saved successfully to the 'visuals' folder! ---")
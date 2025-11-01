import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load time-series CSVs with the correct structure
active_path = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
gen_path = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
reactive_path = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

# index_col=0 makes the first column ('Node' or 'type') the DataFrame index
active_df = pd.read_csv(active_path, index_col=0)
gen_df = pd.read_csv(gen_path, index_col=0)
reactive_df = pd.read_csv(reactive_path, index_col=0)

def plot_timeseries_profile(df, entity, days=1):
    """
    Plot a time series profile for a given node (bus) or type (for generation).

    Parameters:
    - df: DataFrame with index (Node/type) and columns (t0, t1, ...)
    - entity: String, index name (bus number for consumption or 'solar'/'wind' for generation)
    - days: Number of days to plot (24h each)
    """
    hours = 24 * days
    if entity not in df.index:
        print(f"Error: {entity} not found in data index. Available: {list(df.index)[:5]} ...")
        return
    time_indices = df.columns[:hours]
    data = df.loc[entity, time_indices]  # Select the correct row and time range
    plt.figure(figsize=(12, 5))
    plt.plot(range(hours), data.values, marker='o', linestyle='-')
    plt.title(f"Time Series Profile for {entity} - First {days} Day(s)")
    plt.xlabel('Hour')
    plt.ylabel('Value (kW or kVar)')
    plt.grid(True)
    plt.show()

# Example 1: Plot active power consumption for a selected bus/node, e.g., '10'
node_number = 10  # Replace with real node number as string (use active_df.index to list options)
days_to_plot = 3
plot_timeseries_profile(active_df, node_number, days_to_plot)

# Example 2: Plot generation for 'solar' or 'wind'
plot_timeseries_profile(gen_df, 'solar', days_to_plot)
plot_timeseries_profile(gen_df, 'wind', days_to_plot)

# Example 3: Plot reactive power consumption for a selected bus/node, e.g., '10'
plot_timeseries_profile(reactive_df, node_number, days_to_plot)

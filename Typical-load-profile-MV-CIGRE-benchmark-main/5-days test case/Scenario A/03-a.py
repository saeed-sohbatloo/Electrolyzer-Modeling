"""
03_timeseries_network_loading.py

This script:
- Loads the CIGRE MV test network.
- Loads time-series active power profiles from a CSV file for each node (bus).
- Maps node numbers in the profile file to pandapower load elements.
- Attaches a time-series controller (ConstControl) to each load for dynamic simulation.
- (You can extend it for generation/reactive in next steps.)

Assumptions:
- The number and order of nodes in 'Active_Node_Consumption.csv' match the loads in net.load.
- All files exist at specified paths.
"""

import pandas as pd
import pandapower.networks as pn
import pandapower as pp
import matplotlib.pyplot as plt
from pandapower.control import ConstControl

# STEP 1: Load the CIGRE MV network
net = pn.create_cigre_network_mv(with_der=False)

# STEP 2: Load the active node consumption profile
active_path = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
active_df = pd.read_csv(active_path, index_col=0)

# Confirm profile shape and index
print("Active node consumption profile shape:", active_df.shape)
print("First five nodes:", active_df.index[:5])

# STEP 3: Map profile nodes to pandapower load elements
# Assumes direct mapping: row index in active_df corresponds to row in net.load
# If you have a mapping table, use that instead.

# Get all load element indices
load_indices = net.load.index.tolist()

# Make sure number of loads matches number of profile nodes
if len(active_df) != len(load_indices):
    print("WARNING: Number of loads and profile nodes do not match.")
    print("Loads in network:", len(load_indices), "; Nodes in profile:", len(active_df))

# STEP 4: Attach ConstControl time-series controllers to each load for active power (p_mw)
# Pandapower expects MW, CSV is usually in kW, so divide by 1000

for idx, node in enumerate(active_df.index):
    if idx >= len(load_indices):
        print(f"Warning: Profile index {idx} exceeds number of loads in network.")
        continue
    # Extract the time series data for this node as a numpy array and convert to MW
    # DataFrame columns are typically t0, t1, ... for each hour
    profile_series = active_df.loc[node].values / 1000.0  # Convert kW to MW
    ConstControl(net, element='load', element_index=load_indices[idx], variable='p_mw',
                 profile_name='active_power_' + str(node), data_source=profile_series)

print("ConstControl(s) successfully added to all loads.")

# STEP 5: Run a time-series simulation loop for the first N time-steps (hours)
n_hours = 24  # Number of hours (or active_df.shape[1])
v_profile = []

for t in range(n_hours):
    # Update all loads for this time-step
    for i, ctrl in enumerate(net.controller):
        # each ConstControl expects a value for this timestep
        ctrl.data_source = active_df.iloc[i].values[t] / 1000.0
    # Run power flow
    pp.runpp(net)
    # Store the voltage of a specific bus (the last bus in this example)
    v_profile.append(net.res_bus.vm_pu.iloc[-1])  # You can change to any bus

# STEP 6: Plot the voltage profile of the selected bus over simulation
plt.figure(figsize=(10,4))
plt.plot(range(n_hours), v_profile, marker='o')
plt.title("Voltage profile at the end bus (over time)")
plt.xlabel("Hour")
plt.ylabel("Voltage (p.u.)")
plt.grid(True)
plt.show()

# Educational notes and tips:
# - Use the print statements to understand the data mapping.
# - Inspect net.load and active_df carefully if you have mapping issues.
# - You can expand the code in the same way for reactive power (q_mvar) or generation.
# - Use different profiles for other elements by adapting ConstControl assignments.

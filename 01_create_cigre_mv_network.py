"""
01_create_cigre_mv_network.py

This script creates and visualizes the CIGRE MV test network using pandapower.
No time-series or external profile data is loaded yet.
All code comments and explanations are in English for international research usage.
"""

import pandapower.networks as pn
import pandapower as pp
import matplotlib.pyplot as plt

# Step 1: Create the CIGRE MV test network
# Option: Set 'with_der=True' to include distributed energy resources (DER) generation profiles
# Option: Set 'with_der=False' for a basic passive network
net = pn.create_cigre_network_mv(with_der=False)  # Currently disables DERs, only passive loads

# Step 2: Network summary printout – buses, lines, trafo, loads, generation units, etc.
print("Network tables overview:")
print(net)

# Step 3: Draw the single line diagram for network topology review
# Option: 'simple_plot(net)' produces a static diagram (for large networks can be slow)
pp.plotting.simple_plot(net)

# Step 4: Run a basic power flow (load flow) calculation
pp.runpp(net)

# Find the index of the end (last) bus for profile extraction (can be edited to target another bus)
end_bus_index = net.bus.index[-1]
end_bus_voltage = net.res_bus.vm_pu.loc[end_bus_index]
print(f"End bus voltage (pu): {end_bus_voltage}")

# Step 5: Plot voltage profile for all buses after load flow calculation
# Option: Change x-axis label ('Bus Index') if using different bus naming schemes
plt.figure(figsize=(10, 4))
plt.plot(net.res_bus.vm_pu.values, marker='o')
plt.title('Bus Voltage Profile - Base Case')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.grid(True)
plt.show()

# Optional Steps:
# - Save network and result tables to CSV or Excel: pd.DataFrame(net.bus).to_csv('bus_data.csv')
# - Customize plot style, export images, etc.


net.load
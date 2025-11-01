# 02_timeseries_powerflow_cigre_electrolyzer_fixed.py
"""
Time-series power flow for CIGRE MV network with electrolyzer.
- Properly adds electrolyzer to the bus while keeping base loads.
- Saves and plots voltage before/after electrolyzer for comparison.
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# User configuration
# -------------------------
# Paths to your CSVs (put your real paths here):
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"


START_HOUR = 0
DURATION_HOURS = 24
OUT_DIR = "./timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer configuration
ELECTROLYZER_BUS = 10
ELECTROLYZER_POWER_MW = 10.0
ELECTROLYZER_PF = 0.95

# -------------------------
# Helper functions
# -------------------------
def read_timeseries_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df

def map_node_to_bus(node_label, net):
    try:
        maybe_idx = int(node_label)
        if maybe_idx in net.bus.index:
            return maybe_idx
    except:
        pass
    if 'name' in net.bus.columns:
        for idx, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return idx
    for idx in net.bus.index:
        if str(idx) == str(node_label).strip():
            return idx
    return None

def assign_loads(net, bus_idx, p_mw, q_mvar=0.0):
    """Assign or update load at a bus."""
    loads = net.load[net.load.bus == bus_idx]
    if len(loads) > 0:
        for li in loads.index:
            net.load.at[li, 'p_mw'] = p_mw / len(loads)
            net.load.at[li, 'q_mvar'] = q_mvar / len(loads)
    else:
        pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=q_mvar)

def assign_generation_profile(net, gen_label, p_mw):
    bus_idx = map_node_to_bus(gen_label, net)
    if bus_idx is None:
        bus_idx = net.bus.index[0]
    sgens = net.sgen[net.sgen.bus == bus_idx]
    if len(sgens) > 0:
        for si in sgens.index:
            net.sgen.at[si,'p_mw'] = p_mw / len(sgens)
    else:
        pp.create_sgen(net, bus=bus_idx, p_mw=p_mw, q_mvar=0.0)

# -------------------------
# Main time-series loop
# -------------------------
def run_timeseries(net, active_df, reactive_df=None, gen_df=None,
                   electrolyzer_bus=None, electrolyzer_p_mw=0.0,
                   electrolyzer_pf=0.95):

    elec_q = electrolyzer_p_mw * np.tan(np.arccos(electrolyzer_pf))

    bus_voltage_before = pd.DataFrame(index=range(DURATION_HOURS), columns=net.bus.index)
    bus_voltage_after = pd.DataFrame(index=range(DURATION_HOURS), columns=net.bus.index)

    node_to_bus_map = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    for t in range(DURATION_HOURS):
        col = active_df.columns[START_HOUR + t]

        # --- Assign base loads for this timestep ---
        for node in active_df.index:
            bus_idx = node_to_bus_map[str(node)]
            if bus_idx is None:
                continue
            p_mw = active_df.at[node, col]
            q_mw = reactive_df.at[node, col] if reactive_df is not None else 0.0
            assign_loads(net, bus_idx, p_mw, q_mw)

        # --- Run PF before electrolyzer ---
        try:
            pp.runpp(net)
            bus_voltage_before.loc[t, :] = net.res_bus.vm_pu
        except pp.LoadflowNotConverged:
            bus_voltage_before.loc[t, :] = np.nan

        # --- Add electrolyzer ---
        if electrolyzer_bus is not None and electrolyzer_p_mw > 0:
            assign_loads(net, electrolyzer_bus, electrolyzer_p_mw, elec_q)

        # --- Assign generation ---
        if gen_df is not None:
            for gen_label in gen_df.index:
                assign_generation_profile(net, gen_label, gen_df.at[gen_label, col])

        # --- Run PF after electrolyzer ---
        try:
            pp.runpp(net)
            bus_voltage_after.loc[t, :] = net.res_bus.vm_pu
        except pp.LoadflowNotConverged:
            bus_voltage_after.loc[t, :] = np.nan

    # --- Save results ---
    bus_voltage_before.to_csv(os.path.join(OUT_DIR,'bus_voltage_before.csv'))
    bus_voltage_after.to_csv(os.path.join(OUT_DIR,'bus_voltage_after.csv'))

    # --- Plot voltage comparison at electrolyzer bus ---
    if electrolyzer_bus is not None:
        plt.figure(figsize=(10,5))
        plt.plot(bus_voltage_before.index, bus_voltage_before[electrolyzer_bus], label='Before Electrolyzer')
        plt.plot(bus_voltage_after.index, bus_voltage_after[electrolyzer_bus], label='After Electrolyzer')
        plt.xlabel('Hour')
        plt.ylabel('Voltage (p.u.)')
        plt.title(f'Voltage at Bus {electrolyzer_bus}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,f'voltage_comparison_bus_{electrolyzer_bus}.png'))
        plt.show()

    return bus_voltage_before, bus_voltage_after

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der=False)
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH) if REACTIVE_PATH else None
    gen_df = read_timeseries_csv(GEN_PATH) if GEN_PATH else None

    bus_before, bus_after = run_timeseries(net, active_df, reactive_df, gen_df,
                                           electrolyzer_bus=ELECTROLYZER_BUS,
                                           electrolyzer_p_mw=ELECTROLYZER_POWER_MW,
                                           electrolyzer_pf=ELECTROLYZER_PF)

    print("✅ Completed. Voltages saved and plotted.")

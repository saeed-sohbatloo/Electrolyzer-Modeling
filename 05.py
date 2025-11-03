"""
Controlled time-series power flow for CIGRE MV network
Electrolyzer ON/OFF control logic (decision applies to next hour)
"""
"""
Controlled Time-Series Power Flow Simulation for a Medium-Voltage CIGRE Network

This script implements a time-series power flow analysis for a medium-voltage distribution network
using the pandapower library, with a focus on the operation of an electrolyzer under local ON/OFF control.

Key Features:
1. Loads time-series CSV files for active loads, reactive loads, and distributed generation.
2. Maps CSV nodes to network buses.
3. Simulates three scenarios:
   a) Network without the electrolyzer.
   b) Network with the electrolyzer always ON.
   c) Network with the electrolyzer controlled via an ON/OFF rule:
      - The control logic uses results from the previous hour to decide
        whether the electrolyzer should be ON or OFF in the next hour.
      - Decision criteria include maximum line loading and bus voltage at the electrolyzer bus.
4. Collects results including bus voltages, line loadings, transformer loadings, and electrolyzer status.
5. Generates plots for:
   - Voltage at the electrolyzer bus (all three scenarios)
   - Loading of lines connected to the electrolyzer bus (all three scenarios)
   - Transformer loading (all three scenarios)
   - Electrolyzer ON/OFF status over time

Execution Flow:
1. Load network and CSV time series.
2. Initialize the electrolyzer at the specified bus (initially OFF).
3. For each hour:
   a) Apply base loads and generation profiles.
   b) Run power flow with the current state of the electrolyzer.
   c) Record results (voltages, line/trafo loadings).
   d) Determine the electrolyzer status for the next hour based on control rules.
4. After completing all hours, generate plots for analysis.

This script is intended for studying the impact of controlled flexible loads (electrolyzer)
on voltage profiles, line loadings, and transformer utilization in a time-series simulation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn

# -------------------------
# USER CONFIGURATION
# -------------------------
## Paths to your CSVs (put your real paths here):
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"


DURATION_HOURS = 120
OUT_DIR = "timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

ELECTROLYZER_BUS = None  # will set interactively
ELECTROLYZER_P_NOMINAL_MW = 1.0
ELECTROLYZER_PF = 0.95
LINE_LOADING_THRESHOLD = 100.0
VOLTAGE_THRESHOLD_PU = 0.98

# Highlight lines/buses for plots
HIGHLIGHT_LINES = []
HIGHLIGHT_BUSES = []

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def read_timeseries_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df

def map_node_to_bus(node_label, net):
    try:
        idx = int(node_label)
        if idx in net.bus.index:
            return idx
    except Exception:
        pass
    if 'name' in net.bus.columns:
        for i, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return i
    for i in net.bus.index:
        if str(i) == str(node_label).strip():
            return i
    return None

def ensure_electrolyzer_load(net, bus_idx, name="Electrolyzer_controlled"):
    existing = net.load[net.load['name'] == name]
    if len(existing) > 0:
        return existing.index[0]
    return pp.create_load(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=name)

def set_electrolyzer_power(net, name, p_mw, q_mvar):
    idx = net.load[net.load['name'] == name].index
    if len(idx) == 0:
        raise ValueError(f"Electrolyzer load '{name}' not found")
    i = idx[0]
    net.load.at[i, 'p_mw'] = float(p_mw)
    net.load.at[i, 'q_mvar'] = float(q_mvar)

def assign_base_loads(net, active_df, reactive_df, node_to_bus_map, t_col, electrolyzer_name):
    bus_aggregate = {}
    for node in active_df.index:
        bus_idx = node_to_bus_map.get(str(node))
        if bus_idx is None:
            continue
        p_mw = float(active_df.at[node, t_col])
        q_mvar = float(reactive_df.at[node, t_col]) if reactive_df is not None else 0.0
        prev = bus_aggregate.get(bus_idx, (0.0,0.0))
        bus_aggregate[bus_idx] = (prev[0]+p_mw, prev[1]+q_mvar)

    for bus_idx, (p_total,q_total) in bus_aggregate.items():
        loads_at_bus = net.load[(net.load.bus==bus_idx) & (net.load.name != electrolyzer_name)]
        if len(loads_at_bus) > 0:
            total_existing = loads_at_bus.p_mw.sum()
            for li in loads_at_bus.index:
                share = loads_at_bus.at[li,'p_mw']/total_existing if total_existing>0 else 1.0/len(loads_at_bus)
                net.load.at[li,'p_mw'] = p_total*share
                net.load.at[li,'q_mvar'] = q_total*share
        else:
            pp.create_load(net, bus=bus_idx, p_mw=p_total, q_mvar=q_total, name=f"ts_load_bus_{bus_idx}")

def assign_generation_profiles(net, gen_df, node_to_bus_map, t_col):
    if gen_df is None:
        return
    for gen_label in gen_df.index:
        p_mw = float(gen_df.at[gen_label, t_col])
        mapped_bus = map_node_to_bus(gen_label, net)
        if mapped_bus is None:
            mapped_bus = int(net.bus.index[0])
        sgens_at_bus = net.sgen[net.sgen.bus==mapped_bus]
        if len(sgens_at_bus)>0:
            for si in sgens_at_bus.index:
                net.sgen.at[si,'p_mw'] = p_mw/len(sgens_at_bus)
        else:
            pp.create_sgen(net, bus=mapped_bus, p_mw=p_mw, q_mvar=0.0, name=f"ts_sgen_bus_{mapped_bus}")

# -------------------------
# TIMESERIES FUNCTION
# -------------------------
def run_timeseries(net, active_df, reactive_df, gen_df,
                   electrolyzer_bus, p_nominal, pf,
                   line_thresh, v_thresh, duration_hours):

    ELEC_NAME = "Electrolyzer_controlled"
    ensure_electrolyzer_load(net, electrolyzer_bus, ELEC_NAME)
    set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)

    node_to_bus = {str(n): map_node_to_bus(n, net) for n in active_df.index}

    hours = list(range(duration_hours))

    # Store results
    bus_v = {"none": pd.DataFrame(index=hours, columns=net.bus.index),
             "always": pd.DataFrame(index=hours, columns=net.bus.index),
             "controlled": pd.DataFrame(index=hours, columns=net.bus.index)}
    line_l = {"none": pd.DataFrame(index=hours, columns=net.line.index),
              "always": pd.DataFrame(index=hours, columns=net.line.index),
              "controlled": pd.DataFrame(index=hours, columns=net.line.index)}
    trafo_l = {"none": pd.DataFrame(index=hours, columns=net.trafo.index),
               "always": pd.DataFrame(index=hours, columns=net.trafo.index),
               "controlled": pd.DataFrame(index=hours, columns=net.trafo.index)}
    elec_status = pd.Series(index=hours, dtype=object)

    q_nominal = p_nominal * np.tan(np.arccos(pf))

    # -------------------
    # NETWORK WITHOUT ELECTROLYZER
    # -------------------
    set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)
    for t,col in enumerate(active_df.columns[:duration_hours]):
        assign_base_loads(net, active_df, reactive_df, node_to_bus, col, ELEC_NAME)
        assign_generation_profiles(net, gen_df, node_to_bus, col)
        pp.runpp(net)
        bus_v["none"].loc[t,:] = net.res_bus.vm_pu.values
        line_l["none"].loc[t,:] = net.res_line.loading_percent.values
        trafo_l["none"].loc[t,:] = net.res_trafo.loading_percent.values if len(net.trafo)>0 else 0.0

    # -------------------
    # NETWORK WITH ELECTROLYZER ALWAYS ON
    # -------------------
    set_electrolyzer_power(net, ELEC_NAME, p_nominal, q_nominal)
    for t,col in enumerate(active_df.columns[:duration_hours]):
        assign_base_loads(net, active_df, reactive_df, node_to_bus, col, ELEC_NAME)
        assign_generation_profiles(net, gen_df, node_to_bus, col)
        pp.runpp(net)
        bus_v["always"].loc[t,:] = net.res_bus.vm_pu.values
        line_l["always"].loc[t,:] = net.res_line.loading_percent.values
        trafo_l["always"].loc[t,:] = net.res_trafo.loading_percent.values if len(net.trafo)>0 else 0.0

    # -------------------
    # NETWORK WITH CONTROLLED ELECTROLYZER
    # -------------------
    # initialize electrolyzer OFF for hour 0
    elec_next_on = False
    for t,col in enumerate(active_df.columns[:duration_hours]):
        assign_base_loads(net, active_df, reactive_df, node_to_bus, col, ELEC_NAME)
        assign_generation_profiles(net, gen_df, node_to_bus, col)

        # Apply electrolyzer from previous decision
        if elec_next_on:
            set_electrolyzer_power(net, ELEC_NAME, p_nominal, q_nominal)
            elec_status.loc[t] = "ON"
        else:
            set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)
            elec_status.loc[t] = "OFF"

        pp.runpp(net)
        bus_v["controlled"].loc[t,:] = net.res_bus.vm_pu.values
        line_l["controlled"].loc[t,:] = net.res_line.loading_percent.values
        trafo_l["controlled"].loc[t,:] = net.res_trafo.loading_percent.values if len(net.trafo)>0 else 0.0

        # Decide for next hour
        v_at_bus = net.res_bus.vm_pu.at[electrolyzer_bus]
        max_line = net.res_line.loading_percent.max()
        elec_next_on = (max_line < line_thresh) and (v_at_bus > v_thresh)

    return bus_v, line_l, trafo_l, elec_status

# -------------------------
# MAIN
# -------------------------
if __name__=="__main__":
    net = pn.create_cigre_network_mv()
    print(net)

    if ELECTROLYZER_BUS is None:
        ELECTROLYZER_BUS = int(input("Enter the electrolyzer bus index: "))

    HIGHLIGHT_BUSES.append(ELECTROLYZER_BUS)

    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH) if os.path.exists(REACTIVE_PATH) else None
    gen_df = read_timeseries_csv(GEN_PATH) if os.path.exists(GEN_PATH) else None

    bus_v, line_l, trafo_l, elec_status = run_timeseries(
        net, active_df, reactive_df, gen_df,
        ELECTROLYZER_BUS, ELECTROLYZER_P_NOMINAL_MW, ELECTROLYZER_PF,
        LINE_LOADING_THRESHOLD, VOLTAGE_THRESHOLD_PU,
        DURATION_HOURS
    )

    # -------------------------
    # PLOTTING
    # -------------------------
    hours = list(range(DURATION_HOURS))

    # 1) Voltage at electrolyzer bus
    plt.figure(figsize=(12,5))
    plt.plot(hours, bus_v["none"][ELECTROLYZER_BUS], label="Without Electrolyzer", linestyle='--')
    plt.plot(hours, bus_v["always"][ELECTROLYZER_BUS], label="Electrolyzer Always ON", linestyle='-')
    plt.plot(hours, bus_v["controlled"][ELECTROLYZER_BUS], label="Electrolyzer Controlled", linestyle='-.')
    plt.xlabel("Hour")
    plt.ylabel("Voltage [p.u.]")
    plt.title(f"Voltage at Bus {ELECTROLYZER_BUS}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2) Lines connected to electrolyzer bus
    connected_lines = net.line[(net.line.from_bus==ELECTROLYZER_BUS) | (net.line.to_bus==ELECTROLYZER_BUS)].index
    plt.figure(figsize=(12,5))
    for li in connected_lines:
        plt.plot(hours, line_l["none"][li], '--', label=f"Line {li} No Elec")
        plt.plot(hours, line_l["always"][li], '-', label=f"Line {li} Always ON")
        plt.plot(hours, line_l["controlled"][li], '-.', label=f"Line {li} Controlled")
    plt.xlabel("Hour")
    plt.ylabel("Line Loading [%]")
    plt.title(f"Lines connected to Bus {ELECTROLYZER_BUS}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3) Transformer loading
    plt.figure(figsize=(12,5))
    for tr in trafo_l["none"].columns:
        plt.plot(hours, trafo_l["none"][tr], '--', label=f"Trafo {tr} No Elec")
        plt.plot(hours, trafo_l["always"][tr], '-', label=f"Trafo {tr} Always ON")
        plt.plot(hours, trafo_l["controlled"][tr], '-.', label=f"Trafo {tr} Controlled")
    plt.xlabel("Hour")
    plt.ylabel("Loading [%]")
    plt.title("Transformer Loading Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4) Electrolyzer ON/OFF status
    plt.figure(figsize=(12,3))
    status_numeric = elec_status.replace({'ON': 1, 'OFF': 0}).infer_objects()
    plt.step(hours, status_numeric, where='mid', label='Electrolyzer ON/OFF')
    plt.xlabel("Hour")
    plt.ylabel("ON=1 / OFF=0")
    plt.title("Electrolyzer Status Time Series")
    plt.grid(True)
    plt.show()

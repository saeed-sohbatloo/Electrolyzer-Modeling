import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn

"""
This script simulates a time-series power flow for the Cigre MV network with two scenarios:
1. Without electrolyzer.
2. With an electrolyzer that:
   - Switches ON/OFF based on local voltage and line loading.
   - Acts as an AVR (voltage controller) by adjusting reactive power (Q) within given limits.

Author: Saeed Sohbatloo
"""

# -------------------------
# === USER CONFIGURATION ===
# -------------------------
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

START_HOUR = 0
DURATION_HOURS = 72
OUT_DIR = "./timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer parameters
ELECTROLYZER_BUS = None  # will be entered by user
ELECTROLYZER_P_NOMINAL_MW = 1.0
ELECTROLYZER_PF = 0.95
LINE_LOADING_THRESHOLD = 100.0
VOLTAGE_THRESHOLD_PU = 0.98

# Reactive power control parameters
Q_MAX = 0.5    # Maximum reactive capability in Mvar
DQ_OVER = 0.06
DQ_UNDER = 0.04
V_DB = 0.01
V_OVER = 1 - V_DB
V_UNDER = 1 + V_DB

# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

def map_node_to_bus(node_label, net: pp.pandapowerNet):
    try:
        idx = int(node_label)
        if idx in net.bus.index:
            return idx
    except:
        pass
    if 'name' in net.bus.columns:
        for i, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return i
    for i in net.bus.index:
        if str(i) == str(node_label).strip():
            return i
    return None

def ensure_electrolyzer_load(net: pp.pandapowerNet, bus_idx: int, name="Electrolyzer"):
    existing = net.load[net.load['name'] == name]
    if len(existing) > 0:
        return existing.index[0]
    return pp.create_load(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=name)

def set_electrolyzer_power(net: pp.pandapowerNet, name: str, p_mw: float, q_mvar: float):
    idx = net.load[net.load['name'] == name].index
    if len(idx) == 0:
        raise ValueError(f"Electrolyzer '{name}' not found")
    i = idx[0]
    net.load.at[i, 'p_mw'] = float(p_mw)
    net.load.at[i, 'q_mvar'] = float(q_mvar)

def assign_base_loads(net, active_df, reactive_df, node_to_bus, col, electrolyzer_name="Electrolyzer"):
    bus_aggregate = {}
    for node in active_df.index:
        bus_idx = node_to_bus.get(str(node))
        if bus_idx is None:
            continue
        p_kw = float(active_df.at[node, col])
        p_mw = p_kw / 1.0
        q_mvar = 0.0
        if reactive_df is not None and node in reactive_df.index:
            q_kw = float(reactive_df.at[node, col])
            q_mvar = q_kw / 1.0
        prev = bus_aggregate.get(bus_idx, (0.0, 0.0))
        bus_aggregate[bus_idx] = (prev[0] + p_mw, prev[1] + q_mvar)
    for bus_idx, (p_total, q_total) in bus_aggregate.items():
        loads_at_bus = net.load[(net.load.bus == bus_idx) & (net.load.name != electrolyzer_name)]
        if len(loads_at_bus) > 0:
            total_existing = loads_at_bus.p_mw.sum()
            for li in loads_at_bus.index:
                share = net.load.at[li, 'p_mw'] / total_existing if total_existing > 0 else 1.0 / len(loads_at_bus)
                net.load.at[li, 'p_mw'] = p_total * share
                net.load.at[li, 'q_mvar'] = q_total * share
        else:
            pp.create_load(net, bus=bus_idx, p_mw=p_total, q_mvar=q_total, name=f"ts_load_bus_{bus_idx}")

def assign_generation_profiles(net, gen_df, col):
    if gen_df is None:
        return
    for gen_label in gen_df.index:
        val_kw = float(gen_df.at[gen_label, col])
        p_mw = val_kw / 1.0
        mapped_bus = map_node_to_bus(gen_label, net)
        if mapped_bus is None:
            mapped_bus = int(net.bus.index[0])
        sgens_at_bus = net.sgen[net.sgen.bus == mapped_bus]
        if len(sgens_at_bus) > 0:
            for si in sgens_at_bus.index:
                net.sgen.at[si, 'p_mw'] = p_mw / len(sgens_at_bus)
        else:
            pp.create_sgen(net, bus=mapped_bus, p_mw=p_mw, q_mvar=0.0, name=f"ts_sgen_bus_{mapped_bus}")

# -------------------------
# === Time-Series Simulation ===
# -------------------------
def run_pf_scenario(active_df, reactive_df, gen_df, net, electrolyzer=False, p_nom=1.0, pf=0.95):
    ELEC_NAME = "Electrolyzer_controlled"
    if electrolyzer:
        ensure_electrolyzer_load(net, ELECTROLYZER_BUS, name=ELEC_NAME)
    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    hours = list(range(DURATION_HOURS))
    bus_v = pd.DataFrame(index=hours, columns=[str(b) for b in net.bus.index], dtype=float)
    line_l = pd.DataFrame(index=hours, columns=[str(l) for l in net.line.index], dtype=float)

    if electrolyzer:
        elec_status = pd.Series(index=hours, dtype=object)
        elec_p_set = pd.Series(index=hours, dtype=float)
        elec_q_set = pd.Series(index=hours, dtype=float)
    else:
        elec_status = elec_p_set = elec_q_set = None

    q_nom = p_nom * np.tan(np.arccos(pf))

    for t in hours:
        col = active_df.columns[START_HOUR + t]
        assign_base_loads(net, active_df, reactive_df, node_to_bus, col, electrolyzer_name=ELEC_NAME)
        assign_generation_profiles(net, gen_df, col)

        if electrolyzer:
            pp.runpp(net)
            max_line = net.res_line.loading_percent.max()
            v_bus = net.res_bus.vm_pu.at[ELECTROLYZER_BUS]

            # === Active Power Control ===
            if (max_line < LINE_LOADING_THRESHOLD) and (v_bus > VOLTAGE_THRESHOLD_PU):
                p_set = p_nom
                status = "ON"
            else:
                p_set = 0.0
                status = "OFF"

            # === Reactive Power Control (Local AVR) ===
            q_prev = net.load.loc[net.load.name == ELEC_NAME, 'q_mvar'].iloc[0]
            if status == "ON":
                if v_bus < V_OVER:
                    q_delta = min(0.5, (1.0 - v_bus) * DQ_OVER)
                elif v_bus > V_UNDER:
                    q_delta = max(-0.5, (1.0 - v_bus) * DQ_UNDER)
                else:
                    q_delta = 0.0
                q_value = np.clip(q_prev + q_delta, -Q_MAX, Q_MAX)
            else:
                q_value = 0.0

            set_electrolyzer_power(net, ELEC_NAME, p_set, q_value)
            elec_status[t] = status
            elec_p_set[t] = p_set
            elec_q_set[t] = q_value

        pp.runpp(net)
        bus_v.loc[t, :] = net.res_bus.vm_pu.values
        line_l.loc[t, :] = net.res_line.loading_percent.values

    return bus_v, line_l, elec_status, elec_p_set, elec_q_set

# -------------------------
# === MAIN ===
# -------------------------
if __name__ == "__main__":
    net_no_elec = pn.create_cigre_network_mv()
    net_ctrl_elec = pn.create_cigre_network_mv()

    if ELECTROLYZER_BUS is None:
        ELECTROLYZER_BUS = int(input("Enter electrolyzer bus index: "))

    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH)
    gen_df = read_timeseries_csv(GEN_PATH)

    # === Scenario 1: Without electrolyzer ===
    bus_v_no, line_l_no, _, _, _ = run_pf_scenario(active_df, reactive_df, gen_df, net_no_elec, electrolyzer=False)

    # === Scenario 2: With controlled electrolyzer (P + Q control) ===
    bus_v_ctrl, line_l_ctrl, elec_status, elec_p_set, elec_q_set = run_pf_scenario(
        active_df, reactive_df, gen_df, net_ctrl_elec,
        electrolyzer=True, p_nom=ELECTROLYZER_P_NOMINAL_MW, pf=ELECTROLYZER_PF
    )

    # === Plot Voltage Comparison ===
    plt.figure(figsize=(12,6))
    plt.plot(bus_v_no[str(ELECTROLYZER_BUS)], '--', label=f"Bus {ELECTROLYZER_BUS} - Without Electrolyzer")
    plt.plot(bus_v_ctrl[str(ELECTROLYZER_BUS)], '-', label=f"Bus {ELECTROLYZER_BUS} - With P&Q Control")
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage Comparison (With and Without Electrolyzer)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot Electrolyzer Q control ===
    plt.figure(figsize=(12,4))
    plt.plot(elec_q_set, label="Electrolyzer Q (Mvar)")
    plt.xlabel("Hour")
    plt.ylabel("Q (Mvar)")
    plt.title("Reactive Power Control of Electrolyzer (AVR Behavior)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # === Plot Combined Voltage and Reactive Power ===
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: Voltage
    color1 = 'tab:blue'
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Voltage (p.u.)", color=color1)
    ax1.plot(bus_v_ctrl[str(ELECTROLYZER_BUS)], linestyle='--', color=color1, linewidth=1, label="Voltage (p.u.)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    # Right axis: Reactive Power
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Reactive Power Q (Mvar)", color=color2)
    ax2.plot(elec_q_set, linestyle='-', color=color2, linewidth=2, label="Reactive Power (Mvar)")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and layout
    plt.title(f"Voltage and Reactive Power at Bus {ELECTROLYZER_BUS}")
    fig.tight_layout()

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.show()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn

'''
This Python script performs a time-series power flow simulation on a medium-voltage test network (Cigre MV). It supports two scenarios:
Without an electrolyzer – the network runs normally with all loads and generation.
With a controlled electrolyzer – the electrolyzer load can turn ON or OFF based on line loading and bus voltage. Its active and reactive power are applied only if certain conditions are met.
'''

# -------------------------
# === USER CONFIGURATION ===
# -------------------------
# Paths to your CSVs (put your real paths here):
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

START_HOUR = 0
DURATION_HOURS = 72
OUT_DIR = "./timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer parameters
ELECTROLYZER_BUS = None  # Will be entered by user
ELECTROLYZER_P_NOMINAL_MW = 1.0
ELECTROLYZER_PF = 0.95
LINE_LOADING_THRESHOLD = 100.0
VOLTAGE_THRESHOLD_PU = 0.98

# Highlighted buses for plotting
HIGHLIGHT_BUSES = []

# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

def map_node_to_bus(node_label, net: pp.pandapowerNet):
    # Try numeric mapping first
    try:
        idx = int(node_label)
        if idx in net.bus.index:
            return idx
    except:
        pass
    # Try name column
    if 'name' in net.bus.columns:
        for i, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return i
    # Try as string index
    for i in net.bus.index:
        if str(i) == str(node_label).strip():
            return i
    return None

def ensure_electrolyzer_load(net: pp.pandapowerNet, bus_idx: int, name: str = "Electrolyzer"):
    existing = net.load[net.load['name'] == name]
    if len(existing) > 0:
        return existing.index[0]
    return pp.create_load(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=name)

def set_electrolyzer_power(net: pp.pandapowerNet, name: str, p_mw: float, q_mvar: float):
    idx = net.load[net.load['name'] == name].index
    if len(idx) == 0:
        raise ValueError(f"Electrolyzer load named '{name}' not found")
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
            if total_existing == 0:
                for li in loads_at_bus.index:
                    net.load.at[li, 'p_mw'] = p_total / len(loads_at_bus)
                    net.load.at[li, 'q_mvar'] = q_total / len(loads_at_bus)
            else:
                for li in loads_at_bus.index:
                    share = net.load.at[li, 'p_mw'] / total_existing if total_existing > 0 else 1.0/len(loads_at_bus)
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
# === Run scenario ===
# -------------------------
def run_pf_scenario(active_df, reactive_df, gen_df, net, electrolyzer=False, p_nom=1.0, pf=0.95):
    ELEC_NAME = "Electrolyzer_controlled"
    if electrolyzer:
        ensure_electrolyzer_load(net, ELECTROLYZER_BUS, name=ELEC_NAME)
    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    hours = list(range(DURATION_HOURS))
    bus_v = pd.DataFrame(index=hours, columns=[str(b) for b in net.bus.index], dtype=float)
    line_l = pd.DataFrame(index=hours, columns=[str(l) for l in net.line.index], dtype=float)
    elec_status = pd.Series(index=hours, dtype=object) if electrolyzer else None
    elec_p_set = pd.Series(index=hours, dtype=float) if electrolyzer else None

    q_nom = p_nom * np.tan(np.arccos(pf))

    for t in hours:
        col = active_df.columns[START_HOUR + t]
        assign_base_loads(net, active_df, reactive_df, node_to_bus, col, electrolyzer_name=ELEC_NAME)
        assign_generation_profiles(net, gen_df, col)

        if electrolyzer:
            try:
                pp.runpp(net, calculate_voltage_angles=True)
                max_line = float(net.res_line.loading_percent.max())
                v_bus = float(net.res_bus.vm_pu.at[ELECTROLYZER_BUS])
                if (max_line < LINE_LOADING_THRESHOLD) and (v_bus > VOLTAGE_THRESHOLD_PU):
                    set_electrolyzer_power(net, ELEC_NAME, p_nom, q_nom)
                    elec_status[t] = "ON"
                    elec_p_set[t] = p_nom
                else:
                    set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)
                    elec_status[t] = "OFF"
                    elec_p_set[t] = 0.0
            except:
                set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)
                elec_status[t] = "OFF"
                elec_p_set[t] = 0.0

        try:
            pp.runpp(net, calculate_voltage_angles=True)
            bus_v.loc[t, :] = net.res_bus.vm_pu.values
            line_l.loc[t, :] = net.res_line.loading_percent.values
        except:
            bus_v.loc[t, :] = np.nan
            line_l.loc[t, :] = np.nan

    return bus_v, line_l, elec_status, elec_p_set

# -------------------------
# === Main execution ===
# -------------------------
if __name__ == "__main__":
    net_no_elec = pn.create_cigre_network_mv()
    net_ctrl_elec = pn.create_cigre_network_mv()

    if ELECTROLYZER_BUS is None:
        ELECTROLYZER_BUS = int(input("Enter electrolyzer bus index: "))
    HIGHLIGHT_BUSES.append(ELECTROLYZER_BUS)

    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH) if os.path.exists(REACTIVE_PATH) else None
    gen_df = read_timeseries_csv(GEN_PATH) if os.path.exists(GEN_PATH) else None

    # Scenario 1: Without electrolyzer
    bus_v_no, line_l_no, _, _ = run_pf_scenario(active_df, reactive_df, gen_df, net_no_elec, electrolyzer=False)

    # Scenario 2: With controlled electrolyzer
    bus_v_ctrl, line_l_ctrl, elec_status, elec_p_set = run_pf_scenario(active_df, reactive_df, gen_df, net_ctrl_elec,
                                                                      electrolyzer=True,
                                                                      p_nom=ELECTROLYZER_P_NOMINAL_MW,
                                                                      pf=ELECTROLYZER_PF)

    # === Plot voltage comparison ===
    plt.figure(figsize=(12,6))
    for b in HIGHLIGHT_BUSES:
        plt.plot(bus_v_no[str(b)], '--', label=f"Bus {b} Ohne Elektrolyzer")
        plt.plot(bus_v_ctrl[str(b)], '-', label=f"Bus {b} Mit Controller")
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage Comparison: Ohne Elektrolyzer vs Mit Controller")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"voltage_comparison.png"))
    plt.show()

    # === Plot average line loading ===
    plt.figure(figsize=(12,5))
    plt.plot(line_l_no.mean(axis=1), '--', label="Avg Line Loading Ohne Elektrolyzer")
    plt.plot(line_l_ctrl.mean(axis=1), '-', label="Avg Line Loading Mit Controller")
    plt.xlabel("Hour")
    plt.ylabel("Loading (%)")
    plt.title("Average Line Loading Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"line_loading_comparison.png"))
    plt.show()

    # === Electrolyzer status ===
    if elec_status is not None:
        plt.figure(figsize=(12,3))
        status_numeric = elec_status.replace({'ON':1, 'OFF':0}).astype(float)
        plt.step(status_numeric.index, status_numeric, where='mid', label='Electrolyzer ON(1)/OFF(0)')
        plt.plot(status_numeric.index, elec_p_set, color='orange', label='Electrolyzer P_set (MW)')
        plt.xlabel("Hour")
        plt.title("Electrolyzer Status and P setpoint")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,"electrolyzer_status.png"))
        plt.show()

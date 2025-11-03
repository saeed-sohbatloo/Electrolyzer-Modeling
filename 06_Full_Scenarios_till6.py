import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
# -------------------------
# USER CONFIGURATION
# USER CONFIGURATION
# -------------------------
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"
GEN_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"


START_HOUR = 0
DURATION_HOURS = 120
ELECTROLYZER_P_NOMINAL_MW = 1.0
ELECTROLYZER_PF = 0.95
LINE_LOADING_THRESHOLD = 100.0
VOLTAGE_THRESHOLD_PU = 0.98

# Reactive power (Q) control
Q_MAX = 0.5
DQ_OVER = 0.506
DQ_UNDER = .404
V_DB = 0.01
V_OVER = 1.0 - V_DB
V_UNDER = 1.0 + V_DB

# -------------------------
# DEFINE SCENARIOS
# -------------------------
SCENARIOS = [
    {'name': 'No Electrolyzer', 'has_electrolyzer': False, 'onoff_control': False, 'q_control': False},
    {'name': 'Electrolyzer ', 'has_electrolyzer': True, 'onoff_control': False, 'q_control': False},
    {'name': 'Electrolyzer with Q-Controller', 'has_electrolyzer': True, 'onoff_control': False, 'q_control': True}
]

# -------------------------
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
# RUN TIMESERIES FOR ONE SCENARIO
# -------------------------
def run_timeseries(active_df, reactive_df, gen_df, net, scenario, electrolyzer_bus):
    """
    Run time-series simulation for a single scenario.

    scenario: dict with keys:
        - 'name': descriptive name
        - 'has_electrolyzer': True/False
        - 'onoff_control': True/False
        - 'q_control': True/False
    """
    # Extract scenario options
    has_elec = scenario.get('has_electrolyzer', False)
    onoff_ctrl = scenario.get('onoff_control', False)
    q_ctrl = scenario.get('q_control', False)

    ELEC_NAME = "Electrolyzer"

    # Create electrolyzer load if present
    if has_elec:
        ensure_electrolyzer_load(net, electrolyzer_bus, name=ELEC_NAME)

    # Map CSV nodes to bus indices (keys as strings)
    node_to_bus = {str(n): map_node_to_bus(n, net) for n in active_df.index}

    hours = list(range(DURATION_HOURS))

    bus_v = pd.DataFrame(index=hours, columns=[str(b) for b in net.bus.index], dtype=float)
    line_l = pd.DataFrame(index=hours, columns=[str(l) for l in net.line.index], dtype=float)

    if has_elec:
        elec_status = pd.Series(index=hours, dtype=object)
        elec_p_set = pd.Series(index=hours, dtype=float)
        elec_q_set = pd.Series(index=hours, dtype=float)
    else:
        elec_status = pd.Series(index=hours, dtype=object).fillna("OFF")
        elec_p_set = pd.Series(index=hours, dtype=float).fillna(0.0)
        elec_q_set = pd.Series(index=hours, dtype=float).fillna(0.0)

    # Initial P and Q
    prev_p_set = ELECTROLYZER_P_NOMINAL_MW if has_elec and not onoff_ctrl else 0.0
    prev_q_set = 0.0

    for t in hours:
        t_col = active_df.columns[START_HOUR + t]

        # Assign base loads
        assign_base_loads(net, active_df, reactive_df, node_to_bus, t_col, electrolyzer_name=ELEC_NAME if has_elec else "")

        # Assign generation
        assign_generation_profiles(net, gen_df, t_col)

        # Apply previous P/Q
        if has_elec:
            set_electrolyzer_power(net, ELEC_NAME, prev_p_set, prev_q_set)

        # Run initial PF
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            bus_v.loc[t, :] = np.nan
            line_l.loc[t, :] = np.nan
            if has_elec:
                elec_status[t] = "OFF"
                elec_p_set[t] = 0.0
                elec_q_set[t] = 0.0
            continue

        bus_v.loc[t, :] = net.res_bus.vm_pu.values
        line_l.loc[t, :] = net.res_line.loading_percent.values

        # Determine voltage at electrolyzer bus
        v_bus = net.res_bus.vm_pu.at[electrolyzer_bus] if has_elec else None

        # Reactive power control
        if has_elec and q_ctrl and prev_p_set > 0.0:
            if v_bus < V_OVER:
                q_delta = min(Q_MAX, (1.0 - v_bus) * DQ_OVER)
            elif v_bus > V_UNDER:
                q_delta = max(-Q_MAX, (1.0 - v_bus) * DQ_UNDER)
            else:
                q_delta = 0.0
            q_value = np.clip(prev_q_set + q_delta, -Q_MAX, Q_MAX)
            set_electrolyzer_power(net, ELEC_NAME, prev_p_set, q_value)
        elif has_elec:
            q_value = prev_q_set if prev_p_set > 0 else 0.0

        # Re-run PF to include Q effect
        if has_elec and q_ctrl:
            try:
                pp.runpp(net)
                bus_v.loc[t, :] = net.res_bus.vm_pu.values
                line_l.loc[t, :] = net.res_line.loading_percent.values
            except pp.LoadflowNotConverged:
                bus_v.loc[t, :] = np.nan
                line_l.loc[t, :] = np.nan

        # ON/OFF decision for next hour
        if has_elec and onoff_ctrl:
            max_line = net.res_line.loading_percent.max()
            if (max_line < LINE_LOADING_THRESHOLD) and (v_bus > VOLTAGE_THRESHOLD_PU):
                next_p_set = ELECTROLYZER_P_NOMINAL_MW
                next_status = "ON"
            else:
                next_p_set = 0.0
                next_status = "OFF"
        else:
            next_p_set = prev_p_set
            next_status = "ON" if prev_p_set > 0 else "OFF"

        # Record values
        if has_elec:
            elec_status[t] = "ON" if prev_p_set > 0 else "OFF"
            elec_p_set[t] = prev_p_set
            elec_q_set[t] = q_value

        prev_p_set = next_p_set
        prev_q_set = q_value if next_p_set > 0 else 0.0

    return bus_v, line_l, elec_status, elec_p_set, elec_q_set

# -------------------------
# MAIN LOOP
# -------------------------
if __name__ == "__main__":
    electrolyzer_bus = int(input("Enter electrolyzer bus index: "))
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH)
    gen_df = read_timeseries_csv(GEN_PATH)

    net_template = pn.create_cigre_network_mv(with_der=False)

    results = {}
    for sc in SCENARIOS:
        print(f"Running scenario: {sc['name']}")
        net_copy = copy.deepcopy(net_template)
        bus_v, line_l, elec_status, elec_p, elec_q = run_timeseries(active_df, reactive_df, gen_df, net_copy, sc, electrolyzer_bus)
        results[sc['name']] = {'bus_v': bus_v, 'line_l': line_l, 'elec_status': elec_status, 'elec_p': elec_p, 'elec_q': elec_q}

    # Example: plot voltage for all scenarios
    plt.figure(figsize=(12,5))
    for sc in SCENARIOS:
        plt.plot(results[sc['name']]['bus_v'][str(electrolyzer_bus)], label=sc['name'])
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage at Electrolyzer Bus")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =======================
    # AFTER RUNNING ALL SCENARIOS
    # =======================
    hours = range(DURATION_HOURS)

    # Create dictionaries for plotting
    bus_v_dict = {}
    elec_q_dict = {}
    elec_status_dict = {}
    has_elec_dict = {}
    onoff_ctrl_dict = {}

    for sc in SCENARIOS:
        name = sc['name']
        bus_v_dict[name] = results[name]['bus_v']
        elec_q_dict[name] = results[name]['elec_q'] if sc['has_electrolyzer'] else None
        elec_status_dict[name] = results[name]['elec_status'] if sc['has_electrolyzer'] else None
        has_elec_dict[name] = sc['has_electrolyzer']
        onoff_ctrl_dict[name] = sc['onoff_control']

    # --- Plot voltage at electrolyzer bus for all scenarios ---
    plt.figure(figsize=(12, 5))
    for sc_name, bus_v in bus_v_dict.items():
        if has_elec_dict[sc_name]:
            plt.plot(hours, bus_v[str(electrolyzer_bus)], label=sc_name)
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage at Electrolyzer Bus")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot Electrolyzer Q control for all scenarios with electrolyzer ---
    plt.figure(figsize=(12, 4))
    for sc_name, elec_q in elec_q_dict.items():
        if elec_q is not None:
            plt.plot(hours, elec_q, label=f"{sc_name} - Q (Mvar)")
    plt.xlabel("Hour")
    plt.ylabel("Reactive Power Q (Mvar)")
    plt.title("Electrolyzer Reactive Power Control (AVR Behavior)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot Combined Voltage and Reactive Power for a selected scenario ---
    # --- Plot: Combined Voltage and Reactive Power at electrolyzer bus (All Scenarios) ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: Voltage (take one reference, e.g., "Always ON" scenario)
    color1 = 'tab:blue'
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Voltage (p.u.)", color=color1)

    # Select a scenario to show voltage
    bus_v_ref = None
    for sc in SCENARIOS:
        if has_elec_dict[sc['name']] and sc['onoff_control']:
            bus_v_ref = bus_v_dict[sc['name']]
            break
    if bus_v_ref is None:
        bus_v_ref = bus_v_dict["Electrolyzer with Q-Controller"]

    ax1.plot(hours, bus_v_ref[str(electrolyzer_bus)], linestyle='--', color=color1, linewidth=1.5,
             label='Voltage (Reference)')

    # Reference voltage limits (dotted faint)
    ax1.axhline(V_OVER, color=color1, linestyle=':', linewidth=1, alpha=0.5, label=f"V_OVER={V_OVER:.3f}")
    ax1.axhline(V_UNDER, color=color1, linestyle=':', linewidth=1, alpha=0.5, label=f"V_UNDER={V_UNDER:.3f}")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Right axis: Reactive Power Q
    ax2 = ax1.twinx()
    ax2.set_ylabel("Reactive Power Q (Mvar)", color='black')

    # Define colors and linestyles for each scenario
    q_colors = ['tab:red', 'tab:orange', 'tab:green']
    q_linestyles = ['.', '-', '-']

    for i, sc in enumerate(SCENARIOS):
        name = sc['name']
        if not has_elec_dict[name]:
            continue
        elec_q_series = elec_q_dict[name]
        ax2.plot(hours, elec_q_series, linestyle=q_linestyles[i], color=q_colors[i], linewidth=2,
                 label=f"{name} - Q (Mvar)")

    # Reference line Q=0
    ax2.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f"Voltage and Reactive Power at Bus {electrolyzer_bus} (All Scenarios)")
    fig.tight_layout()
    plt.show()

    # --- Plot Electrolyzer ON/OFF status for scenarios with ON/OFF control ---
    plt.figure(figsize=(12, 3))
    for sc_name, elec_status in elec_status_dict.items():
        if onoff_ctrl_dict[sc_name]:
            plt.step(hours, elec_status.replace({'ON': 1, 'OFF': 0}).astype(int), where='post', label=sc_name)
    plt.xlabel("Hour")
    plt.ylabel("ON=1 / OFF=0")
    plt.title("Electrolyzer ON/OFF Status")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

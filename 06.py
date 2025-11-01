"""
Time-series power flow for CIGRE MV network with electrolyzer ON/OFF + local Q control.

Control algorithm (summary):
- Start with electrolyzer OFF at hour 0 (P=0, Q=0).
- For each hour t:
    1) Apply P setpoint decided at end of previous hour (this determines whether electrolyzer is ON in hour t).
    2) Use the previous Q (or zero) initially and run power flow to get voltage estimate.
    3) If electrolyzer is ON (P>0), compute local Q adjustment based on measured voltage (AVR-like) and physical Q limits,
       set that Q and re-run power flow so the final hour-t results include Q effect.
       If electrolyzer is OFF, Q=0.
    4) Record bus voltages, line loading and transformer loading for hour t (after Q applied).
    5) Based on these results (hour t), decide whether electrolyzer should be ON or OFF for hour t+1
       using thresholds on max line loading and the electrolyzer bus voltage.
- Repeat until end of horizon.

Outputs:
- Time series of bus voltages, line loadings, transformer loadings.
- Time series of electrolyzer P and Q setpoints and ON/OFF status.
- Plots comparing three scenarios:
    a) No electrolyzer
    b) Electrolyzer always ON (P fixed, Q controlled)
    c) Electrolyzer controlled ON/OFF (P decided for next hour) + Q controlled when ON
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
import copy

# -------------------------
# USER CONFIGURATION
# -------------------------
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"
GEN_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"

START_HOUR = 0
DURATION_HOURS = 120
OUT_DIR = "./timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer parameters
ELECTROLYZER_BUS = None  # asked interactively
ELECTROLYZER_P_NOMINAL_MW = 1.0
ELECTROLYZER_PF = 0.95
LINE_LOADING_THRESHOLD = 100.0   # [%]
VOLTAGE_THRESHOLD_PU = 0.98     # [p.u.]

# Reactive power control (AVR-like)
Q_MAX = 0.5      # Mvar (absolute max reactive capability)
DQ_OVER = 0.06   # gain factor for under-voltage -> inject Q
DQ_UNDER = 0.04  # gain factor for over-voltage -> absorb Q
V_DB = 0.01
V_OVER = 1.0 - V_DB   # lower bound
V_UNDER = 1.0 + V_DB  # upper bound

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    """Read a CSV timeseries where index = node labels and columns = time steps."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Timeseries file not found: {path}")
    return pd.read_csv(path, index_col=0)

def map_node_to_bus(node_label, net: pp.pandapowerNet):
    """Map CSV node label to pandapower bus index (tries integer index, bus name, stringified index)."""
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

def ensure_electrolyzer_load(net: pp.pandapowerNet, bus_idx: int, name: str = "Electrolyzer_controlled"):
    """Ensure a load element for the electrolyzer exists (initially p=0,q=0). Return load index."""
    existing = net.load[net.load['name'] == name]
    if len(existing) > 0:
        return existing.index[0]
    return pp.create_load(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=name)

def set_electrolyzer_power(net: pp.pandapowerNet, name: str, p_mw: float, q_mvar: float):
    """Set p and q for electrolyzer load identified by name."""
    idx = net.load[net.load['name'] == name].index
    if len(idx) == 0:
        raise ValueError(f"Electrolyzer load '{name}' not found in net.load")
    i = idx[0]
    net.load.at[i, 'p_mw'] = float(p_mw)
    net.load.at[i, 'q_mvar'] = float(q_mvar)

def assign_base_loads(net: pp.pandapowerNet,
                      active_df: pd.DataFrame,
                      reactive_df: pd.DataFrame,
                      node_to_bus_map: dict,
                      t_col: str,
                      electrolyzer_name: str = "Electrolyzer_controlled"):
    """
    Assign base loads (from CSV) to existing loads on each bus for a single timestep column t_col.
    The electrolyzer load (by name) is not overwritten here.
    """
    bus_aggregate = {}
    for node in active_df.index:
        bus_idx = node_to_bus_map.get(str(node))
        if bus_idx is None:
            continue
        p_val = float(active_df.at[node, t_col])
        p_mw = p_val / 1.0  # adjust if CSV units differ
        q_mvar = 0.0
        if reactive_df is not None and node in reactive_df.index:
            q_val = float(reactive_df.at[node, t_col])
            q_mvar = q_val / 1.0
        prev = bus_aggregate.get(bus_idx, (0.0, 0.0))
        bus_aggregate[bus_idx] = (prev[0] + p_mw, prev[1] + q_mvar)

    for bus_idx, (p_total, q_total) in bus_aggregate.items():
        loads_at_bus = net.load[(net.load.bus == bus_idx) & (net.load.name != electrolyzer_name)]
        if len(loads_at_bus) > 0:
            total_existing = loads_at_bus.p_mw.sum()
            if total_existing == 0:
                # distribute evenly
                for li in loads_at_bus.index:
                    net.load.at[li, 'p_mw'] = p_total / len(loads_at_bus)
                    net.load.at[li, 'q_mvar'] = q_total / len(loads_at_bus)
            else:
                # preserve proportional shares
                for li in loads_at_bus.index:
                    share = net.load.at[li, 'p_mw'] / total_existing if total_existing > 0 else 1.0/len(loads_at_bus)
                    net.load.at[li, 'p_mw'] = p_total * share
                    net.load.at[li, 'q_mvar'] = q_total * share
        else:
            pp.create_load(net, bus=bus_idx, p_mw=p_total, q_mvar=q_total, name=f"ts_load_bus_{bus_idx}")

def assign_generation_profiles(net: pp.pandapowerNet, gen_df: pd.DataFrame, t_col: str):
    """Assign generation profiles (from CSV) to sgens; fallback to first bus if mapping fails."""
    if gen_df is None:
        return
    for gen_label in gen_df.index:
        val = float(gen_df.at[gen_label, t_col])
        p_mw = val / 1.0
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
# TIMESERIES FUNCTION (mode-aware)
# -------------------------
def run_timeseries_control(active_df, reactive_df, gen_df, net, p_nom, pf, mode='controlled'):
    """
    Run time-series for one network copy.
    mode: 'none' | 'always' | 'controlled'
      - 'none'     : electrolyzer not present (P=0,Q=0)
      - 'always'   : electrolyzer always ON (P=p_nom each hour), Q still controlled locally
      - 'controlled': electrolyzer ON/OFF controlled (decision for next hour), Q controlled locally when ON

    Returns:
      bus_v_df, line_l_df, trafo_l_df, elec_status_series, elec_p_series, elec_q_series
    """
    ELEC_NAME = "Electrolyzer_controlled"
    # create electrolyzer load if mode != 'none'
    if mode != 'none':
        ensure_electrolyzer_load(net, ELECTROLYZER_BUS, name=ELEC_NAME)
    # pre-map nodes->buses
    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    hours = list(range(DURATION_HOURS))
    bus_v = pd.DataFrame(index=hours, columns=[str(b) for b in net.bus.index], dtype=float)
    line_l = pd.DataFrame(index=hours, columns=[str(l) for l in net.line.index], dtype=float)
    trafo_l = pd.DataFrame(index=hours, columns=[str(t) for t in net.trafo.index], dtype=float) if len(net.trafo) > 0 else None

    elec_status = pd.Series(index=hours, dtype=object) if mode != 'none' else pd.Series(index=hours, dtype=object).fillna("OFF")
    elec_p_set = pd.Series(index=hours, dtype=float) if mode != 'none' else pd.Series(index=hours, dtype=float).fillna(0.0)
    elec_q_set = pd.Series(index=hours, dtype=float) if mode != 'none' else pd.Series(index=hours, dtype=float).fillna(0.0)

    # initial P,Q applied at hour 0 (previous decision). Start with OFF.
    prev_p_set = 0.0
    prev_q_set = 0.0

    # If mode=='always', make prev_p_set = p_nom initially and keep it always
    if mode == 'always':
        prev_p_set = p_nom

    for t in hours:
        t_col = active_df.columns[START_HOUR + t]

        # assign base loads and generation for this hour
        assign_base_loads(net, active_df, reactive_df, node_to_bus, t_col, electrolyzer_name=ELEC_NAME)
        assign_generation_profiles(net, gen_df, t_col)

        # Apply previous decision's P and previous Q (so PF baseline includes prev settings)
        if mode == 'none':
            # ensure electrolyzer absent/zero
            pass
        else:
            set_electrolyzer_power(net, ELEC_NAME, prev_p_set, prev_q_set)

        # initial PF to get estimated voltages (before Q adjustment)
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except pp.LoadflowNotConverged:
            # store NaNs and continue, but ensure next decision defaults to OFF
            bus_v.loc[t, :] = np.nan
            line_l.loc[t, :] = np.nan
            if trafo_l is not None:
                trafo_l.loc[t, :] = np.nan
            # next decision default OFF if controlled
            if mode == 'controlled':
                next_p_set = 0.0
                next_status = "OFF"
            else:
                next_p_set = prev_p_set  # keep same for 'always' / 'none'
                next_status = "ON" if prev_p_set > 0 else "OFF"
            prev_p_set = next_p_set
            prev_q_set = 0.0
            elec_status[t] = "ON" if prev_p_set > 0 else "OFF"
            elec_p_set[t] = prev_p_set
            elec_q_set[t] = prev_q_set
            continue

        # measure local voltage after initial PF
        if mode == 'none':
            v_bus_initial = None
        else:
            v_bus_initial = float(net.res_bus.vm_pu.at[ELECTROLYZER_BUS])

        # --- Q control (AVR-like) applied within same hour if electrolyzer is actually ON this hour ---
        # Determine whether electrolyzer is ON in this hour (applied P from prev decision or always mode)
        applied_p = prev_p_set if mode != 'none' else 0.0
        if mode != 'none' and applied_p > 0.0:
            # compute Q_delta based on local voltage (use v_bus_initial)
            q_prev = float(net.load.loc[net.load.name == ELEC_NAME, 'q_mvar'].iloc[0])
            if v_bus_initial is None:
                q_delta = 0.0
            else:
                if v_bus_initial < V_OVER:
                    # under-voltage -> inject reactive power (positive)
                    q_delta = (1.0 - v_bus_initial) * DQ_OVER
                elif v_bus_initial > V_UNDER:
                    # over-voltage -> absorb reactive power (negative)
                    q_delta = (1.0 - v_bus_initial) * DQ_UNDER
                else:
                    q_delta = 0.0
            # Clip delta and accumulate
            q_cand = q_prev + q_delta
            q_value = float(np.clip(q_cand, -Q_MAX, Q_MAX))
        else:
            q_value = 0.0

        # Apply updated Q (and same P) and re-run PF so current-hour results include the Q effect
        if mode != 'none':
            set_electrolyzer_power(net, ELEC_NAME, applied_p, q_value)

        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except pp.LoadflowNotConverged:
            # if second run fails, record NaNs and set next decision to OFF
            bus_v.loc[t, :] = np.nan
            line_l.loc[t, :] = np.nan
            if trafo_l is not None:
                trafo_l.loc[t, :] = np.nan
            if mode == 'controlled':
                next_p_set = 0.0
                next_status = "OFF"
            else:
                next_p_set = prev_p_set
                next_status = "ON" if prev_p_set > 0 else "OFF"
            prev_p_set = next_p_set
            prev_q_set = 0.0
            elec_status[t] = "ON" if applied_p > 0 else "OFF"
            elec_p_set[t] = applied_p
            elec_q_set[t] = 0.0
            continue

        # Save final "after-Q" results for this hour
        bus_v.loc[t, :] = net.res_bus.vm_pu.values
        line_l.loc[t, :] = net.res_line.loading_percent.values
        if trafo_l is not None:
            trafo_l.loc[t, :] = net.res_trafo.loading_percent.values

        # Decide next hour P (ON/OFF) based on results of this hour (after Q applied)
        if mode == 'controlled':
            max_line = float(net.res_line.loading_percent.max())
            v_bus_after = float(net.res_bus.vm_pu.at[ELECTROLYZER_BUS])
            if (max_line < LINE_LOADING_THRESHOLD) and (v_bus_after > VOLTAGE_THRESHOLD_PU):
                next_p_set = p_nom
                next_status = "ON"
            else:
                next_p_set = 0.0
                next_status = "OFF"
        elif mode == 'always':
            next_p_set = p_nom
            next_status = "ON"
        else:  # mode == 'none'
            next_p_set = 0.0
            next_status = "OFF"

        # Prepare for next hour: prev_p_set becomes next_p_set; prev_q_set for next hour:
        # If next hour will be OFF -> prev_q_set = 0; if ON, we keep current q_value as starting point
        prev_p_set = next_p_set
        prev_q_set = q_value if next_p_set > 0 else 0.0

        # Record what was applied during hour t (applied_p and q_value)
        elec_status[t] = "ON" if applied_p > 0.0 else "OFF"
        elec_p_set[t] = applied_p
        elec_q_set[t] = q_value

    return bus_v, line_l, trafo_l, elec_status, elec_p_set, elec_q_set

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Ask user for electrolyzer bus if not configured
    if ELECTROLYZER_BUS is None:
        ELECTROLYZER_BUS = int(input("Enter electrolyzer bus index: "))

    # Create three independent copies of the network for scenario isolation
    net_template = pn.create_cigre_network_mv(with_der=False)
    net_no = copy.deepcopy(net_template)
    net_always = copy.deepcopy(net_template)
    net_ctrl = copy.deepcopy(net_template)

    # Read timeseries
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH) if os.path.exists(REACTIVE_PATH) else None
    gen_df = read_timeseries_csv(GEN_PATH) if os.path.exists(GEN_PATH) else None

    # Run three scenarios
    bus_v_no, line_l_no, trafo_l_no, _, _, _ = run_timeseries_control(active_df, reactive_df, gen_df, net_no,
                                                                      p_nom=0.0, pf=ELECTROLYZER_PF, mode='none')

    bus_v_always, line_l_always, trafo_l_always, elec_status_always, elec_p_always, elec_q_always = run_timeseries_control(
        active_df, reactive_df, gen_df, net_always, p_nom=ELECTROLYZER_P_NOMINAL_MW, pf=ELECTROLYZER_PF, mode='always'
    )

    bus_v_ctrl, line_l_ctrl, trafo_l_ctrl, elec_status_ctrl, elec_p_ctrl, elec_q_ctrl = run_timeseries_control(
        active_df, reactive_df, gen_df, net_ctrl, p_nom=ELECTROLYZER_P_NOMINAL_MW, pf=ELECTROLYZER_PF, mode='controlled'
    )

    hours = range(DURATION_HOURS)

    # --- Plot: Voltage at electrolyzer bus (3 scenarios) ---
    plt.figure(figsize=(12, 5))
    plt.plot(hours, bus_v_no[str(ELECTROLYZER_BUS)], '--', label='No Electrolyzer')
    plt.plot(hours, bus_v_always[str(ELECTROLYZER_BUS)], '-', label='Electrolyzer Always ON')
    plt.plot(hours, bus_v_ctrl[str(ELECTROLYZER_BUS)], '-.', label='Electrolyzer Controlled ON/OFF')
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title(f"Voltage at Bus {ELECTROLYZER_BUS} (3 scenarios)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot: Reactive power Q (Always vs Controlled) ---
    plt.figure(figsize=(12, 4))
    plt.plot(hours, elec_q_always, '-', label='Q (Always ON)')
    plt.plot(hours, elec_q_ctrl, '-', label='Q (Controlled ON/OFF)')
    plt.xlabel("Hour")
    plt.ylabel("Reactive Power Q (Mvar)")
    plt.title("Electrolyzer Reactive Power (Always vs Controlled)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot: Electrolyzer ON/OFF status (Controlled) ---
    plt.figure(figsize=(12, 3))
    plt.step(hours, elec_status_ctrl.replace({'ON': 1, 'OFF': 0}).astype(int), where='post')
    plt.xlabel("Hour")
    plt.ylabel("ON=1 / OFF=0")
    plt.title("Electrolyzer ON/OFF Status (Controlled)")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot: Line loading for lines connected to electrolyzer bus (average) ---
    elec_lines = net_ctrl.line[(net_ctrl.line.from_bus == ELECTROLYZER_BUS) | (net_ctrl.line.to_bus == ELECTROLYZER_BUS)].index
    elec_lines_str = [str(i) for i in elec_lines]

    plt.figure(figsize=(12, 5))
    if len(elec_lines_str) > 0:
        plt.plot(hours, line_l_no[elec_lines_str].mean(axis=1), '--', label='No Electrolyzer')
        plt.plot(hours, line_l_always[elec_lines_str].mean(axis=1), '-', label='Always ON')
        plt.plot(hours, line_l_ctrl[elec_lines_str].mean(axis=1), '-.', label='Controlled')
    else:
        plt.text(0.5, 0.5, "No lines connected to electrolyzer bus", transform=plt.gca().transAxes)
    plt.xlabel("Hour")
    plt.ylabel("Line loading (%)")
    plt.title(f"Average Line Loading for Lines Connected to Bus {ELECTROLYZER_BUS}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot: Transformer loading over time (average / per trafo) ---
    if trafo_l_no is not None:
        plt.figure(figsize=(12, 5))
        # plot average trafo loading
        plt.plot(hours, trafo_l_no.mean(axis=1), '--', label='No Electrolyzer (avg)')
        plt.plot(hours, trafo_l_always.mean(axis=1), '-', label='Always ON (avg)')
        plt.plot(hours, trafo_l_ctrl.mean(axis=1), '-.', label='Controlled (avg)')
        plt.xlabel("Hour")
        plt.ylabel("Transformer loading (%)")
        plt.title("Average Transformer Loading Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Plot: Combined Voltage and Reactive Power at electrolyzer bus (Controlled + Always ON) ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: Voltage (Controlled)
    color1 = 'tab:blue'
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Voltage (p.u.)", color=color1)
    ax1.plot(hours, bus_v_ctrl[str(ELECTROLYZER_BUS)],
             linestyle='--', color=color1, linewidth=1.5, label='Voltage (Controlled)')
    # Reference voltage limits (dotted faint)
    ax1.axhline(V_OVER, color=color1, linestyle=':', linewidth=1, alpha=0.5, label=f"V_OVER={V_OVER:.3f}")
    ax1.axhline(V_UNDER, color=color1, linestyle=':', linewidth=1, alpha=0.5, label=f"V_UNDER={V_UNDER:.3f}")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Right axis: Reactive Power Q
    ax2 = ax1.twinx()
    color2_ctrl = 'tab:red'
    color2_always = 'tab:orange'
    ax2.set_ylabel("Reactive Power Q (Mvar)", color=color2_ctrl)

    # Q controlled (ON/OFF)
    ax2.plot(hours, elec_q_ctrl, linestyle='-', color=color2_ctrl, linewidth=2,
             label='Reactive Power (Controlled)')

    # Added: plot Q for always ON scenario
    ax2.plot(hours, elec_q_always, linestyle='--', color=color2_always, linewidth=2,
             label='Reactive Power (Always ON)')

    # Reference line Q=0
    ax2.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Q = 0')
    ax2.tick_params(axis='y', labelcolor=color2_ctrl)

    # Title and layout
    plt.title(f"Voltage and Reactive Power at Bus {ELECTROLYZER_BUS} (Controlled vs Always ON)")
    fig.tight_layout()

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.show()

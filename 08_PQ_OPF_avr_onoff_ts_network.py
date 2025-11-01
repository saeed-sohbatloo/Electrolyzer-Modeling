"""
Advanced Time-Series OPF Simulation with Electrolyzer and Local Voltage Control
===============================================================================

Author: Saeed Sohbatloo (adapted)
Date: 2025-10-31

Description:
------------
This script performs a time-series Optimal Power Flow (OPF) simulation on the MV CIGRE benchmark network
using pandapower. The main focus is on studying the impact of a controllable electrolyzer on network
voltage profiles under three scenarios:

1. No electrolyzer.
2. Electrolyzer without local voltage control.
3. Electrolyzer with local voltage control (AVR-style reactive power control).

The script handles multiple days of operation, hourly varying load and generation, and calculates
both active (P) and reactive (Q) power of the electrolyzer. Daily energy consumption and cost
are also tracked.

Scientific/Technical Details:
-----------------------------
1. The electrolyzer is modeled as a negative generator (sgen) or as a controllable load.
2. Electrolyzer ON/OFF is determined based on local bus voltage and line loading limits.
3. Reactive power (Q) of the electrolyzer can be adjusted to maintain local voltage (AVR behavior).
4. Active power (P) is limited by nominal capacity; if network conditions violate constraints, it is set to 0.
5. OPF is solved for each timestep, ensuring network feasibility and voltage limits.
6. Multi-day simulation: hourly profiles are concatenated, daily aggregation is performed for energy plots.
7. Forward-filling is applied to missing voltages to avoid plotting errors.
8. Outputs include:
   - CSV files for bus voltage, line loading, electrolyzer P/Q/cost.
   - Voltage comparison plots for the electrolyzer bus across all scenarios.
   - Daily energy bar charts for the electrolyzer.

Inputs:
-------
- Active load CSV [MW]
- Reactive load CSV [MVar]
- Generation CSV [MW]
- Electrolyzer nominal power, power factor, bus index
- Optional hourly electricity cost array

Outputs:
--------
- bus_voltage.csv
- line_loading.csv
- electrolyzer_results.csv
- Plots: electrolyzer P, Q, cost, voltage comparison, daily energy consumption
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional

# -------------------------
# === USER CONFIGURATION ===
# -------------------------
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"

OUT_DIR = "./timeseries_opf_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer parameters
ELECTROLYZER_BUS = 8        # Bus where electrolyzer is connected
ELECTROLYZER_POWER_MW = 1.0   # Max active power [MW]
ELECTROLYZER_PF = 0.95        # Power factor (used to compute Q)
COST_PER_HOUR = np.array([100, 120, 150, 180, 160, 140, 120, 110,
                          100, 90, 80, 70, 60, 70, 80, 90,
                          100, 120, 150, 180, 160, 140, 120, 100]) /10000 #added at updating part # €/MW

# Highlight buses and lines for plotting
HIGHLIGHT_BUSES = [ELECTROLYZER_BUS]
HIGHLIGHT_LINES = [11, 14]

DURATION_HOURS = 24   # total hours (5 days)
START_HOUR = 0

# -------------------------
# === HELPER FUNCTIONS ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

def map_node_to_bus(node_label, net: pp.pandapowerNet) -> Optional[int]:
    try:
        idx = int(node_label)
        if idx in net.bus.index: return idx
    except: pass
    if 'name' in net.bus.columns:
        for i, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return i
    return None

def assign_load(net: pp.pandapowerNet, bus_idx: int, p_mw: float, q_mvar: float = 0.0):
    existing = net.load[net.load.bus == bus_idx]
    if len(existing) > 0:
        total_existing = existing.p_mw.sum()
        for li in existing.index:
            share = net.load.at[li,'p_mw']/total_existing if total_existing>0 else 1/len(existing)
            net.load.at[li,'p_mw'] = p_mw*share
            net.load.at[li,'q_mvar'] = q_mvar*share
    else:
        pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=q_mvar)

def assign_generation(net: pp.pandapowerNet, gen_label: str, value_mw: float):
    bus_idx = map_node_to_bus(gen_label, net)
    if bus_idx is None: bus_idx = net.bus.index[0]
    sgens = net.sgen[net.sgen.bus == bus_idx]
    if len(sgens) > 0:
        for si in sgens.index: net.sgen.at[si,'p_mw'] = value_mw/len(sgens)
    else:
        pp.create_sgen(net, bus=bus_idx, p_mw=value_mw, q_mvar=0.0)
# -------------------------
# === TIMESERIES OPF FUNCTION WITH ELECTROLYZER CONTROL ===
# -------------------------
# -------------------------
# === TIMESERIES OPF FUNCTION WITH ELECTROLYZER CONTROL ===
# -------------------------
def run_timeseries_opf_with_control(net: pp.pandapowerNet,
                                    active_df: pd.DataFrame,
                                    reactive_df: Optional[pd.DataFrame] = None,
                                    gen_df: Optional[pd.DataFrame] = None,
                                    electrolyzer_bus: Optional[int] = None,
                                    electrolyzer_p_mw: float = 0.0,
                                    electrolyzer_pf: float = 1.0,
                                    duration_hours: int = 24,
                                    start_hour: int = 0,
                                    control_voltage: bool = False,
                                    line_loading_threshold: float = 100.0,
                                    voltage_threshold_pu: float = 0.98,
                                    q_max: float = 0.5,
                                    highlight_buses: List[int] = [],
                                    highlight_lines: List[int] = [],
                                    cost_per_hour: Optional[np.ndarray] = None,
                                    out_dir: str = "./"):

    elec_q_nom = electrolyzer_p_mw * np.tan(np.arccos(electrolyzer_pf))
    bus_v = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    line_l = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    trafo_l = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo)>0 else None
    electrolyzer_p, electrolyzer_q, electrolyzer_cost, electrolyzer_status = [], [], [], []

    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    # Extend hourly cost
    if cost_per_hour is not None:
        repeats = int(np.ceil(duration_hours / len(cost_per_hour)))
        cost_per_hour_full = np.tile(cost_per_hour, repeats)[:duration_hours]
    else:
        cost_per_hour_full = np.zeros(duration_hours)

    # Initialize sgen_idx
    sgen_idx = None

    # Add electrolyzer as controllable sgen
    if electrolyzer_bus is not None and electrolyzer_p_mw > 0:
        sgen_idx = pp.create_sgen(net, bus=electrolyzer_bus, p_mw=0.0, q_mvar=0.0,
                                  min_p_mw=-electrolyzer_p_mw, max_p_mw=0.0,
                                  min_q_mvar=-elec_q_nom, max_q_mvar=elec_q_nom,
                                  controllable=True, name="Electrolyzer")
        pp.create_poly_cost(net, sgen_idx, et='sgen', cp0_eur=0.0, cp1_eur_per_mw=0.0, cp2_eur_per_mw2=0.0)

    for t in range(duration_hours):
        col = active_df.columns[start_hour + t % active_df.shape[1]]

        # Update hourly cost
        if sgen_idx is not None:
            net.poly_cost.at[sgen_idx, "cp1_eur_per_mw"] = cost_per_hour_full[t]/10000

        # Assign loads
        for node in active_df.index:
            bus_idx = node_to_bus[str(node)]
            if bus_idx is None: continue
            assign_load(net, bus_idx, active_df.at[node,col],
                        reactive_df.at[node,col] if reactive_df is not None else 0.0)
        # Assign generation
        if gen_df is not None:
            for gl in gen_df.index:
                assign_generation(net, gl, gen_df.at[gl,col])

        # --- Electrolyzer ON/OFF & Q Control ---
        if sgen_idx is not None and control_voltage:
            pp.runpp(net)
            v_bus = net.res_bus.vm_pu.at[electrolyzer_bus]
            max_line = net.res_line.loading_percent.max()
            if (v_bus > voltage_threshold_pu) and (max_line < line_loading_threshold):
                p_set = -electrolyzer_p_mw
                status = "ON"
                # Simple local AVR
                if v_bus < 1.0:
                    q_set = min(q_max, (1.0 - v_bus)*10)
                else:
                    q_set = max(-q_max, (1.0 - v_bus)*10)
            else:
                p_set = 0.0
                q_set = 0.0
                status = "OFF"
            net.sgen.at[sgen_idx,'p_mw'] = p_set
            net.sgen.at[sgen_idx,'q_mvar'] = q_set
        elif sgen_idx is not None:
            # Always ON, no voltage control
            net.sgen.at[sgen_idx,'p_mw'] = -electrolyzer_p_mw
            net.sgen.at[sgen_idx,'q_mvar'] = 0.0
            status = "ON"

        # Run OPF
        try:
            pp.runopp(net, calculate_voltage_angles=True)
        except Exception as e:
            print(f"OPF failed at hour {t}: {e}")
            bus_v.loc[t,:] = np.nan
            line_l.loc[t,:] = np.nan
            if trafo_l is not None: trafo_l.loc[t,:] = np.nan
            electrolyzer_p.append(np.nan)
            electrolyzer_q.append(np.nan)
            electrolyzer_cost.append(np.nan)
            electrolyzer_status.append("FAIL")
            continue

        # Store results
        bus_v.loc[t,:] = net.res_bus.vm_pu
        line_l.loc[t,:] = net.res_line.loading_percent
        if trafo_l is not None: trafo_l.loc[t,:] = net.res_trafo.loading_percent
        if sgen_idx is not None:
            p_val = net.res_sgen.p_mw.at[sgen_idx]
            q_val = net.res_sgen.q_mvar.at[sgen_idx]
            electrolyzer_p.append(p_val)
            electrolyzer_q.append(q_val)
            electrolyzer_cost.append(-p_val * cost_per_hour_full[t])
            electrolyzer_status.append(status)
        else:
            electrolyzer_p.append(0.0)
            electrolyzer_q.append(0.0)
            electrolyzer_cost.append(0.0)
            electrolyzer_status.append("NA")

    return bus_v, line_l, trafo_l, electrolyzer_p, electrolyzer_q, electrolyzer_cost, electrolyzer_status

# === MAIN EXECUTION FOR THREE SCENARIOS ===
net_base = pn.create_cigre_network_mv()
net_no_ctrl = pn.create_cigre_network_mv()
net_ctrl = pn.create_cigre_network_mv()

active_df = read_timeseries_csv(ACTIVE_PATH)
reactive_df = read_timeseries_csv(REACTIVE_PATH)
gen_df = read_timeseries_csv(GEN_PATH)

# 1. No electrolyzer
bus_v_no, _, _, _, _, _, _ = run_timeseries_opf_with_control(net_base, active_df, reactive_df, gen_df,
                                                             electrolyzer_bus=None,
                                                             duration_hours=DURATION_HOURS)

# 2. Electrolyzer always ON, no voltage control
# 2. Electrolyzer always ON, no voltage control
bus_v_no_ctrl, _, _, elec_p_no_ctrl, elec_q_no_ctrl, elec_cost_no_ctrl, elec_status_no_ctrl = \
    run_timeseries_opf_with_control(net_no_ctrl, active_df, reactive_df, gen_df,
                                    electrolyzer_bus=ELECTROLYZER_BUS,
                                    electrolyzer_p_mw=ELECTROLYZER_POWER_MW,
                                    electrolyzer_pf=ELECTROLYZER_PF,
                                    duration_hours=DURATION_HOURS,
                                    control_voltage=False,
                                    cost_per_hour=COST_PER_HOUR)


# 3. Electrolyzer with local voltage control
bus_v_ctrl, _, _, elec_p_ctrl, elec_q_ctrl, elec_cost_ctrl, elec_status_ctrl = \
    run_timeseries_opf_with_control(net_ctrl, active_df, reactive_df, gen_df,
                                    electrolyzer_bus=ELECTROLYZER_BUS,
                                    electrolyzer_p_mw=ELECTROLYZER_POWER_MW,
                                    electrolyzer_pf=ELECTROLYZER_PF,
                                    duration_hours=DURATION_HOURS,
                                    control_voltage=True,
                                    cost_per_hour=COST_PER_HOUR)

# --- Plot voltage comparison ---
plt.figure(figsize=(12,6))
plt.plot(bus_v_no[ELECTROLYZER_BUS], '--', label="No Electrolyzer")
plt.plot(bus_v_no_ctrl[ELECTROLYZER_BUS], '-', label="Electrolyzer without voltage control")
plt.plot(bus_v_ctrl[ELECTROLYZER_BUS], '-', label="Electrolyzer with voltage control")
plt.xlabel("Hour")
plt.ylabel("Voltage (p.u.)")
plt.title(f"Voltage at Electrolyzer Bus {ELECTROLYZER_BUS}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

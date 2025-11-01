"""
Advanced Timeseries OPF with Controllable Electrolyzer (Multiple Days)
========================================================================

Description:
------------
This script runs a timeseries Optimal Power Flow (OPF) on the MV CIGRE benchmark network
using pandapower. A controllable electrolyzer is modeled as a negative generator (sgen),
allowing optimization of its active and reactive power across multiple days.

Features:
---------
1. Automatically extends hourly electricity cost for multiple days.
2. Tracks electrolyzer active/reactive power and cost per timestep.
3. Saves bus voltages, line and transformer loading.
4. Plots voltage and line-loading heatmaps.
5. Daily energy consumption summary.
6. Forward-fills missing voltages to avoid plotting errors.
7. Highlight specific buses and lines.

Inputs:
-------
- Active load CSV (MW)
- Reactive load CSV (MVar)
- Generation CSV (MW)
- Optional hourly electricity cost array

Outputs:
--------
- CSV files: bus_voltage.csv, line_loading.csv, trafo_loading.csv (if applicable)
- electrolyzer_results.csv
- Plots: electrolyzer power, reactive power, cost, voltage/line heatmaps, daily energy
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
                          100, 120, 150, 180, 160, 140, 120, 100]) #/10000 added at updating part # €/MW

# Highlight buses and lines for plotting
HIGHLIGHT_BUSES = [ELECTROLYZER_BUS]
HIGHLIGHT_LINES = [11, 14]

DURATION_HOURS = 120   # total hours (5 days)
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
# === TIMESERIES OPF FUNCTION ===
# -------------------------
def run_timeseries_opf_advanced(net: pp.pandapowerNet,
                                active_df: pd.DataFrame,
                                reactive_df: Optional[pd.DataFrame] = None,
                                gen_df: Optional[pd.DataFrame] = None,
                                electrolyzer_bus: Optional[int] = None,
                                electrolyzer_p_mw: float = 0.0,
                                electrolyzer_pf: float = 1.0,
                                duration_hours: int = 24,
                                start_hour: int = 0,
                                highlight_buses: List[int] = [],
                                highlight_lines: List[int] = [],
                                cost_per_hour: Optional[np.ndarray] = None,
                                out_dir: str = "./"):

    elec_q = electrolyzer_p_mw * np.tan(np.arccos(electrolyzer_pf))

    bus_v = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    line_l = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    trafo_l = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo)>0 else None
    electrolyzer_p, electrolyzer_q, electrolyzer_cost = [], [], []

    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    # --- Extend hourly cost for multiple days if needed ---
    if cost_per_hour is not None:
        repeats = int(np.ceil(duration_hours / len(cost_per_hour)))
        cost_per_hour_full = np.tile(cost_per_hour, repeats)[:duration_hours]
    else:
        cost_per_hour_full = np.zeros(duration_hours)

    # Add electrolyzer as controllable sgen (negative load)
    if electrolyzer_bus is not None and electrolyzer_p_mw>0:
        sgen_idx = pp.create_sgen(net, bus=electrolyzer_bus, p_mw=0.0, q_mvar=0.0,
                                  min_p_mw=-electrolyzer_p_mw, max_p_mw=0.0,
                                  min_q_mvar=-elec_q, max_q_mvar=elec_q,
                                  controllable=True, name="Electrolyzer")
        pp.create_poly_cost(net, sgen_idx, et='sgen', cp0_eur=0.0, cp1_eur_per_mw=0.0, cp2_eur_per_mw2=0.0)

    for t in range(duration_hours):
        col = active_df.columns[start_hour+t % active_df.shape[1]]  # wrap if necessary

        # Update hourly cost
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

        # Run OPF
        try:
            pp.runopp(net, calculate_voltage_angles=True)
        except Exception as e:
            print(f"Warning: OPF failed at timestep {t}: {e}")
            bus_v.loc[t,:] = np.nan
            line_l.loc[t,:] = np.nan
            if trafo_l is not None: trafo_l.loc[t,:] = np.nan
            electrolyzer_p.append(np.nan)
            electrolyzer_q.append(np.nan)
            electrolyzer_cost.append(np.nan)
            continue

        # Store results
        bus_v.loc[t,:] = net.res_bus.vm_pu
        line_l.loc[t,:] = net.res_line.loading_percent
        if trafo_l is not None: trafo_l.loc[t,:] = net.res_trafo.loading_percent
        p_val = net.res_sgen.p_mw.at[sgen_idx]
        q_val = net.res_sgen.q_mvar.at[sgen_idx]
        electrolyzer_p.append(p_val)
        electrolyzer_q.append(q_val)
        electrolyzer_cost.append(-p_val * cost_per_hour_full[t])

        print(f"Hour {t}: Electrolyzer P = {-p_val:.3f} MW, Cost = {electrolyzer_cost[-1]:.2f} €")

    bus_v_numeric = bus_v.astype(float).ffill()

    # --- Save CSV results ---
    bus_v.to_csv(os.path.join(out_dir,"bus_voltage.csv"))
    line_l.to_csv(os.path.join(out_dir,"line_loading.csv"))
    if trafo_l is not None: trafo_l.to_csv(os.path.join(out_dir,"trafo_loading.csv"))
    pd.DataFrame({"electrolyzer_p_mw": electrolyzer_p, "electrolyzer_q_mvar": electrolyzer_q,
                  "electrolyzer_cost_eur": electrolyzer_cost}).to_csv(os.path.join(out_dir,"electrolyzer_results.csv"), index=False)

    # --- Plotting ---
    hours = np.arange(duration_hours)
    electrolyzer_p_kw = np.array(electrolyzer_p) * -1000  # MW -> kW

    plt.figure(figsize=(12,4))
    plt.plot(hours, electrolyzer_p_kw, 'b-o'); plt.xlabel("Hour"); plt.ylabel("Power [kW]")
    plt.title("Electrolyzer Active Power"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"electrolyzer_power_kw.png")); plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(hours, electrolyzer_q, 'orange', marker='s'); plt.xlabel("Hour"); plt.ylabel("Reactive Power [MVar]")
    plt.title("Electrolyzer Reactive Power"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"electrolyzer_q.png")); plt.show()

    fig, ax1 = plt.subplots(figsize=(12,4))
    ax2 = ax1.twinx()
    ax1.plot(hours, electrolyzer_p_kw, 'b-o', label='Power [kW]')
    ax2.plot(hours, electrolyzer_cost, 'r-s', label='Cost [€]')
    ax1.set_xlabel("Hour"); ax1.set_ylabel("Power [kW]", color='b'); ax2.set_ylabel("Cost [€]", color='r')
    plt.title("Electrolyzer Power vs Cost"); ax1.grid(True); fig.tight_layout()
    plt.savefig(os.path.join(out_dir,"electrolyzer_power_cost.png")); plt.show()

    plt.figure(figsize=(12,6))
    plt.imshow(bus_v_numeric.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Voltage [pu]'); plt.xlabel("Hour"); plt.ylabel("Bus Index")
    plt.title("Voltage Profile Heatmap"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"bus_voltage_heatmap.png")); plt.show()

    plt.figure(figsize=(12,6))
    plt.imshow(line_l.T.astype(float), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='Loading [%]'); plt.xlabel("Hour"); plt.ylabel("Line Index")
    plt.title("Line Loading Heatmap"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"line_loading_heatmap.png")); plt.show()

    # --- Daily Energy Consumption (per day) ---
    hours_per_day = 24
    num_days = int(np.ceil(duration_hours / hours_per_day))

    daily_energy_kwh = []
    for d in range(num_days):
        start_idx = d * hours_per_day
        end_idx = min((d + 1) * hours_per_day, duration_hours)
        energy = np.sum(electrolyzer_p_kw[start_idx:end_idx])
        daily_energy_kwh.append(energy)

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_days + 1), daily_energy_kwh, color='green')
    plt.xlabel("Day")
    plt.ylabel("Energy [kWh]")
    plt.title("Daily Electrolyzer Energy Consumption")
    plt.xticks(range(1, num_days + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_energy_per_day.png"))
    plt.show()

    return bus_v, line_l, trafo_l, electrolyzer_p, electrolyzer_q, electrolyzer_cost

# -------------------------
# === MAIN EXECUTION ===
# -------------------------
if __name__=="__main__":
    net = pn.create_cigre_network_mv()
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH)
    gen_df = read_timeseries_csv(GEN_PATH)

    bus_v, line_l, trafo_l, elec_p, elec_q, elec_cost = run_timeseries_opf_advanced(
        net=net,
        active_df=active_df,
        reactive_df=reactive_df,
        gen_df=gen_df,
        electrolyzer_bus=ELECTROLYZER_BUS,
        electrolyzer_p_mw=ELECTROLYZER_POWER_MW,
        electrolyzer_pf=ELECTROLYZER_PF,
        duration_hours=DURATION_HOURS,
        start_hour=START_HOUR,
        highlight_buses=HIGHLIGHT_BUSES,
        highlight_lines=HIGHLIGHT_LINES,
        cost_per_hour=COST_PER_HOUR,
        out_dir=OUT_DIR
    )

    print("✅ Advanced timeseries OPF with electrolyzer and plots completed.")

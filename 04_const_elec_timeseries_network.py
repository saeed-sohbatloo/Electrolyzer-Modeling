# 02_timeseries_powerflow_cigre_advanced_plot.py
"""
Advanced Time-series Power Flow Runner with Electrolyzer
- Adds electrolyzer dynamically at a specified bus
- Compares before/after conditions for buses, lines, and transformers
- Includes plots for min/max/average and selected elements
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List

# -------------------------
# === USER CONFIGURATION ===
# -------------------------
# Paths to your CSVs (put your real paths here):
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

WITH_DER = False
START_HOUR = 0
DURATION_HOURS =  120
OUT_DIR = "./timeseries_results_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer configuration
ELECTROLYZER_BUS = 8
ELECTROLYZER_POWER_MW = 1.0
ELECTROLYZER_PF = 0.95

# Highlight buses and lines for custom plots
HIGHLIGHT_BUSES = [ELECTROLYZER_BUS]
HIGHLIGHT_LINES = [6,11,14]  # Example: lines for separate plot
# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df

def map_node_to_bus(node_label, net: pp.pandapowerNet) -> Optional[int]:
    try:
        idx = int(node_label)
        if idx in net.bus.index: return idx
    except: pass
    if 'name' in net.bus.columns:
        for i, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return i
    for i in net.bus.index:
        if str(i) == str(node_label).strip(): return i
    return None

def assign_load(net: pp.pandapowerNet, bus_idx: int, p_mw: float, q_mvar: float = 0.0):
    existing = net.load[net.load.bus == bus_idx]
    if len(existing) > 0:
        total_existing = existing.p_mw.sum()
        if total_existing == 0:
            for li in existing.index:
                net.load.at[li, 'p_mw'] = p_mw/len(existing)
                net.load.at[li, 'q_mvar'] = q_mvar/len(existing)
        else:
            for li in existing.index:
                share = net.load.at[li,'p_mw']/total_existing
                net.load.at[li,'p_mw'] = p_mw*share
                net.load.at[li,'q_mvar'] = q_mvar*share
    else:
        pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=q_mvar)

def assign_generation_profile(net: pp.pandapowerNet, gen_label: str, value_mw: float):
    bus_idx = map_node_to_bus(gen_label, net)
    if bus_idx is None: bus_idx = net.bus.index[0]
    sgens = net.sgen[net.sgen.bus == bus_idx]
    if len(sgens) > 0:
        for si in sgens.index: net.sgen.at[si,'p_mw'] = value_mw/len(sgens)
    else:
        pp.create_sgen(net, bus=bus_idx, p_mw=value_mw, q_mvar=0.0)

# -------------------------
# === Timeseries routine ===
# -------------------------
def run_timeseries_advanced(net: pp.pandapowerNet,
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
                            out_dir: str = "./"):

    elec_q = electrolyzer_p_mw * np.tan(np.arccos(electrolyzer_pf))

    # Initialize DataFrames
    bus_v_before = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    bus_v_after = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    line_l_before = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    line_l_after = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    trafo_l_before = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo)>0 else None
    trafo_l_after = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo)>0 else None

    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    for t in range(duration_hours):
        col = active_df.columns[start_hour+t]

        # Assign base loads
        for node in active_df.index:
            bus_idx = node_to_bus[str(node)]
            if bus_idx is None: continue
            p_mw = active_df.at[node,col]
            q_mw = reactive_df.at[node,col] if reactive_df is not None else 0.0
            assign_load(net,bus_idx,p_mw,q_mw)

        # --- PF before electrolyzer ---
        try:
            pp.runpp(net)
            bus_v_before.loc[t,:] = net.res_bus.vm_pu
            line_l_before.loc[t,:] = net.res_line.loading_percent
            if trafo_l_before is not None: trafo_l_before.loc[t,:] = net.res_trafo.loading_percent
        except pp.LoadflowNotConverged:
            bus_v_before.loc[t,:] = np.nan
            line_l_before.loc[t,:] = np.nan
            if trafo_l_before is not None: trafo_l_before.loc[t,:] = np.nan

        # Add electrolyzer
        if electrolyzer_bus is not None and electrolyzer_p_mw>0:
            assign_load(net,electrolyzer_bus,electrolyzer_p_mw,elec_q)

        # Assign generation
        if gen_df is not None:
            for gl in gen_df.index:
                assign_generation_profile(net,gl,gen_df.at[gl,col])

        # --- PF after electrolyzer ---
        try:
            pp.runpp(net)
            bus_v_after.loc[t,:] = net.res_bus.vm_pu
            line_l_after.loc[t,:] = net.res_line.loading_percent
            if trafo_l_after is not None: trafo_l_after.loc[t,:] = net.res_trafo.loading_percent
        except pp.LoadflowNotConverged:
            bus_v_after.loc[t,:] = np.nan
            line_l_after.loc[t,:] = np.nan
            if trafo_l_after is not None: trafo_l_after.loc[t,:] = np.nan

    # Save CSVs
    bus_v_before.to_csv(os.path.join(out_dir,"bus_v_before.csv"))
    bus_v_after.to_csv(os.path.join(out_dir,"bus_v_after.csv"))
    line_l_before.to_csv(os.path.join(out_dir,"line_l_before.csv"))
    line_l_after.to_csv(os.path.join(out_dir,"line_l_after.csv"))
    if trafo_l_before is not None:
        trafo_l_before.to_csv(os.path.join(out_dir,"trafo_l_before.csv"))
        trafo_l_after.to_csv(os.path.join(out_dir,"trafo_l_after.csv"))

    # ========================
    # === Plotting Section ===
    # ========================

    # --- 1) Bus voltage comparison for all highlight buses ---
    plt.figure(figsize=(12,5))
    for b in highlight_buses:
        plt.plot(bus_v_before.index,bus_v_before[b],label=f"Bus {b} Before")
        plt.plot(bus_v_after.index,bus_v_after[b],label=f"Bus {b} After")
        plt.plot(bus_v_after[b].idxmax(),bus_v_after[b].max(),'o',label=f"Max Bus {b}")
        plt.plot(bus_v_after[b].idxmin(),bus_v_after[b].min(),'x',label=f"Min Bus {b}")
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Electrolyzer's Bus Voltages Before/After Electrolyzer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"highlight_bus_voltage_comparison.png"))
    plt.show()

    # --- 2) Average + min/max line loading ---
    plt.figure(figsize=(12,5))
    avg_before = line_l_before.mean(axis=1)
    avg_after = line_l_after.mean(axis=1)
    max_line = line_l_after.max(axis=1)
    min_line = line_l_after.min(axis=1)
    plt.plot(avg_before,label="Avg Line Loading Before")
    plt.plot(avg_after,label="Avg Line Loading After")
    plt.plot(max_line,label=f"Max Line Loading After ({line_l_after.idxmax(axis=1)[0]})")
    plt.plot(min_line,label=f"Min Line Loading After ({line_l_after.idxmin(axis=1)[0]})")
    plt.xlabel("Hour")
    plt.ylabel("Loading (%)")
    plt.title("Line Loading Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"line_loading_avg_min_max.png"))
    plt.show()

    # --- 3) Individual line before/after ---
    for ln in highlight_lines:
        plt.figure(figsize=(10,5))
        plt.plot(line_l_before.index,line_l_before[ln],label=f"Line {ln} Before")
        plt.plot(line_l_after.index,line_l_after[ln],label=f"Line {ln} After")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title(f"Line {ln} Loading Before/After Electrolyzer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f"line_{ln}_comparison.png"))
        plt.show()

    # --- 4) Lines connected to a specific bus ---
    for bus in highlight_buses:
        connected_lines = net.line[(net.line.from_bus==bus)|(net.line.to_bus==bus)].index
        plt.figure(figsize=(12,5))
        for ln in connected_lines:
            plt.plot(line_l_before.index,line_l_before[ln],label=f"Line {ln} Before")
            plt.plot(line_l_after.index,line_l_after[ln],label=f"Line {ln} After")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title(f"Lines connected to Bus {bus} Loading Before/After")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f"lines_bus_{bus}_comparison.png"))
        plt.show()

    # --- 5) Transformers comparison ---
    if trafo_l_before is not None:
        plt.figure(figsize=(12,5))
        for tr in trafo_l_before.columns:
            plt.plot(trafo_l_before.index,trafo_l_before[tr],label=f"Trafo {tr} Before")
            plt.plot(trafo_l_after.index,trafo_l_after[tr],label=f"Trafo {tr} After")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title("Transformer Loading Before/After Electrolyzer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"trafo_comparison.png"))
        plt.show()

    return {
        "bus_v_before": bus_v_before,
        "bus_v_after": bus_v_after,
        "line_l_before": line_l_before,
        "line_l_after": line_l_after,
        "trafo_l_before": trafo_l_before,
        "trafo_l_after": trafo_l_after
    }

# -------------------------
# === Main Execution ===
# -------------------------
if __name__=="__main__":
    net = pn.create_cigre_network_mv(with_der=WITH_DER)
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH)
    gen_df = read_timeseries_csv(GEN_PATH)

    results = run_timeseries_advanced(
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
        out_dir=OUT_DIR
    )

    print("✅ Advanced timeseries completed with additional plots.")

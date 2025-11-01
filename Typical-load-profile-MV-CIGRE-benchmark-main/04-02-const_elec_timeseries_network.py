# 02_timeseries_powerflow_cigre_with_electrolyzer.py
"""
Time-series power flow runner for the CIGRE MV network (pandapower) with electrolyzer.
- Adds an electrolyzer dynamically at a specified bus for each timestep.
- Ensures original network loads and generation are preserved.
- Collects bus voltages, line and transformer loading, total load/generation/losses.
- Plots comparison of network conditions before and after electrolyzer installation.
- Saves results as CSVs for post-processing.
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
ACTIVE_PATH = "/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"


WITH_DER = False  # Use network with DER elements
START_HOUR = 0
DURATION_HOURS = 24
OUT_DIR = "../timeseries_results/timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer configuration
ELECTROLYZER_BUS = 5  # bus index for electrolyzer
ELECTROLYZER_POWER_MW = 10.0
ELECTROLYZER_PF = 0.95  # lagging power factor
HIGHLIGHT_BUSES = [ELECTROLYZER_BUS]  # for plotting

# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    """Read a CSV timeseries with first column as node index/name."""
    df = pd.read_csv(path, index_col=0)
    return df

def map_node_to_bus(node_label, net: pp.pandapowerNet) -> Optional[int]:
    """Map a CSV node label to a pandapower bus index."""
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

def assign_load(net: pp.pandapowerNet, bus_idx: int, p_mw: float, q_mvar: float = 0.0):
    """
    Assign or update a load at bus_idx.
    If existing loads, values are overwritten proportionally.
    """
    existing = net.load[net.load.bus == bus_idx]
    if len(existing) > 0:
        total_existing = existing.p_mw.sum()
        if total_existing == 0:
            for li in existing.index:
                net.load.at[li, 'p_mw'] = p_mw / len(existing)
                net.load.at[li, 'q_mvar'] = q_mvar / len(existing)
        else:
            for li in existing.index:
                share = net.load.at[li, 'p_mw'] / total_existing
                net.load.at[li, 'p_mw'] = p_mw * share
                net.load.at[li, 'q_mvar'] = q_mvar * share
    else:
        pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=q_mvar)

def assign_generation_profile(net: pp.pandapowerNet, gen_label: str, value_mw: float):
    """Assign or create sgen at bus based on label mapping."""
    bus_idx = map_node_to_bus(gen_label, net)
    if bus_idx is None:
        bus_idx = net.bus.index[0]  # fallback to slack bus
    sgens = net.sgen[net.sgen.bus == bus_idx]
    if len(sgens) > 0:
        for si in sgens.index:
            net.sgen.at[si, 'p_mw'] = value_mw / len(sgens)
    else:
        pp.create_sgen(net, bus=bus_idx, p_mw=value_mw, q_mvar=0.0)

# -------------------------
# === Main timeseries function ===
# -------------------------
def run_timeseries_with_electrolyzer(net: pp.pandapowerNet,
                                     active_df: pd.DataFrame,
                                     reactive_df: Optional[pd.DataFrame] = None,
                                     gen_df: Optional[pd.DataFrame] = None,
                                     electrolyzer_bus: Optional[int] = None,
                                     electrolyzer_p_mw: float = 0.0,
                                     electrolyzer_pf: float = 1.0,
                                     duration_hours: int = 24,
                                     start_hour: int = 0,
                                     out_dir: str = "./"):
    """
    Runs time-series PF, adds electrolyzer at each timestep, and saves voltages and loading.
    Produces before/after comparison for bus voltages, line and transformer loadings.
    """
    elec_q = electrolyzer_p_mw * np.tan(np.arccos(electrolyzer_pf))

    bus_voltage_before = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    bus_voltage_after = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    line_loading_before = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    line_loading_after = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    trafo_loading_before = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo) > 0 else None
    trafo_loading_after = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo) > 0 else None

    node_to_bus_map = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    for t in range(duration_hours):
        col = active_df.columns[start_hour + t]

        # --- Assign base loads ---
        for node in active_df.index:
            bus_idx = node_to_bus_map[str(node)]
            if bus_idx is None:
                continue
            p_mw = active_df.at[node, col]
            q_mw = reactive_df.at[node, col] if reactive_df is not None else 0.0
            assign_load(net, bus_idx, p_mw, q_mw)

        # --- Run PF before electrolyzer ---
        try:
            pp.runpp(net)
            bus_voltage_before.loc[t, :] = net.res_bus.vm_pu
            line_loading_before.loc[t, :] = net.res_line.loading_percent
            if trafo_loading_before is not None:
                trafo_loading_before.loc[t, :] = net.res_trafo.loading_percent
        except pp.LoadflowNotConverged:
            bus_voltage_before.loc[t, :] = np.nan
            line_loading_before.loc[t, :] = np.nan
            if trafo_loading_before is not None:
                trafo_loading_before.loc[t, :] = np.nan

        # --- Add electrolyzer load ---
        if electrolyzer_bus is not None and electrolyzer_p_mw > 0:
            assign_load(net, electrolyzer_bus, electrolyzer_p_mw, elec_q)

        # --- Assign generation ---
        if gen_df is not None:
            for gl in gen_df.index:
                assign_generation_profile(net, gl, gen_df.at[gl, col])

        # --- Run PF after electrolyzer ---
        try:
            pp.runpp(net)
            bus_voltage_after.loc[t, :] = net.res_bus.vm_pu
            line_loading_after.loc[t, :] = net.res_line.loading_percent
            if trafo_loading_after is not None:
                trafo_loading_after.loc[t, :] = net.res_trafo.loading_percent
        except pp.LoadflowNotConverged:
            bus_voltage_after.loc[t, :] = np.nan
            line_loading_after.loc[t, :] = np.nan
            if trafo_loading_after is not None:
                trafo_loading_after.loc[t, :] = np.nan

    # --- Save CSVs ---
    bus_voltage_before.to_csv(os.path.join(out_dir, "bus_voltage_before.csv"))
    bus_voltage_after.to_csv(os.path.join(out_dir, "bus_voltage_after.csv"))
    line_loading_before.to_csv(os.path.join(out_dir, "line_loading_before.csv"))
    line_loading_after.to_csv(os.path.join(out_dir, "line_loading_after.csv"))
    if trafo_loading_before is not None:
        trafo_loading_before.to_csv(os.path.join(out_dir, "trafo_loading_before.csv"))
        trafo_loading_after.to_csv(os.path.join(out_dir, "trafo_loading_after.csv"))

    # -------------------------
    # === Plotting ===
    # -------------------------
    # 1) Bus voltage comparison at electrolyzer bus
    if electrolyzer_bus is not None:
        plt.figure(figsize=(10,5))
        plt.plot(bus_voltage_before.index, bus_voltage_before[electrolyzer_bus], label="Before Electrolyzer")
        plt.plot(bus_voltage_after.index, bus_voltage_after[electrolyzer_bus], label="After Electrolyzer")
        plt.xlabel("Hour")
        plt.ylabel("Voltage (p.u.)")
        plt.title(f"Bus {electrolyzer_bus} Voltage Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bus_{electrolyzer_bus}_voltage_comparison.png"))
        plt.show()

    # 2) Line loading comparison (sum over all lines)
    plt.figure(figsize=(10,5))
    plt.plot(line_loading_before.mean(axis=1), label="Avg Line Loading Before")
    plt.plot(line_loading_after.mean(axis=1), label="Avg Line Loading After")
    plt.xlabel("Hour")
    plt.ylabel("Loading (%)")
    plt.title("Average Line Loading Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_loading_comparison.png"))
    plt.show()

    # 3) Transformer loading comparison
    if trafo_loading_before is not None:
        plt.figure(figsize=(10,5))
        plt.plot(trafo_loading_before.mean(axis=1), label="Avg Trafo Loading Before")
        plt.plot(trafo_loading_after.mean(axis=1), label="Avg Trafo Loading After")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title("Average Transformer Loading Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trafo_loading_comparison.png"))
        plt.show()

    return {
        "bus_voltage_before": bus_voltage_before,
        "bus_voltage_after": bus_voltage_after,
        "line_loading_before": line_loading_before,
        "line_loading_after": line_loading_after,
        "trafo_loading_before": trafo_loading_before,
        "trafo_loading_after": trafo_loading_after
    }


# -------------------------
# === Main execution ===
# -------------------------
if __name__ == "__main__":
    # Create network
    net = pn.create_cigre_network_mv(with_der=False)

    # Run timeseries with electrolyzer
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH)
    gen_df = read_timeseries_csv(GEN_PATH)

    results = run_timeseries_with_electrolyzer(
        net=net,
        active_df=active_df,
        reactive_df=reactive_df,
        gen_df=gen_df,
        electrolyzer_bus=ELECTROLYZER_BUS,
        electrolyzer_p_mw=ELECTROLYZER_POWER_MW,
        electrolyzer_pf=ELECTROLYZER_PF,
        start_hour=START_HOUR,
        duration_hours=DURATION_HOURS,
        out_dir=OUT_DIR
    )

    print("✅ Time-series completed. CSVs and plots saved in:", OUT_DIR)

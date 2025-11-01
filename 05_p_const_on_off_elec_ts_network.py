# 02_timeseries_powerflow_cigre_electrolyzer_control.py
"""
Time-series power flow runner for the CIGRE MV test network (pandapower)
with local ON/OFF control logic for an electrolyzer.

Behavior:
- Loads CSV time series for active consumption, reactive consumption, and generation.
- Maps CSV rows to pandapower buses (flexible matching).
- Runs power flow for each hour:
    1) Apply base loads (electrolyzer load kept at 0)
    2) Run PF => measure max line loading and local bus voltage
    3) Apply local control rule:
         if (max_line_loading < line_thresh) and (v_at_bus > v_thresh):
             turn electrolyzer ON (p = p_nominal)
         else:
             keep electrolyzer OFF (p = 0)
    4) Run PF again to obtain "after" results
- Collects results and produces comparison plots (before/after):
    - Bus voltages (highlighted)
    - Line loadings (selected lines and avg/min/max)
    - Lines connected to chosen bus
    - Transformer loading before/after
    - Electrolyzer ON/OFF status time series
"""

import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn

# -------------------------
# === USER CONFIGURATION ===
# -------------------------
## Paths to your CSVs (put your real paths here):
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

WITH_DER = False  # create template network with DERs or not
START_HOUR = 0
DURATION_HOURS = 72
OUT_DIR = "timeseries_results/timeseries_results_controlled"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer (control) parameters
ELECTROLYZER_BUS = None        # set interactively below or set an integer here
ELECTROLYZER_P_NOMINAL_MW = 1.0  # nominal electrolyzer power in MW (example)
ELECTROLYZER_PF = 0.95         # power factor of electrolyzer (lagging)
LINE_LOADING_THRESHOLD = 100.0  # percent (%) threshold for line loading
VOLTAGE_THRESHOLD_PU = 0.98    # voltage threshold in p.u.
#q_nominal_mvar = ELECTROLYZER_P_NOMINAL_MW * np.tan(np.arccos(ELECTROLYZER_PF))

# Plotting highlights
HIGHLIGHT_BUSES = []  # list of bus indices to highlight (will include electrolyzer bus)
HIGHLIGHT_LINES = []  # list of line indices for single-line comparisons

# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    """Read CSV timeseries file where index=nodes and columns=time steps."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, index_col=0)
    return df

def map_node_to_bus(node_label, net: pp.pandapowerNet) -> Optional[int]:
    """
    Try to map a node label from CSV to a pandapower bus index.
    Accepts numeric indices, bus name matching (case-insensitive), or string of index.
    """
    # try integer index
    try:
        idx = int(node_label)
        if idx in net.bus.index:
            return idx
    except Exception:
        pass

    # try matching bus name
    if 'name' in net.bus.columns:
        for i, name in zip(net.bus.index, net.bus['name']):
            if str(name).strip().lower() == str(node_label).strip().lower():
                return i

    # try matching stringified index
    for i in net.bus.index:
        if str(i) == str(node_label).strip():
            return i
    return None

def ensure_electrolyzer_load(net: pp.pandapowerNet, bus_idx: int, name: str = "Electrolyzer"):
    """
    Ensure electrolyzer load element exists in the network.
    If already exists (by name) it does nothing, else it creates a load with p_mw=0 initially.
    Returns the load index.
    """
    existing = net.load[net.load['name'] == name]
    if len(existing) > 0:
        return existing.index[0]
    # create new load with zero power initially (we will update p_mw each timestep)
    lid = pp.create_load(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=name)
    return lid

def set_electrolyzer_power(net: pp.pandapowerNet, name: str, p_mw: float, q_mvar: float):
    """
    Set electrolyzer load power by name. If not found, raises ValueError.
    """
    idx = net.load[net.load['name'] == name].index
    if len(idx) == 0:
        raise ValueError(f"Electrolyzer load named '{name}' not found in net.load")
    i = idx[0]
    net.load.at[i, 'p_mw'] = float(p_mw)
    net.load.at[i, 'q_mvar'] = float(q_mvar)

def assign_base_loads(net: pp.pandapowerNet,
                      active_df: pd.DataFrame,
                      reactive_df: Optional[pd.DataFrame],
                      node_to_bus_map: Dict[str, Optional[int]],
                      t_col: str,
                      electrolyzer_name: str = "Electrolyzer"):
    """
    Assign the base time-varying loads from CSV to network loads for one time column.
    This function will:
    - For each CSV node that maps to a bus, sum the node's p/q into the existing loads at that bus.
    - When distributing between multiple load elements at the same bus, it WILL NOT override
      the electrolyzer named element (we treat electrolyzer as separate and keep it unchanged here).
    IMPORTANT: This function overwrites p_mw/q_mvar of existing non-electrolyzer loads to match the distribution
    of the CSV value among those load elements.
    """
    # For each bus, collect CSV total p and q to assign (excluding electrolyzer)
    bus_aggregate = {}  # bus_idx -> (p_mw_total, q_mvar_total)
    for node in active_df.index:
        bus_idx = node_to_bus_map.get(str(node))
        if bus_idx is None:
            continue
        p_kw = float(active_df.at[node, t_col])
        p_mw = p_kw / 1.0  # CSV already in MW in your workflow; if kW -> divide by 1000
        q_mvar = 0.0
        if reactive_df is not None and node in reactive_df.index:
            q_kw = float(reactive_df.at[node, t_col])
            q_mvar = q_kw / 1.0
        prev = bus_aggregate.get(bus_idx, (0.0, 0.0))
        bus_aggregate[bus_idx] = (prev[0] + p_mw, prev[1] + q_mvar)

    # Now assign to existing loads at each bus, excluding the electrolyzer load
    for bus_idx, (p_total, q_total) in bus_aggregate.items():
        loads_at_bus = net.load[(net.load.bus == bus_idx) & (net.load.name != electrolyzer_name)]
        if len(loads_at_bus) > 0:
            total_existing = loads_at_bus.p_mw.sum()
            if total_existing == 0:
                # distribute evenly among non-electrolyzer loads
                for li in loads_at_bus.index:
                    net.load.at[li, 'p_mw'] = p_total / len(loads_at_bus)
                    net.load.at[li, 'q_mvar'] = q_total / len(loads_at_bus)
            else:
                # preserve proportional shares relative to their existing p_mw
                for li in loads_at_bus.index:
                    share = net.load.at[li, 'p_mw'] / total_existing if total_existing > 0 else 1.0/len(loads_at_bus)
                    net.load.at[li, 'p_mw'] = p_total * share
                    net.load.at[li, 'q_mvar'] = q_total * share
        else:
            # no non-electrolyzer loads at this bus -> create a single load representing CSV
            pp.create_load(net, bus=bus_idx, p_mw=p_total, q_mvar=q_total, name=f"ts_load_bus_{bus_idx}")

def assign_generation_profiles(net: pp.pandapowerNet, gen_df: Optional[pd.DataFrame], t_col: str):
    """
    Assign generation profiles from gen_df to sgens in net.
    The mapping is attempted by row name -> bus (map_node_to_bus logic).
    If mapping fails, the generation is placed on the first bus (fallback).
    """
    if gen_df is None:
        return
    for gen_label in gen_df.index:
        val_kw = float(gen_df.at[gen_label, t_col])
        p_mw = val_kw / 1.0
        # simple mapping: if a generator row matches a bus name or index, set sgen on that bus
        mapped_bus = map_node_to_bus(gen_label, net)
        if mapped_bus is None:
            mapped_bus = int(net.bus.index[0])
        sgens_at_bus = net.sgen[net.sgen.bus == mapped_bus]
        if len(sgens_at_bus) > 0:
            # distribute evenly among sgens at that bus
            for si in sgens_at_bus.index:
                net.sgen.at[si, 'p_mw'] = p_mw / len(sgens_at_bus)
        else:
            pp.create_sgen(net, bus=mapped_bus, p_mw=p_mw, q_mvar=0.0, name=f"ts_sgen_bus_{mapped_bus}")

# -------------------------
# === Main controlled timeseries routine ===
# -------------------------
def run_timeseries_with_local_control(active_path: str,
                                      reactive_path: Optional[str],
                                      gen_path: Optional[str],
                                      net: pp.pandapowerNet,
                                      start_hour: int,
                                      duration_hours: int,
                                      out_dir: str,
                                      electrolyzer_bus: int,
                                      p_nominal_mw: float,
                                      pf: float,
                                      line_thresh: float,
                                      v_thresh: float,
                                      highlight_buses: List[int] = None,
                                      highlight_lines: List[int] = None):
    """
    Main runner:
    - active_path, reactive_path, gen_path: CSV files
    - electrolyzer_bus: bus index where electrolyzer sits
    - p_nominal_mw: nominal active power when ON (MW)
    - pf: power factor (lagging) -> used to calculate q
    - line_thresh, v_thresh: decision thresholds
    Returns dict of DataFrames with before/after results and electrolyzer status.
    """
    active_df = read_timeseries_csv(active_path)
    reactive_df = read_timeseries_csv(reactive_path) if reactive_path else None
    gen_df = read_timeseries_csv(gen_path) if gen_path else None

    # check length
    available_hours = active_df.shape[1]
    if start_hour + duration_hours > available_hours:
        raise ValueError("Requested simulation range exceeds timeseries length.")

    # ensure electrolyzer load exists (p_mw initially zero)
    ELEC_NAME = "Electrolyzer_controlled"
    ensure_electrolyzer_load(net, electrolyzer_bus, name=ELEC_NAME)
    # make sure electrolyzer is zero to start
    set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)

    # pre-map CSV node->bus to speed up
    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    # prepare result containers (index 0..duration_hours-1)
    hours = list(range(duration_hours))
    bus_v_before = pd.DataFrame(index=hours, columns=net.bus.index.astype(str), dtype=float)
    bus_v_after = pd.DataFrame(index=hours, columns=net.bus.index.astype(str), dtype=float)
    line_l_before = pd.DataFrame(index=hours, columns=net.line.index.astype(str), dtype=float)
    line_l_after = pd.DataFrame(index=hours, columns=net.line.index.astype(str), dtype=float)
    trafo_l_before = pd.DataFrame(index=hours, columns=net.trafo.index.astype(str), dtype=float) if len(net.trafo)>0 else None
    trafo_l_after = pd.DataFrame(index=hours, columns=net.trafo.index.astype(str), dtype=float) if len(net.trafo)>0 else None
    totals_before = pd.DataFrame(index=hours, columns=['gen_mw','load_mw','losses_mw'], dtype=float)
    totals_after = pd.DataFrame(index=hours, columns=['gen_mw','load_mw','losses_mw'], dtype=float)

    elec_status = pd.Series(index=hours, dtype=object)   # 'ON' or 'OFF'
    elec_p_set = pd.Series(index=hours, dtype=float)
    elec_q_set = pd.Series(index=hours, dtype=float)

    # Pre-calc electrolyzer reactive per nominal p
    q_nominal = p_nominal_mw * np.tan(np.arccos(pf))

    print(f"Starting controlled time-series run: hours {start_hour} .. {start_hour + duration_hours - 1}")
    for t in hours:
        col = active_df.columns[start_hour + t]

        # --- 1) Assign base loads from CSV (do NOT touch electrolyzer load here) ---
        assign_base_loads(net, active_df, reactive_df, node_to_bus, t_col=col, electrolyzer_name=ELEC_NAME)

        # --- Optionally assign generation profile (before decision) to reflect base gen ---
        # If you want generation to be considered during decision, assign it here.
        assign_generation_profiles(net, gen_df, t_col=col)

        # --- 2) Run PF to measure 'before' state (electrolyzer currently p=0) ---
        try:
            pp.runpp(net, calculate_voltage_angles=True)
            # record bus voltages and line/trafo loading
            bus_v_before.loc[t, :] = net.res_bus.vm_pu.values
            line_l_before.loc[t, :] = net.res_line.loading_percent.values
            if trafo_l_before is not None:
                trafo_l_before.loc[t, :] = net.res_trafo.loading_percent.values
            # totals
            gen_sum = (net.res_gen.p_mw.sum() if hasattr(net, 'res_gen') else 0.0) + (net.res_sgen.p_mw.sum() if hasattr(net, 'res_sgen') else 0.0)
            load_sum = net.res_load.p_mw.sum() if hasattr(net, 'res_load') else 0.0
            losses = 0.0
            if hasattr(net, 'res_line') and 'pl_mw' in net.res_line.columns:
                losses += net.res_line.pl_mw.sum()
            if hasattr(net, 'res_trafo') and 'pl_mw' in net.res_trafo.columns:
                losses += net.res_trafo.pl_mw.sum()
            totals_before.loc[t, :] = [gen_sum, load_sum, losses]
        except pp.LoadflowNotConverged:
            bus_v_before.loc[t, :] = np.nan
            line_l_before.loc[t, :] = np.nan
            if trafo_l_before is not None:
                trafo_l_before.loc[t, :] = np.nan
            totals_before.loc[t, :] = [np.nan, np.nan, np.nan]

        # --- 3) Decide ON/OFF using control rule based on 'before' results ---
        # compute max line loading and local voltage at electrolyzer bus (use before results)
        try:
            max_line_loading = float(line_l_before.loc[t, :].max())
        except Exception:
            max_line_loading = np.nan
        try:
            v_at_bus = float(bus_v_before.loc[t, str(electrolyzer_bus)])
        except Exception:
            v_at_bus = np.nan

        # Control logic
        if np.isfinite(max_line_loading) and np.isfinite(v_at_bus):
            if (max_line_loading < line_thresh) and (v_at_bus > v_thresh):
                # Turn on electrolyzer: set electrolyzer load to nominal P and Q
                set_electrolyzer_power(net, ELEC_NAME, p_nominal_mw, q_nominal)
                elec_status.loc[t] = "ON"
                elec_p_set.loc[t] = p_nominal_mw
                elec_q_set.loc[t] = q_nominal
            else:
                # Turn off
                set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)
                elec_status.loc[t] = "OFF"
                elec_p_set.loc[t] = 0.0
                elec_q_set.loc[t] = 0.0
        else:
            # if before-run had NaNs or missing data, default OFF
            set_electrolyzer_power(net, ELEC_NAME, 0.0, 0.0)
            elec_status.loc[t] = "OFF"
            elec_p_set.loc[t] = 0.0
            elec_q_set.loc[t] = 0.0

        # --- 4) After applying electrolyzer power, re-run PF to obtain 'after' results ---
        try:
            pp.runpp(net, calculate_voltage_angles=True)
            bus_v_after.loc[t, :] = net.res_bus.vm_pu.values
            line_l_after.loc[t, :] = net.res_line.loading_percent.values
            if trafo_l_after is not None:
                trafo_l_after.loc[t, :] = net.res_trafo.loading_percent.values
            gen_sum = (net.res_gen.p_mw.sum() if hasattr(net, 'res_gen') else 0.0) + (net.res_sgen.p_mw.sum() if hasattr(net, 'res_sgen') else 0.0)
            load_sum = net.res_load.p_mw.sum() if hasattr(net, 'res_load') else 0.0
            losses = 0.0
            if hasattr(net, 'res_line') and 'pl_mw' in net.res_line.columns:
                losses += net.res_line.pl_mw.sum()
            if hasattr(net, 'res_trafo') and 'pl_mw' in net.res_trafo.columns:
                losses += net.res_trafo.pl_mw.sum()
            totals_after.loc[t, :] = [gen_sum, load_sum, losses]
        except pp.LoadflowNotConverged:
            bus_v_after.loc[t, :] = np.nan
            line_l_after.loc[t, :] = np.nan
            if trafo_l_after is not None:
                trafo_l_after.loc[t, :] = np.nan
            totals_after.loc[t, :] = [np.nan, np.nan, np.nan]

        # (Optional) Print status for debugging
        print(f"Hour {t}: status={elec_status.loc[t]}, max_line_before={max_line_loading:.2f}, v_at_bus_before={v_at_bus:.3f}")

    # Save CSV outputs
    bus_v_before.to_csv(os.path.join(out_dir, "bus_voltage_before.csv"), index_label="hour")
    bus_v_after.to_csv(os.path.join(out_dir, "bus_voltage_after.csv"), index_label="hour")
    line_l_before.to_csv(os.path.join(out_dir, "line_loading_before.csv"), index_label="hour")
    line_l_after.to_csv(os.path.join(out_dir, "line_loading_after.csv"), index_label="hour")
    totals_before.to_csv(os.path.join(out_dir, "totals_before.csv"), index_label="hour")
    totals_after.to_csv(os.path.join(out_dir, "totals_after.csv"), index_label="hour")
    elec_status.to_csv(os.path.join(out_dir, "electrolyzer_status.csv"), index_label="hour")
    elec_p_set.to_csv(os.path.join(out_dir, "electrolyzer_p_set.csv"), index_label="hour")
    elec_q_set.to_csv(os.path.join(out_dir, "electrolyzer_q_set.csv"), index_label="hour")
    if trafo_l_before is not None:
        trafo_l_before.to_csv(os.path.join(out_dir, "trafo_loading_before.csv"), index_label="hour")
        trafo_l_after.to_csv(os.path.join(out_dir, "trafo_loading_after.csv"), index_label="hour")


    print(f"Results saved to {out_dir}")

    # -------------------------
    # === Plotting Section ===
    # -------------------------

    # 1) Voltage comparison for highlighted buses
    plt.figure(figsize=(12, 6))
    if highlight_buses is None:
        highlight_buses = [electrolyzer_bus]
    for b in highlight_buses:
        b_str = str(b)
        series_before = bus_v_before[b_str].astype(float)
        series_after = bus_v_after[b_str].astype(float)
        plt.plot(hours, series_before, label=f"Bus {b} With Electrolyzer - Ohne Controller", linestyle='--')
        plt.plot(hours, series_after, label=f"Bus {b} With Electrolyzer - Mit ON/OFF Controller", linestyle='-')
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage Comparison at Highlighted Buses (Both with Electrolyzer)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "highlight_buses_voltage_comparison.png"))
    plt.show()

    # 2) Line loading: average + min/max removed
    plt.figure(figsize=(12, 5))
    avg_before = line_l_before.mean(axis=1)
    avg_after = line_l_after.mean(axis=1)
    plt.plot(hours, avg_before, label="Avg Line Loading With Electrolyzer - Ohne Controller", linestyle='--')
    plt.plot(hours, avg_after, label="Avg Line Loading With Electrolyzer - Mit ON/OFF Controller", linestyle='-')
    plt.xlabel("Hour")
    plt.ylabel("Loading (%)")
    plt.title("Line Loading Comparison (Both with Electrolyzer, max/min points removed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lines_avg_comparison.png"))
    plt.show()

    # 3) Individual highlighted lines: before vs after, min/max points removed
    for ln in (highlight_lines or []):
        ln_str = str(ln)
        if ln_str not in line_l_before.columns:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(hours, line_l_before[ln_str].astype(float), label=f"Line {ln} With Electrolyzer - Ohne Controller",
                 linestyle='--')
        plt.plot(hours, line_l_after[ln_str].astype(float),
                 label=f"Line {ln} With Electrolyzer - Mit ON/OFF Controller", linestyle='-')
        mean_after = line_l_after[ln_str].astype(float).mean()
        plt.axhline(mean_after, color='gray', linestyle=':', label=f"Mean After ({mean_after:.2f}%)")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title(f"Line {ln} Loading Comparison (Both with Electrolyzer)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"line_{ln}_before_after.png"))
        plt.show()

    # 4) Lines connected to the electrolyzer bus
    connected_lines_idx = net.line[
        (net.line.from_bus == electrolyzer_bus) | (net.line.to_bus == electrolyzer_bus)].index
    if len(connected_lines_idx) > 0:
        plt.figure(figsize=(12, 6))
        for li in connected_lines_idx:
            li_str = str(li)
            plt.plot(hours, line_l_before[li_str].astype(float), label=f"Line {li} With Electrolyzer - Ohne Controller",
                     linestyle='--')
            plt.plot(hours, line_l_after[li_str].astype(float),
                     label=f"Line {li} With Electrolyzer - Mit ON/OFF Controller", linestyle='-')
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title(f"Lines connected to Bus {electrolyzer_bus} (Both with Electrolyzer, max/min points removed)")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"lines_connected_bus_{electrolyzer_bus}_comparison.png"))
        plt.show()

    # 5) Transformers comparison
    if trafo_l_before is not None:
        plt.figure(figsize=(12, 6))
        for tr in trafo_l_before.columns:
            plt.plot(hours, trafo_l_before[tr].astype(float), label=f"Trafo {tr} With Electrolyzer - Ohne Controller",
                     linestyle='--')
            plt.plot(hours, trafo_l_after[tr].astype(float),
                     label=f"Trafo {tr} With Electrolyzer - Mit ON/OFF Controller", linestyle='-')
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.title("Transformer Loading Comparison (Both with Electrolyzer)")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trafo_before_after.png"))
        plt.show()

    # 6) Electrolyzer status (ON/OFF) and P setpoint
    plt.figure(figsize=(12, 3))
    status_numeric = elec_status.replace({'ON': 1, 'OFF': 0}).astype(float)
    plt.step(hours, status_numeric, where='mid', label='Electrolyzer ON(1)/OFF(0)')
    plt.twinx()
    plt.plot(hours, elec_p_set.astype(float), color='orange', label='Electrolyzer P_set (MW)')
    plt.title("Electrolyzer Status and P_set (Both with Electrolyzer)")
    plt.xlabel("Hour")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "electrolyzer_status_and_pset.png"))
    plt.show()



    # return results
    return {
        "bus_v_before": bus_v_before,
        "bus_v_after": bus_v_after,
        "line_l_before": line_l_before,
        "line_l_after": line_l_after,
        "trafo_l_before": trafo_l_before,
        "trafo_l_after": trafo_l_after,
        "totals_before": totals_before,
        "totals_after": totals_after,
        "elec_status": elec_status,
        "elec_p_set": elec_p_set,
        "elec_q_set": elec_q_set
    }

# -------------------------
# === Example usage ===
# -------------------------
if __name__ == "__main__":
    # Create network
    net = pn.create_cigre_network_mv(with_der=WITH_DER)
    print(net)  # brief network info

    # Ask user for electrolyzer bus if not configured
    if ELECTROLYZER_BUS is None:
        ELECTROLYZER_BUS = int(input("Enter electrolyzer bus index: "))

    # Add the electrolyzer bus to highlights
    if ELECTROLYZER_BUS not in HIGHLIGHT_BUSES:
        HIGHLIGHT_BUSES.append(ELECTROLYZER_BUS)

    # Ask user for nominal power (optional)
    # ELECTROLYZER_P_NOMINAL_MW = float(input("Electrolyzer nominal power (MW): "))

    # Read time series
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH) if os.path.exists(REACTIVE_PATH) else None
    gen_df = read_timeseries_csv(GEN_PATH) if os.path.exists(GEN_PATH) else None

    # Run controlled timeseries
    results = run_timeseries_with_local_control(
        active_path=ACTIVE_PATH,
        reactive_path=REACTIVE_PATH,
        gen_path=GEN_PATH,
        net=net,
        start_hour=START_HOUR,
        duration_hours=DURATION_HOURS,
        out_dir=OUT_DIR,
        electrolyzer_bus=ELECTROLYZER_BUS,
        p_nominal_mw=ELECTROLYZER_P_NOMINAL_MW,
        pf=ELECTROLYZER_PF,
        line_thresh=LINE_LOADING_THRESHOLD,
        v_thresh=VOLTAGE_THRESHOLD_PU,
        highlight_buses=HIGHLIGHT_BUSES,
        highlight_lines=HIGHLIGHT_LINES
    )

    print("Completed controlled time-series. Outputs in:", OUT_DIR)


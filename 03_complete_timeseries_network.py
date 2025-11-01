# 02_timeseries_powerflow_cigre.py
"""
Time-series power flow runner for the CIGRE MV test network (pandapower).
- Loads CSV time series for active consumption, reactive consumption, and generation.
- Maps CSV rows to pandapower buses (flexible matching).
- Runs power flow for a specified time window (start_hour, duration_hours).
- Collects and saves time-series results (bus voltages, line loading, losses, total gen/load).
- Plots results (voltage traces, line loading heatmap/time-series, totals).

Assumptions:
- CSV files: rows = nodes (node id or name), columns = t0, t1, ... (hourly values)
- Power values in CSV are in kW (convert to MW for pandapower).
- Generation CSV may have rows like 'solar', 'wind' or bus-specific entries.
- This script uses simple load/sgen assignment (overwrites p/q at each timestep).
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, List

# -------------------------
# === USER CONFIGURATION ===
# -------------------------
# Paths to your CSVs (put your real paths here):
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"


# Network creation option
WITH_DER = False  # if you want the network to already include DER elements in the pandapower template

# Time window to run (hours)
START_HOUR = 0        # starting column index (0-based). If your CSV columns are t0,t1,... then 0 means t0.
DURATION_HOURS = 120   # how many hours to run (e.g., 72 for 3 days). Ensure CSV has at least START_HOUR + DURATION_HOURS columns

# Output directory
OUT_DIR = "./timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Choose which buses to highlight plots for (list of bus indices or names). If empty, will pick a few automatically.
HIGHLIGHT_BUSES = []

# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    """Read a CSV timeseries where the first column is the node index/name (index_col=0)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, index_col=0)
    # Ensure columns are sorted as provided; user may have named them 't0','t1' or '0','1' etc.
    return df

def map_node_to_bus(node_label, net: pp.pandapowerNet) -> Optional[int]:
    """
    Try to map a node label from CSV to a pandapower bus index.
    - If node_label is numeric and equals a bus index -> return it.
    - If node_label matches net.bus.name -> return bus.index.
    - If node_label matches net.bus.index cast to string -> return index.
    - Otherwise return None.
    """
    # direct numeric index?
    try:
        maybe_idx = int(node_label)
        if maybe_idx in net.bus.index:
            return maybe_idx
    except Exception:
        pass

    # match against bus names (case-insensitive)
    if 'name' in net.bus.columns:
        names = net.bus['name'].astype(str).to_list()
        for idx, nm in zip(net.bus.index, names):
            if str(nm).strip().lower() == str(node_label).strip().lower():
                return idx

    # match against string representation of index
    for idx in net.bus.index:
        if str(idx) == str(node_label).strip():
            return idx

    return None

def assign_loads_from_profiles(net: pp.pandapowerNet,
                               active_row: pd.Series,
                               reactive_row: Optional[pd.Series],
                               bus_idx: int):
    """
    Assign active (kW) and reactive (kVar) values to pandapower loads at bus_idx.
    If there are existing loads at that bus, we overwrite their values proportionally.
    If no load exists, we create a new one.
    """
    # convert kW to MW
    p_mw = float(active_row) / 1.0
    q_mvar = 0.0
    if reactive_row is not None:
        q_mvar = float(reactive_row) / 1.0

    # find existing loads at this bus
    loads_at_bus = net.load[net.load.bus == bus_idx]
    if len(loads_at_bus) > 0:
        # distribute requested p and q proportionally to existing p_mw and q_mvar
        total_existing_p = loads_at_bus.p_mw.sum()
        if total_existing_p == 0:
            # simply assign equally
            for li in loads_at_bus.index:
                net.load.at[li, 'p_mw'] = p_mw / len(loads_at_bus)
                net.load.at[li, 'q_mvar'] = q_mvar / len(loads_at_bus)
        else:
            for li in loads_at_bus.index:
                share = net.load.at[li, 'p_mw'] / total_existing_p
                net.load.at[li, 'p_mw'] = p_mw * share
                # reactive proportionally as well
                if 'q_mvar' in net.load.columns:
                    net.load.at[li, 'q_mvar'] = q_mvar * share
    else:
        # create a new load element if none exists
        pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=q_mvar, name=f"ts_load_bus_{bus_idx}")

def assign_generation_profile(net: pp.pandapowerNet,
                              gen_label: str,
                              value_kw: float):
    """
    Simple policy:
    - If gen_label matches a bus (map_node_to_bus) -> assign/update sgen at that bus.
    - If gen_label is generic like 'solar' or 'wind', we attempt to find existing sgens by type name and scale them.
    - Otherwise create a new sgen on slack bus (not ideal) — we attempt best effort.
    """
    p_mw = float(value_kw) / 1.0

    # try bus mapping
    mapped = map_node_to_bus(gen_label, net)
    if mapped is not None:
        # find sgens on that bus
        sgens_at_bus = net.sgen[net.sgen.bus == mapped]
        if len(sgens_at_bus) > 0:
            # distribute proportionally
            total = sgens_at_bus.p_mw.sum()
            if total == 0:
                for si in sgens_at_bus.index:
                    net.sgen.at[si, 'p_mw'] = p_mw / len(sgens_at_bus)
            else:
                for si in sgens_at_bus.index:
                    share = net.sgen.at[si, 'p_mw'] / total if total > 0 else 1/len(sgens_at_bus)
                    net.sgen.at[si, 'p_mw'] = p_mw * share
        else:
            # create a new sgen
            pp.create_sgen(net, bus=mapped, p_mw=p_mw, q_mvar=0.0, name=f"ts_sgen_bus_{mapped}")
        return

    # if label like 'solar' or 'wind', map to sgens by name containing these keywords
    key = str(gen_label).strip().lower()
    if 'solar' in key or 'pv' in key:
        matches = [i for i, n in zip(net.sgen.index, net.sgen['name'].astype(str)) if 'pv' in n.lower() or 'solar' in n.lower()]
    elif 'wind' in key:
        matches = [i for i, n in zip(net.sgen.index, net.sgen['name'].astype(str)) if 'wind' in n.lower()]
    else:
        matches = []

    if matches:
        total = net.sgen.loc[matches, 'p_mw'].sum()
        if total == 0:
            for si in matches:
                net.sgen.at[si, 'p_mw'] = p_mw / len(matches)
        else:
            for si in matches:
                share = net.sgen.at[si, 'p_mw'] / total
                net.sgen.at[si, 'p_mw'] = p_mw * share
        return

    # fallback: create sgen at the first non-swing bus (or bus 0)
    fallback_bus = int(net.bus.index[0])
    pp.create_sgen(net, bus=fallback_bus, p_mw=p_mw, q_mvar=0.0, name=f"ts_sgen_fallback_{gen_label}")
    return

# -------------------------
# === Main routine ===
# -------------------------
def run_timeseries_powerflow(active_path: str,
                             reactive_path: Optional[str],
                             gen_path: Optional[str],
                             net: pp.pandapowerNet,
                             start_hour: int,
                             duration_hours: int,
                             out_dir: str,
                             highlight_buses: List = None):
    # Read CSVs
    active_df = read_timeseries_csv(active_path)
    reactive_df = read_timeseries_csv(reactive_path) if reactive_path is not None else None
    gen_df = read_timeseries_csv(gen_path) if gen_path is not None else None

    # Basic checks
    available_hours = active_df.shape[1]
    if start_hour + duration_hours > available_hours:
        raise ValueError(f"Requested range exceeds available data. available_hours={available_hours}, requested_end={start_hour+duration_hours}")

    # Prepare result holders
    bus_voltage_time = pd.DataFrame(index=range(duration_hours), columns=net.bus.index.astype(str))
    line_loading_time = pd.DataFrame(index=range(duration_hours), columns=net.line.index.astype(str))
    trafo_loading_time = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index.astype(str)) if len(net.trafo) > 0 else None
    total_gen = []
    total_load = []
    total_losses = []

    # Pre-map CSV nodes to buses for performance
    node_to_bus_map: Dict[str, Optional[int]] = {}
    for node in active_df.index:
        node_to_bus_map[str(node)] = map_node_to_bus(node, net)

    if gen_df is not None:
        gen_labels = list(gen_df.index)
    else:
        gen_labels = []

    # For progress info
    print(f"Starting time-series run: hours {start_hour} .. {start_hour + duration_hours - 1}")

    for t in range(duration_hours):
        col = active_df.columns[start_hour + t]
        # Reset dynamic elements? We will overwrite loads/sgen p/q
        # Assign loads
        for node in active_df.index:
            bus_idx = node_to_bus_map[str(node)]
            if bus_idx is None:
                # ignore nodes we can't map but print once
                if t == 0:
                    print(f"Warning: Active node '{node}' could not be mapped to any bus in the network. Skipping.")
                continue
            p_kw = active_df.at[node, col]
            q_kw = None
            if reactive_df is not None and node in reactive_df.index:
                q_kw = reactive_df.at[node, col]
            assign_loads_from_profiles(net, p_kw, q_kw, bus_idx)

        # Assign generation
        if gen_df is not None:
            for gl in gen_labels:
                try:
                    p_kw = gen_df.at[gl, col]
                except Exception:
                    # if gen rows are organized differently, ignore
                    continue
                assign_generation_profile(net, gl, p_kw)

        # Run power flow
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except pp.LoadflowNotConverged:
            print(f"Power flow did not converge at timestep {t} (col {col}). Saving NaNs and continuing.")
            # write NaNs for this step
            bus_voltage_time.loc[t, :] = np.nan
            line_loading_time.loc[t, :] = np.nan
            if trafo_loading_time is not None:
                trafo_loading_time.loc[t, :] = np.nan
            total_gen.append(np.nan)
            total_load.append(np.nan)
            total_losses.append(np.nan)
            continue

        # Collect results
        # Bus voltages (pu)
        for b in net.bus.index:
            bus_voltage_time.at[t, str(b)] = net.res_bus.vm_pu.at[b]

        # Line loading percent
        for li in net.line.index:
            line_loading_time.at[t, str(li)] = net.res_line.loading_percent.at[li]

        # Trafo loading percent
        if trafo_loading_time is not None:
            for tr in net.trafo.index:
                trafo_loading_time.at[t, str(tr)] = net.res_trafo.loading_percent.at[tr]

        # Totals: total generation (p_mw of gens + sgens), total load (sum loads), total losses
        gen_sum = 0.0
        if 'p_mw' in net.gen.columns:
            gen_sum += net.res_gen.p_mw.sum() if 'res_gen' in net and hasattr(net, 'res_gen') else 0.0
        if 'p_mw' in net.sgen.columns:
            gen_sum += net.res_sgen.p_mw.sum() if 'res_sgen' in net and hasattr(net, 'res_sgen') else 0.0
        # fallback check: sometimes res_sgen/res_gen exist but columns named differently; try reading net.sgen.p_mw
        try:
            if 'res_sgen' in net and 'p_mw' in net.res_sgen.columns:
                gen_sum = net.res_sgen.p_mw.sum()
        except Exception:
            pass

        load_sum = net.res_load.p_mw.sum() if 'res_load' in net and 'p_mw' in net.res_load.columns else 0.0
        # losses: sum of line losses + trafo losses if available
        losses = 0.0
        if 'pl_mw' in net.res_line.columns:
            losses += net.res_line.pl_mw.sum()
        if 'pl_mw' in getattr(net, 'res_trafo', pd.DataFrame()).columns:
            losses += net.res_trafo.pl_mw.sum()

        total_gen.append(gen_sum)
        total_load.append(load_sum)
        total_losses.append(losses)

    # Convert lists to Series/DataFrame
    totals_df = pd.DataFrame({
        'total_gen_mw': total_gen,
        'total_load_mw': total_load,
        'total_losses_mw': total_losses
    }, index=range(duration_hours))

    # Save results
    bus_voltage_time.to_csv(os.path.join(out_dir, 'bus_voltage_time.csv'), index_label='hour')
    line_loading_time.to_csv(os.path.join(out_dir, 'line_loading_time.csv'), index_label='hour')
    if trafo_loading_time is not None:
        trafo_loading_time.to_csv(os.path.join(out_dir, 'trafo_loading_time.csv'), index_label='hour')
    totals_df.to_csv(os.path.join(out_dir, 'totals_time.csv'), index_label='hour')

    print(f"Saved results to {out_dir}")

    # -------------------------
    # === Plotting section ===
    # -------------------------
    # 1) Plot voltage traces for highlight buses (or the 5 most critical buses by final voltage)
    # -------------------------
    # === Plotting section (Enhanced) ===
    # -------------------------
    import seaborn as sns

    # === 1) Bus voltages (existing, unchanged) ===
    if not highlight_buses:
        try:
            final_voltages = bus_voltage_time.iloc[-1].astype(float)
            worst = final_voltages.sort_values().index[:14].tolist()
            highlight_buses = [int(x) for x in worst]
        except Exception:
            highlight_buses = [net.bus.index[0]]

    highlight_buses = [map_node_to_bus(b, net) if not isinstance(b, int) else b for b in highlight_buses]
    highlight_buses = [hb for hb in highlight_buses if hb is not None]

    plt.figure(figsize=(12, 5))
    for b in highlight_buses:
        plt.plot(range(duration_hours), bus_voltage_time[str(b)].astype(float).values, marker='o', label=f"Bus {b}")
    plt.title("Bus Voltage (p.u.) over Time")
    plt.xlabel("Hour")
    plt.ylabel("Voltage (p.u.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "voltages_highlight.png"))
    plt.show()

    # === 2) 5 most loaded lines ===
    final_loading = line_loading_time.iloc[-1].astype(float)
    top5_lines = final_loading.sort_values(ascending=False).index[:10]

    plt.figure(figsize=(12, 5))
    for ln in top5_lines:
        plt.plot(range(duration_hours), line_loading_time[ln].astype(float).values, marker='o', label=f"Line {ln}")
    plt.title("Top 5 Most Loaded Lines over Time")
    plt.xlabel("Hour")
    plt.ylabel("Loading (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top5_line_loading.png"))
    plt.show()

    # === 3) User-selected lines ===
    HIGHLIGHT_LINES = [1, 3, 5]  # کاربر در اینجا شماره خطوط دلخواه را وارد کند
    valid_lines = [str(l) for l in HIGHLIGHT_LINES if str(l) in line_loading_time.columns]

    if valid_lines:
        plt.figure(figsize=(12, 5))
        for ln in valid_lines:
            plt.plot(range(duration_hours), line_loading_time[ln].astype(float).values, marker='o', label=f"Line {ln}")
        plt.title("Selected Line Loadings over Time")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "selected_line_loading.png"))
        plt.show()

    # === 4) Transformer loading ===
    if trafo_loading_time is not None and len(trafo_loading_time.columns) > 0:
        plt.figure(figsize=(12, 5))
        for tr in trafo_loading_time.columns:
            plt.plot(range(duration_hours), trafo_loading_time[tr].astype(float).values, marker='o', label=f"Trafo {tr}")
        plt.title("Transformer Loading (%) over Time")
        plt.xlabel("Hour")
        plt.ylabel("Loading (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trafo_loading.png"))
        plt.show()

    # === 5) Total generation, load, losses ===
    plt.figure(figsize=(10, 5))
    plt.plot(totals_df.index, totals_df['total_load_mw'], marker='o', label='Total Load (MW)')
    plt.plot(totals_df.index, totals_df['total_gen_mw'], marker='o', label='Total Generation (MW)')
    plt.plot(totals_df.index, totals_df['total_losses_mw'], marker='x', label='Total Losses (MW)')
    plt.title("System Totals over Time")
    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "system_totals.png"))
    plt.show()

    # === 6) Solar and Wind generation breakdown ===
    if gen_df is not None:
        solar_rows = [r for r in gen_df.index if 'solar' in r.lower() or 'pv' in r.lower()]
        wind_rows = [r for r in gen_df.index if 'wind' in r.lower()]

        solar_sum = gen_df.loc[solar_rows].sum() if solar_rows else pd.Series(0, index=gen_df.columns)
        wind_sum = gen_df.loc[wind_rows].sum() if wind_rows else pd.Series(0, index=gen_df.columns)

        plt.figure(figsize=(10, 5))
        plt.plot(solar_sum[start_hour:start_hour + duration_hours].values, label="Solar Generation (kW)")
        plt.plot(wind_sum[start_hour:start_hour + duration_hours].values, label="Wind Generation (kW)")
        plt.title("Solar and Wind Generation over Time")
        plt.xlabel("Hour")
        plt.ylabel("Power (kW)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "solar_wind_generation.png"))
        plt.show()


    return {
        'bus_voltage_time': bus_voltage_time,
        'line_loading_time': line_loading_time,
        'trafo_loading_time': trafo_loading_time,
        'totals_df': totals_df
    }

# -------------------------
# === Example usage ===
# -------------------------
if __name__ == "__main__":
    # Create network
    net = pn.create_cigre_network_mv(with_der=WITH_DER)

    # Optional: print basic info
    print(net)

    # Run timeseries
    results = run_timeseries_powerflow(
        active_path=ACTIVE_PATH,
        reactive_path=REACTIVE_PATH,
        gen_path=GEN_PATH,
        net=net,
        start_hour=START_HOUR,
        duration_hours=DURATION_HOURS,
        out_dir=OUT_DIR,
        highlight_buses=HIGHLIGHT_BUSES
    )

    print("Done. Results DataFrames keys:", results.keys())
    print("Files saved in:", OUT_DIR)




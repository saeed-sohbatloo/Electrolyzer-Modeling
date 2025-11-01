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
# Paths to CSV files with timeseries
ACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Active_Node_Consumption.csv"
GEN_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

OUT_DIR = "./timeseries_opf_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Electrolyzer parameters
ELECTROLYZER_BUS = 0
ELECTROLYZER_POWER_MW = 1.0       # Max active power [MW]
ELECTROLYZER_PF = 0.95            # Power factor
COST_PER_HOUR = np.array([100, 120, 150, 180, 160, 140, 120, 110,
                          100, 90, 80, 70, 60, 70, 80, 90,
                          100, 120, 150, 180, 160, 140, 120, 100])/10000  # €/MW

# Highlight elements for plotting
HIGHLIGHT_BUSES = [ELECTROLYZER_BUS]
HIGHLIGHT_LINES = [11, 14]

DURATION_HOURS = 24
START_HOUR = 0

# -------------------------
# === Helper functions ===
# -------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    """Read CSV timeseries data"""
    return pd.read_csv(path, index_col=0)

def map_node_to_bus(node_label, net: pp.pandapowerNet) -> Optional[int]:
    """Map node label from CSV to pandapower bus index"""
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
    """Assign MW and MVar to existing load(s) on a bus, or create new load"""
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
    """Assign MW to an existing generator or create a new sgen"""
    bus_idx = map_node_to_bus(gen_label, net)
    if bus_idx is None: bus_idx = net.bus.index[0]
    sgens = net.sgen[net.sgen.bus == bus_idx]
    if len(sgens) > 0:
        for si in sgens.index: net.sgen.at[si,'p_mw'] = value_mw/len(sgens)
    else:
        pp.create_sgen(net, bus=bus_idx, p_mw=value_mw, q_mvar=0.0)

# -------------------------
# === Timeseries OPF routine ===
# -------------------------
def run_timeseries_opf(net: pp.pandapowerNet,
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
    """
    Run timeseries OPF with controllable electrolyzer
    - Electrolyzer modeled as negative sgen (load)
    - Hourly variable cost is applied via poly_cost
    """

    # Compute reactive power of electrolyzer based on PF
    elec_q = electrolyzer_p_mw * np.tan(np.arccos(electrolyzer_pf))

    # Initialize results DataFrames
    bus_v = pd.DataFrame(index=range(duration_hours), columns=net.bus.index)
    line_l = pd.DataFrame(index=range(duration_hours), columns=net.line.index)
    trafo_l = pd.DataFrame(index=range(duration_hours), columns=net.trafo.index) if len(net.trafo)>0 else None
    electrolyzer_p = []

    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    # -----------------------------
    # Add controllable electrolyzer as negative sgen
    # -----------------------------
    if electrolyzer_bus is not None and electrolyzer_p_mw>0:
        sgen_idx = pp.create_sgen(net, bus=electrolyzer_bus, p_mw=0.0, q_mvar=0.0,
                                  min_p_mw=-electrolyzer_p_mw, max_p_mw=0.0,
                                  min_q_mvar=-elec_q, max_q_mvar=elec_q,
                                  controllable=True, name="Electrolyzer")
        # Create poly_cost placeholder; will update hourly
        pp.create_poly_cost(net, sgen_idx, et='sgen', cp0_eur=0.0, cp1_eur_per_mw=0.0, cp2_eur_per_mw2=0.0)

    # -----------------------------
    # Timeseries loop
    # -----------------------------
    for t in range(duration_hours):
        col = active_df.columns[start_hour+t]

        # Update electrolyzer cost for this hour
        net.poly_cost.at[sgen_idx, "cp1_eur_per_mw"] = COST_PER_HOUR[t]

        # Assign loads
        for node in active_df.index:
            bus_idx = node_to_bus[str(node)]
            if bus_idx is None: continue
            p_mw = active_df.at[node,col]
            q_mvar = reactive_df.at[node,col] if reactive_df is not None else 0.0
            assign_load(net, bus_idx, p_mw, q_mvar)

        # Assign generation
        if gen_df is not None:
            for gl in gen_df.index:
                assign_generation(net, gl, gen_df.at[gl,col])

        # Run OPF
        try:
            pp.runopp(net, calculate_voltage_angles=True)
        except Exception as e:
            print(f"Warning: OPF failed at timestep {t}: {e}")
            continue

        # Store results
        bus_v.loc[t,:] = net.res_bus.vm_pu
        line_l.loc[t,:] = net.res_line.loading_percent
        if trafo_l is not None: trafo_l.loc[t,:] = net.res_trafo.loading_percent
        electrolyzer_p.append(net.res_sgen.p_mw.at[sgen_idx])

        print(f"Hour {t}: Electrolyzer P = {-electrolyzer_p[-1]:.3f} MW")

    # -----------------------------
    # Save results
    # -----------------------------
    bus_v.to_csv(os.path.join(out_dir,"bus_voltage.csv"))
    line_l.to_csv(os.path.join(out_dir,"line_loading.csv"))
    if trafo_l is not None: trafo_l.to_csv(os.path.join(out_dir,"trafo_loading.csv"))
    pd.DataFrame({"electrolyzer_p_mw": electrolyzer_p}).to_csv(os.path.join(out_dir,"electrolyzer_p.csv"), index=False)

    # --- Plot Electrolyzer Power in kW ---
    electrolyzer_p_kw = np.array(electrolyzer_p) * -1000  # Convert MW -> kW
    hours = np.arange(duration_hours)

    plt.figure(figsize=(12, 4))
    plt.plot(hours, electrolyzer_p_kw, marker='o')
    plt.xlabel("Hour")
    plt.ylabel("Electrolyzer P [kW]")
    plt.title("Electrolyzer Active Power (kW)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "electrolyzer_power_kw.png"))
    plt.show()

    # Save kW values
    pd.DataFrame({"electrolyzer_p_kw": electrolyzer_p_kw}).to_csv(
        os.path.join(out_dir, "electrolyzer_p_kw.csv"), index=False
    )

    return bus_v, line_l, trafo_l, electrolyzer_p

# -------------------------
# === Main Execution ===
# -------------------------
if __name__=="__main__":
    # Load MV CIGRE network
    net = pn.create_cigre_network_mv()

    # Read timeseries CSVs
    active_df = read_timeseries_csv(ACTIVE_PATH)
    reactive_df = read_timeseries_csv(REACTIVE_PATH)
    gen_df = read_timeseries_csv(GEN_PATH)

    # Run timeseries OPF
    bus_v, line_l, trafo_l, electrolyzer_p = run_timeseries_opf(
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

    print("✅ Timeseries OPF with electrolyzer and hourly cost completed.")

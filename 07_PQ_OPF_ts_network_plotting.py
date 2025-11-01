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
GEN_PATH= "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Generation.csv"
REACTIVE_PATH = "/Users/mac/Downloads/Master Theises/MV-Cigre-github-test-data/Typical-load-profile-MV-CIGRE-benchmark-main/5-days test case/Scenario A/Reactive_Node_Consumption.csv"

OUT_DIR = "./timeseries_opf_results"
os.makedirs(OUT_DIR, exist_ok=True)

ELECTROLYZER_BUS = 8
ELECTROLYZER_POWER_MW = 1.0
ELECTROLYZER_PF = 0.95

HIGHLIGHT_BUSES = [ELECTROLYZER_BUS]
HIGHLIGHT_LINES = [11, 14]

DURATION_HOURS = 24
START_HOUR = 0

# €/MW for each hour
COST_PER_HOUR = np.array([100, 120, 150, 180, 160, 140, 120, 110,
                          100, 90, 80, 70, 60, 70, 80, 90,
                          100, 120, 150, 180, 160, 140, 120, 100])

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
# === Timeseries OPF routine ===
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
    electrolyzer_p = []
    electrolyzer_q = []
    electrolyzer_cost = []

    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    # Add controllable electrolyzer (as negative load)
    if electrolyzer_bus is not None and electrolyzer_p_mw>0:
        sgen_idx = pp.create_sgen(net, bus=electrolyzer_bus, p_mw=0.0, q_mvar=0.0,
                                  min_p_mw=-electrolyzer_p_mw, max_p_mw=0.0,
                                  min_q_mvar=-elec_q, max_q_mvar=elec_q,
                                  controllable=True, name="Electrolyzer")
        pp.create_poly_cost(net, sgen_idx, et='sgen', cp0_eur=0.0, cp1_eur_per_mw=cost_per_hour[0], cp2_eur_per_mw2=0.0)

    for t in range(duration_hours):
        col = active_df.columns[start_hour+t]
        # Update hourly cost dynamically
        if electrolyzer_bus is not None:
            net.poly_cost.at[sgen_idx, "cp1_eur_per_mw"] = cost_per_hour[t]/10000 # change to high Sensitivity

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
            electrolyzer_p.append(np.nan)
            electrolyzer_q.append(np.nan)
            electrolyzer_cost.append(np.nan)
            bus_v.loc[t,:] = np.nan
            line_l.loc[t,:] = np.nan
            if trafo_l is not None: trafo_l.loc[t,:] = np.nan
            continue

        # Store results
        bus_v.loc[t,:] = net.res_bus.vm_pu
        line_l.loc[t,:] = net.res_line.loading_percent
        if trafo_l is not None: trafo_l.loc[t,:] = net.res_trafo.loading_percent

        p_val = net.res_sgen.p_mw.at[sgen_idx] if electrolyzer_bus is not None else 0.0
        q_val = net.res_sgen.q_mvar.at[sgen_idx] if electrolyzer_bus is not None else 0.0
        electrolyzer_p.append(p_val)
        electrolyzer_q.append(q_val)
        electrolyzer_cost.append(-p_val * cost_per_hour[t])  # negative because load

        print(f"Hour {t}: Electrolyzer P = {-p_val:.3f} MW, Cost = {electrolyzer_cost[-1]:.2f} €")

    # Save CSVs
    bus_v.to_csv(os.path.join(out_dir,"bus_voltage.csv"))
    line_l.to_csv(os.path.join(out_dir,"line_loading.csv"))
    if trafo_l is not None: trafo_l.to_csv(os.path.join(out_dir,"trafo_loading.csv"))
    pd.DataFrame({"electrolyzer_p_mw": electrolyzer_p, "electrolyzer_q_mvar": electrolyzer_q,
                  "electrolyzer_cost_eur": electrolyzer_cost}).to_csv(os.path.join(out_dir,"electrolyzer_results.csv"), index=False)

    # ========================
    # === Plotting Section ===
    # ========================
    # ========================
    # === Plotting Section ===
    # ========================

    # Convert to float and handle NaNs
    bus_v_numeric = bus_v.astype(float).ffill()  # forward-fill missing voltages
    line_l_numeric = line_l.astype(float).fillna(0)  # assume 0% loading if NaN

    hours = np.arange(duration_hours)
    electrolyzer_p_kw = np.array(electrolyzer_p) * -1000  # MW -> kW

    # 1) Electrolyzer Active Power [kW]
    plt.figure(figsize=(12, 4))
    plt.plot(hours, electrolyzer_p_kw, marker='o', label='Electrolyzer P [kW]')
    plt.xlabel("Hour")
    plt.ylabel("Power [kW]")
    plt.title("Electrolyzer Active Power")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "electrolyzer_power_kw.png"))
    plt.show()

    # 2) Electrolyzer Reactive Power
    plt.figure(figsize=(12, 4))
    plt.plot(hours, electrolyzer_q, marker='s', label='Electrolyzer Q [MVar]', color='orange')
    plt.xlabel("Hour")
    plt.ylabel("Reactive Power [MVar]")
    plt.title("Electrolyzer Reactive Power")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "electrolyzer_q.png"))
    plt.show()

    # 3) Electrolyzer Cost vs Power
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.plot(hours, electrolyzer_p_kw, 'b-o', label='Electrolyzer P [kW]')
    ax2.plot(hours, electrolyzer_cost, 'r-s', label='Cost [€]')
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power [kW]", color='b')
    ax2.set_ylabel("Cost [€]", color='r')
    plt.title("Electrolyzer Power vs Cost")
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "electrolyzer_power_cost.png"))
    plt.show()

    # 4) Voltage Heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(bus_v_numeric.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Voltage [pu]')
    plt.xlabel("Hour")
    plt.ylabel("Bus Index")
    plt.title("Voltage Profile Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bus_voltage_heatmap.png"))
    plt.show()

    # 5) Line Loading Heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(line_l_numeric.T, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='Loading [%]')
    plt.xlabel("Hour")
    plt.ylabel("Line Index")
    plt.title("Line Loading Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_loading_heatmap.png"))
    plt.show()

    return bus_v, line_l, trafo_l, electrolyzer_p, electrolyzer_q, electrolyzer_cost

# -------------------------
# === Main Execution ===
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

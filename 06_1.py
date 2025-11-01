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


DURATION_HOURS = 120
ELECTROLYZER_P_NOMINAL_MW = 1.0
ELECTROLYZER_PF = 0.95
Q_MAX = 10.5
DQ_OVER = 10.06
DQ_UNDER = 10.04
V_DB = 0.01
V_OVER = 1.0 - V_DB
V_UNDER = 1.0 + V_DB

LINE_LOADING_THRESHOLD = 10.0
VOLTAGE_THRESHOLD_PU = 0.99

OUT_DIR = "./timeseries_results"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# USER-DEFINED SCENARIOS
# Each scenario is a dict specifying:
# - 'name': str
# - 'has_electrolyzer': bool
# - 'q_control': bool
# - 'onoff_control': bool
# -------------------------
'''SCENARIOS = [
    {'name': 'No Electrolyzer', 'has_electrolyzer': False, 'q_control': False, 'onoff_control': False},
    {'name': 'Electrolyzer, Q OFF', 'has_electrolyzer': True, 'q_control': False, 'onoff_control': False},
    {'name': 'Electrolyzer, Q Controlled', 'has_electrolyzer': True, 'q_control': False, 'onoff_control': True}
]
'''
SCENARIOS = [
    {'name': 'No Electrolyzer', 'has_electrolyzer': False, 'onoff_control': False},
    {'name': 'Electrolyzer with ON/OFF', 'has_electrolyzer': True, 'onoff_control': True},
    {'name': 'Electrolyzer always ON', 'has_electrolyzer': True, 'onoff_control': False}
]

# Ask user for bus index
ELECTROLYZER_BUS = int(input("Enter electrolyzer bus index: "))

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
# TIME-SERIES RUNNER WITH FLEXIBLE CONTROL
# -------------------------
def run_timeseries(active_df, reactive_df, gen_df, net, scenario):
    has_elec = scenario['has_electrolyzer']
    q_control = scenario['q_control']
    onoff_control = scenario['onoff_control']

    ELEC_NAME = 'Electrolyzer'
    if has_elec:
        ensure_electrolyzer_load(net, ELECTROLYZER_BUS, ELEC_NAME)

    node_to_bus = {str(node): map_node_to_bus(node, net) for node in active_df.index}

    hours = range(DURATION_HOURS)
    bus_v = pd.DataFrame(index=hours, columns=[str(b) for b in net.bus.index], dtype=float)
    line_l = pd.DataFrame(index=hours, columns=[str(l) for l in net.line.index], dtype=float)
    trafo_l = pd.DataFrame(index=hours, columns=[str(t) for t in net.trafo.index], dtype=float) if len(net.trafo)>0 else None

    elec_status = pd.Series(index=hours, dtype=object) if has_elec and onoff_control else pd.Series(index=hours, dtype=object).fillna('OFF')
    elec_p_set = pd.Series(index=hours, dtype=float) if has_elec else pd.Series(index=hours, dtype=float).fillna(0.0)
    elec_q_set = pd.Series(index=hours, dtype=float) if has_elec and q_control else pd.Series(index=hours, dtype=float).fillna(0.0)

    prev_p_set = 0.0
    prev_q_set = 0.0
    if has_elec and not onoff_control:
        prev_p_set = ELECTROLYZER_P_NOMINAL_MW

    for t in hours:
        col = active_df.columns[t]
        assign_base_loads(net, active_df, reactive_df, node_to_bus, col, ELEC_NAME)
        assign_generation_profiles(net, gen_df, col)

        if has_elec:
            set_electrolyzer_power(net, ELEC_NAME, prev_p_set, prev_q_set)

        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            bus_v.loc[t,:] = np.nan
            line_l.loc[t,:] = np.nan
            if trafo_l is not None: trafo_l.loc[t,:] = np.nan
            continue

        v_bus = float(net.res_bus.vm_pu.at[ELECTROLYZER_BUS]) if has_elec else None

        # Q control
        if has_elec and q_control and prev_p_set>0.0:
            q_prev = float(net.load.loc[net.load.name==ELEC_NAME,'q_mvar'].iloc[0])
            if v_bus < V_OVER: q_delta = (1-v_bus)*DQ_OVER
            elif v_bus > V_UNDER: q_delta = (1-v_bus)*DQ_UNDER
            else: q_delta = 0.0
            q_value = float(np.clip(q_prev+q_delta, -Q_MAX, Q_MAX))
        elif has_elec:
            q_value = 0.0
        else:
            q_value = 0.0

        if has_elec:
            set_electrolyzer_power(net, ELEC_NAME, prev_p_set, q_value)

        pp.runpp(net)

        # store results
        bus_v.loc[t,:] = net.res_bus.vm_pu.values
        line_l.loc[t,:] = net.res_line.loading_percent.values
        if trafo_l is not None: trafo_l.loc[t,:] = net.res_trafo.loading_percent.values

        elec_status[t] = 'ON' if prev_p_set>0 else 'OFF'
        elec_p_set[t] = prev_p_set
        elec_q_set[t] = q_value

    return bus_v, line_l, trafo_l, elec_status, elec_p_set, elec_q_set

# -------------------------
# MAIN EXECUTION
# -------------------------
net_template = pn.create_cigre_network_mv(with_der=False)
active_df = read_timeseries_csv(ACTIVE_PATH)
reactive_df = read_timeseries_csv(REACTIVE_PATH) if os.path.exists(REACTIVE_PATH) else None
gen_df = read_timeseries_csv(GEN_PATH) if os.path.exists(GEN_PATH) else None

results = {}
for sc in SCENARIOS:
    net_copy = copy.deepcopy(net_template)
    bus_v, line_l, trafo_l, elec_status, elec_p_set, elec_q_set = run_timeseries(active_df, reactive_df, gen_df, net_copy, sc)
    results[sc['name']] = {'bus_v': bus_v, 'line_l': line_l, 'trafo_l': trafo_l, 'elec_status': elec_status, 'elec_p_set': elec_p_set, 'elec_q_set': elec_q_set}

# -------------------------
# PLOTTING EXAMPLE
# -------------------------
hours = range(DURATION_HOURS)
plt.figure(figsize=(12,5))
for sc in SCENARIOS:
    plt.plot(hours, results[sc['name']]['bus_v'][str(ELECTROLYZER_BUS)], label=sc['name'])
plt.xlabel('Hour')
plt.ylabel('Voltage (p.u.)')
plt.title('Voltage at Electrolyzer Bus')
plt.grid(True)
plt.legend()
plt.show()

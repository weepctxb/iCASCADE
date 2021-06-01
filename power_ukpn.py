import pickle

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
import cvxpy as cp

from globalparams import POWER_COLORS
import util


# P.U.1.1 Load nodes and edges data
def load_power_ukpn_data():

    # P.U.1.1.1 Load node data
    try:
        gisnode = pd.read_pickle("data/power_ukpn/in/power_ukpn_gisnode.pkl")
    except IOError:
        gisnode = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                                sheet_name="GIS_node", header=0)
        gisnode.to_pickle("data/power_ukpn/in/power_ukpn_gisnode.pkl")

    # P.U.1.1.2 Load circuit data
    try:
        circuit = pd.read_pickle("data/power_ukpn/in/power_ukpn_circuit.pkl")
    except IOError:
        circuit = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 1 - Circuit Data", header=0)
        circuit.to_pickle("data/power_ukpn/in/power_ukpn_circuit.pkl")

    # P.U.1.1.3 Load transformer 2W data
    try:
        trans2w = pd.read_pickle("data/power_ukpn/in/power_ukpn_trans2w.pkl")
    except IOError:
        trans2w = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 2A - Transformer Data 2W", header=0)
        trans2w.to_pickle("data/power_ukpn/in/power_ukpn_trans2w.pkl")

    # P.U.1.1.4 Load transformer 3W data
    try:
        trans3w = pd.read_pickle("data/power_ukpn/in/power_ukpn_trans3w.pkl")
    except IOError:
        trans3w = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 2B - Transformer Data 3W", header=0)
        trans3w.to_pickle("data/power_ukpn/in/power_ukpn_trans3w.pkl")

    # P.U.1.1.5 Load load data
    try:
        loads = pd.read_pickle("data/power_ukpn/in/power_ukpn_load.pkl")
    except IOError:
        loads = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 3 - Load Data", header=0)
        loads.to_pickle("data/power_ukpn/in/power_ukpn_load.pkl")

    # P.U.1.1.6 Load generator data
    try:
        geners = pd.read_pickle("data/power_ukpn/in/power_ukpn_gen.pkl")
    except IOError:
        geners = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 5 - Generation", header=0)
        geners.to_pickle("data/power_ukpn/in/power_ukpn_gen.pkl")

    return gisnode, circuit, trans2w, trans3w, loads, geners


# P.U.1.2 Create networkx graph
def create_power_ukpn_graph(gisnode, circuit, trans2w, trans3w, loads, geners):

    # P.U.1.2.1 Create graph from edges dataframe - circuits

    # Perform clustering of nodes directly into substation names/locations
    circuit = circuit.groupby(["GSP", "From Substation", "To Substation"]). \
        agg({
        "From Node": lambda x: ",".join(np.unique(x)),
        "From x": lambda x: np.nanmean(x),
        "From y": lambda x: np.nanmean(x),
        "To Node": lambda x: ",".join(np.unique(x)),
        "To x": lambda x: np.nanmean(x),
        "To y": lambda x: np.nanmean(x),
        "Line Name": lambda x: ",".join(np.unique(x)),
        "Operating Voltage kV": lambda x: np.nanmean(x),
        "Positive Sequence Impedance [R] % on 100MVA base": lambda x: np.nanmean(x),
        "Positive Sequence Impedance [X] % on 100MVA base": lambda x: np.nanmean(x),
        "Susceptance [B] % on 100MVA base": lambda x: np.nanmean(x),
        "Rating Amps Summer": lambda x: np.nansum([0. if isinstance(_, str) else _ for _ in x]),
        "Rating Amps Winter": lambda x: np.nansum(x),
        "Circuit Length km": lambda x: np.nanmean([0. if isinstance(_, str) else _ for _ in x])
    }).reset_index()

    # Get list of GSPs
    GSPs = np.unique(circuit["GSP"].tolist())

    # Build graph
    G_c = nx.from_pandas_edgelist(circuit, source="From Substation", target="To Substation",
                                  edge_attr=["Line Name", "GSP",
                                             "From x", "From y",
                                             "To x", "To y",
                                             "Circuit Length km", "Operating Voltage kV",
                                             "Rating Amps Summer", "Rating Amps Winter"],
                                  create_using=nx.DiGraph)

    for u, v in G_c.edges():
        # Node attributes
        G_c.nodes[u]["pos"] = (G_c.edges[u, v]["From x"], G_c.edges[u, v]["From y"])
        G_c.nodes[v]["pos"] = (G_c.edges[u, v]["To x"], G_c.edges[u, v]["To y"])
        G_c.nodes[u]["type"] = "GSP" if u in GSPs else "substation"
        G_c.nodes[v]["type"] = "GSP" if v in GSPs else "substation"
        # Link flow capacity
        G_c.edges[u, v]["flow_cap"] = G_c.edges[u, v]["Operating Voltage kV"] / 1000 \
                                      * G_c.edges[u, v]["Rating Amps Winter"]

    G_c.remove_edges_from(nx.selfloop_edges(G_c))  # Remove self-loops

    # P.U.1.2.2 Create graph from edges dataframe - 2W transformers

    # Perform clustering of nodes directly into substation names/locations
    trans2w = trans2w.groupby(["GSP", "HV Substation", "LV Substation"]). \
        agg({
        "HV Node": lambda x: ",".join(np.unique(x)),
        "HV x": lambda x: np.nanmean(x),
        "HV y": lambda x: np.nanmean(x),
        "Voltage HV kV": lambda x: np.nanmean(x),
        "LV Node": lambda x: ",".join(np.unique(x)),
        "LV x": lambda x: np.nanmean(x),
        "LV y": lambda x: np.nanmean(x),
        "Voltage LV kV": lambda x: np.nanmean(x),
        "Positive Sequence Impedance [R] % on 100MVA base": lambda x: np.nanmean(x),
        "Positive Sequence Impedance [X] % on 100MVA base": lambda x: np.nanmean(x),
        "Zero Sequence Impedance [X] % on 100MVA base": lambda x: np.nanmean(x),
        "Tap Range Minimum %": lambda x: np.nanmean(x),
        "Tap Range Maximum %": lambda x: np.nanmean(x),
        "Transformer Rating Summer MVA": lambda x: np.nansum(x),
        "Transformer Rating Winter MVA": lambda x: np.nansum(x)
    }).reset_index()

    # Build graph
    G_t2 = nx.from_pandas_edgelist(trans2w, source="HV Substation", target="LV Substation",
                                   edge_attr=["GSP",
                                              "HV x", "HV y",
                                              "LV x", "LV y",
                                              "Voltage HV kV", "Voltage LV kV",
                                              "Transformer Rating Summer MVA", "Transformer Rating Winter MVA"],
                                   create_using=nx.DiGraph)

    for u, v in G_t2.edges():
        # Node attributes
        G_t2.nodes[u]["pos"] = (G_t2.edges[u, v]["HV x"], G_t2.edges[u, v]["HV y"])
        G_t2.nodes[v]["pos"] = (G_t2.edges[u, v]["LV x"], G_t2.edges[u, v]["LV y"])
        G_t2.nodes[u]["type"] = "GSP" if u in GSPs else "substation"
        G_t2.nodes[v]["type"] = "GSP" if v in GSPs else "substation"
        # Link flow capacity
        G_t2.edges[u, v]["flow_cap"] = G_t2.edges[u, v]["Transformer Rating Winter MVA"]  # or ["Transformer Rating Summer MVA"]

    G_t2.remove_edges_from(nx.selfloop_edges(G_t2))  # Remove self-loops

    # P.U.1.2.3 Create graph from edges dataframe - 3W transformers

    # Perform clustering of nodes directly into substation names/locations
    trans3w = trans3w.groupby(["GSP", "HV Substation", "LV1 Substation", "LV2 Substation"]). \
        agg({
        "HV Node": lambda x: ",".join(np.unique(x)),
        "HV x": lambda x: np.nanmean(x),
        "HV y": lambda x: np.nanmean(x),
        "Voltage HV kV": lambda x: np.nanmean(x),
        "LV Node 1": lambda x: ",".join(np.unique(x)),
        "LV1 x": lambda x: np.nanmean(x),
        "LV1 y": lambda x: np.nanmean(x),
        "Voltage LV1 kV": lambda x: np.nanmean(x),
        "LV Node 2": lambda x: ",".join(np.unique(x)),
        "LV2 x": lambda x: np.nanmean(x),
        "LV2 y": lambda x: np.nanmean(x),
        "Voltage LV2 kV": lambda x: np.nanmean(x),
        "Positive Sequence Impedance R % on 100MVA base (HV-LV1)": lambda x: np.nanmean(x),
        "Positive Sequence Impedance R % on 100MVA base (HV-LV2)": lambda x: np.nanmean(x),
        "Positive Sequence Impedance X % on 100MVA base (HV-LV1)": lambda x: np.nanmean(x),
        "Positive Sequence Impedance X % on 100MVA base (HV-LV2)": lambda x: np.nanmean(x),
        "Positive Sequence Impedance X % on 100MVA base (LV1-LV2)": lambda x: np.nanmean(x),
        "Zero Sequence Impedance X % on 100MVA base (HV-LV1)": lambda x: np.nanmean(x),
        "Zero Sequence Impedance X % on 100MVA base (HV-LV2)": lambda x: np.nanmean(x),
        "Zero Sequence Impedance X % on 100MVA base (LV1-LV2)": lambda x: np.nanmean(x),
        "Tap Range Minimum HV": lambda x: np.nanmean(x),
        "Tap Range Minimum LV1": lambda x: np.nanmean(x),
        "Tap Range Minimum LV2": lambda x: np.nanmean(x),
        "Tap Range Maximum HV": lambda x: np.nanmean(x),
        "Tap Range Maximum LV1": lambda x: np.nanmean(x),
        "Tap Range Maximum LV2": lambda x: np.nanmean(x),
        "Transformer Rating MVA Summer HV": lambda x: np.nansum(x),
        "Transformer Rating MVA Winter HV": lambda x: np.nansum(x),
        "Transformer Rating MVA Summer LV1": lambda x: np.nansum(x),
        "Transformer Rating MVA Winter LV1": lambda x: np.nansum(x),
        "Transformer Rating MVA Summer LV2": lambda x: np.nansum(x),
        "Transformer Rating MVA Winter LV2": lambda x: np.nansum(x)
    }).reset_index()

    # Build graph
    G_t3 = nx.from_pandas_edgelist(trans3w, source="HV Substation", target="LV1 Substation",
                                     edge_attr=["GSP",
                                                "HV x", "HV y",
                                                "LV1 x", "LV1 y",
                                                "Voltage HV kV", "Voltage LV1 kV",
                                                "Transformer Rating MVA Summer HV",
                                                "Transformer Rating MVA Winter HV",
                                                "Transformer Rating MVA Summer LV1",
                                                "Transformer Rating MVA Winter LV1"],
                                     create_using=nx.DiGraph)

    for u, v in G_t3.edges():
        # Node attributes
        G_t3.nodes[u]["pos"] = (G_t3.edges[u, v]["HV x"], G_t3.edges[u, v]["HV y"])
        G_t3.nodes[v]["pos"] = (G_t3.edges[u, v]["LV1 x"], G_t3.edges[u, v]["LV1 y"])
        G_t3.nodes[u]["type"] = "GSP" if u in GSPs else "substation"
        G_t3.nodes[v]["type"] = "GSP" if v in GSPs else "substation"
        # Link flow capacity
        G_t3.edges[u, v]["flow_cap"] = G_t3.edges[u, v]["Transformer Rating MVA Winter LV1"]  # or ["Transformer Rating MVA Summer LV1"]

    G_t3_2 = nx.from_pandas_edgelist(trans3w, source="HV Substation", target="LV2 Substation",
                                     edge_attr=["GSP",
                                                "HV x", "HV y",
                                                "LV2 x", "LV2 y",
                                                "Voltage HV kV", "Voltage LV2 kV",
                                                "Transformer Rating MVA Summer HV",
                                                "Transformer Rating MVA Winter HV",
                                                "Transformer Rating MVA Summer LV2",
                                                "Transformer Rating MVA Winter LV2"],
                                     create_using=nx.DiGraph)
    
    for u, v in G_t3_2.edges():
        if (u, v) in G_t3.edges():  # If both LV sides of 3W transformer lead to same substation
            # Add flow capacity
            G_t3.edges[u, v]["flow_cap"] += G_t3_2.edges[u, v]["Transformer Rating MVA Winter LV2"]
        else:  # If LV sides of 3W transformer lead to different substations
            # Add new edge manually
            G_t3.add_edge(u, v)
            G_t3.edges[u, v]["GSP"] = G_t3_2.edges[u, v]["GSP"]
            G_t3.edges[u, v]["HV x"] = G_t3_2.edges[u, v]["HV x"]
            G_t3.edges[u, v]["HV y"] = G_t3_2.edges[u, v]["HV y"]
            G_t3.edges[u, v]["LV2 x"] = G_t3_2.edges[u, v]["LV2 x"]
            G_t3.edges[u, v]["LV2 y"] = G_t3_2.edges[u, v]["LV2 y"]
            G_t3.edges[u, v]["Voltage HV kV"] = G_t3_2.edges[u, v]["Voltage HV kV"]
            G_t3.edges[u, v]["Voltage LV2 kV"] = G_t3_2.edges[u, v]["Voltage LV2 kV"]
            G_t3.edges[u, v]["Transformer Rating MVA Summer HV"] = G_t3_2.edges[u, v]["Transformer Rating MVA Summer HV"]
            G_t3.edges[u, v]["Transformer Rating MVA Winter HV"] = G_t3_2.edges[u, v]["Transformer Rating MVA Winter HV"]
            G_t3.edges[u, v]["Transformer Rating MVA Summer LV2"] = G_t3_2.edges[u, v]["Transformer Rating MVA Summer LV2"]
            G_t3.edges[u, v]["Transformer Rating MVA Winter LV2"] = G_t3_2.edges[u, v]["Transformer Rating MVA Winter LV2"]
            # Only need to add in info for LV2 side
            G_t3.nodes[v]["pos"] = (G_t3_2.edges[u, v]["LV2 x"], G_t3_2.edges[u, v]["LV2 y"])
            G_t3.nodes[v]["type"] = "GSP" if v in GSPs else "substation"
            # Link flow capacity
            G_t3.edges[u, v]["flow_cap"] = G_t3_2.edges[u, v]["Transformer Rating MVA Winter LV2"]  # or ["Transformer Rating MVA Summer LV2"]

    G_t3.remove_edges_from(nx.selfloop_edges(G_t3))  # Remove self-loops

    G = nx.compose(G_c, nx.compose(G_t2, G_t3))
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops

    # P.U.1.2.4 Create graph from edges dataframe - loads
    # Summer & Winter
    loads_summer = loads.loc[loads["Season"] == "Summer"]
    loads_winter = loads.loc[loads["Season"] == "Winter"]
    G_nodes_list = list(G.nodes())  # To avoid nodes list changing size during iteration
    for arg1, arg2 in zip(loads_summer.iterrows(), loads_winter.iterrows()):
        row_s = arg1[1]
        row_w = arg2[1]
        for n in G_nodes_list:  # To avoid nodes list changing size during iteration
            if n == row_s["Substation"]:
                load_name = str(row_s["Substation"]) + " Load"
                # FYI This new node is the load with the full substation name, not the node ID!
                G.add_node(load_name,
                           pos=(row_s["x"], row_s["y"]),
                           type="load",
                           Maximum_Demand_1920_MW_summer=row_s["Maximum Demand 19/20 MW"],
                           Maximum_Demand_1920_PF_summer=row_s["Maximum Demand 19/20 PF"],
                           Firm_Capacity_MW_summer=row_s["Firm Capacity MW"],
                           Minimum_Load_Scaling_Factor_summer=row_s["Minimum Load Scaling Factor %"],
                           Maximum_Demand_1920_MW_winter=row_w["Maximum Demand 19/20 MW"],
                           Maximum_Demand_1920_PF_winter=row_w["Maximum Demand 19/20 PF"],
                           Firm_Capacity_MW_winter=row_w["Firm Capacity MW"],
                           Minimum_Load_Scaling_Factor_winter=row_w["Minimum Load Scaling Factor %"])
                G.add_edge(n, load_name, flow_cap=row_w["Firm Capacity MW"])

    # P.U.1.2.5 Create graph from edges dataframe - generators
    geners = geners.loc[geners["Connected / Accepted"] == "Connected"]  # only consider connected generators
    geners = geners.loc[geners["Installed Capacity MW"] >= 1.]  # only consider generators above 1 MW
    for index, row in geners.iterrows():
        for n in G_nodes_list:  # To avoid nodes list changing size during iteration
            if n == row["Substation"]:
                gener_name = str(row["Substation"]) + " Gen"
                # FYI This new node is the generator with the full substation name, not the node ID!
                G.add_node(gener_name,
                           pos=(row["x"], row["y"]),
                           type="generator",
                           Connection_Voltage=row["Connection Voltage kV"],
                           Installed_Capacity=row["Installed Capacity MW"])
                G.add_edge(gener_name, n, flow_cap=row["Installed Capacity MW"])

    # P.U.1.2.6 Insert additional GSP->HV connections missing from dataset
    for _, row in circuit.iterrows():
        u = row["GSP"]
        v = row["From Substation"]
        if not nx.has_path(G, u, v):
            print("Added edge", u, "-", v)
            G.add_edge(u, v)
            G.edges[u, v]["Line Name"] = u + "-" + v
            G.edges[u, v]["GSP"] = u
            G.edges[u, v]["From x"] = G.nodes[u]["pos"][0]
            G.edges[u, v]["From y"] = G.nodes[u]["pos"][1]
            G.edges[u, v]["To x"] = G.nodes[v]["pos"][0]
            G.edges[u, v]["To y"] = G.nodes[v]["pos"][1]
            G.edges[u, v]["Circuit Length km"] = util.haversine(G.nodes[u]["pos"][0], G.nodes[u]["pos"][1],
                                                                G.nodes[v]["pos"][0], G.nodes[v]["pos"][1])
            G.edges[u, v]["Operating Voltage kV"] = row["Operating Voltage kV"]
            G.edges[u, v]["Rating Amps Summer"] = row["Rating Amps Summer"]
            G.edges[u, v]["Rating Amps Winter"] = row["Rating Amps Winter"]
            G.edges[u, v]["flow_cap"] = G.edges[u, v]["Operating Voltage kV"] / 1000 * G.edges[u, v]["Rating Amps Winter"]

    # P.U.1.2.7 Assign node thruflow capacities
    for n in G.nodes():
        predecessors = list(G.predecessors(n))
        successors = list(G.successors(n))
        G.nodes[n]["thruflow_cap"] = max(sum([G.edges[p, n]["flow_cap"] for p in predecessors]),
                                         sum([G.edges[n, s]["flow_cap"] for s in successors]))

    # P.U.1.2.8 Calculate and assign centralities
    bb = dict()
    cc = dict()

    for subG in util.weakly_connected_component_subgraphs(G, copy=True):
        bb.update(nx.current_flow_betweenness_centrality(subG.to_undirected(), normalized=True, weight="flow_cap"))
        cc.update(nx.current_flow_closeness_centrality(subG.to_undirected(), weight="flow_cap"))

    nx.set_node_attributes(G, bb, 'betweenness')
    nx.set_node_attributes(G, cc, 'closeness')

    return G


# P.U.2 Calculate baseline flows
def flowcalc_power_ukpn_graph(G):

    # Duplicate subgraphs in network (generalised)
    for subG in util.weakly_connected_component_subgraphs(G, copy=True):
        constraints = list()

        # P.U.2.1 Initialise flows
        # Add link flows as variables
        for u, v in subG.edges():
            subG.edges[u, v]["flow"] = cp.Variable(nonneg=True, name="flow_{" + str(u) + "," + str(v) + "}")
        # Add node flows (in/gen/con/out) and thruflow as variables
        for n in subG.nodes():
            subG.nodes[n]["flow_out"] = 0.
            subG.nodes[n]["flow_in"] = cp.Variable(nonneg=True, name="flow_in_{" + str(n) + "}") \
                if subG.nodes[n]["type"] == "GSP" else 0.
            subG.nodes[n]["flow_gen"] = cp.Variable(nonneg=True, name="flow_gen_{" + str(n) + "}") \
                if subG.nodes[n]["type"] == "generator" else 0.
            subG.nodes[n]["flow_con"] = subG.nodes[n]["Maximum_Demand_1920_MW_winter"] \
                if subG.nodes[n]["type"] == "load" else 0.

        # DEBUG - For troubleshooting
        if any([subG.nodes[n]["type"] == "GSP" for n in subG.nodes()]):
            print("GSP found: " + str([n for n in subG.nodes() if subG.nodes[n]["type"] == "GSP"]))
        else:
            print("No GSP found for this subgrid!")

        # P.U.2.2 Add constraints
        for n in subG.nodes():
            predecessors = list(subG.predecessors(n))
            successors = list(subG.successors(n))
            # Node flow balance
            constraints.append(sum([subG.edges[p, n]["flow"] for p in predecessors])
                               + subG.nodes[n]["flow_in"] + subG.nodes[n]["flow_gen"]
                               ==
                               sum([subG.edges[n, s]["flow"] for s in successors])
                               + subG.nodes[n]["flow_out"] + subG.nodes[n]["flow_con"])
        # Overall subgraph flow balance
        constraints.append(sum([subG.nodes[n]["flow_in"] + subG.nodes[n]["flow_gen"] for n in subG.nodes()])
                           ==
                           sum([subG.nodes[n]["flow_out"] + subG.nodes[n]["flow_con"] for n in subG.nodes()]))

        # DEBUG - For troubleshooting unbalanced nodes
        for constr in constraints:
            if len(constr.expr.grad) >= 1:
                for g in constr.expr.grad:
                    if "Back Hill" in g._name:
                        print(constr)

        # P.U.2.3 Solve optimisation problem to minimise line capacities exceeding
        objective = cp.Minimize(sum([subG.edges[u, v]["flow"] - subG.edges[u, v]["flow_cap"] for u, v in subG.edges()]))
        flow_balance_model = cp.Problem(objective, constraints)
        flow_balance_model.solve(solver=cp.OSQP, verbose=True, eps_abs=1.e-2, eps_rel=1.e-2)

        # DEBUG - For troubleshooting if constraints are satisfied
        if flow_balance_model.status == cp.OPTIMAL:
            for n in subG.nodes():
                predecessors = list(subG.predecessors(n))
                successors = list(subG.successors(n))
                # Flow balance
                lhs = sum([subG.edges[p, n]["flow"].value for p in predecessors]) \
                      + (subG.nodes[n]["flow_in"].value if isinstance(subG.nodes[n]["flow_in"], cp.Variable) else
                         subG.nodes[n][
                             "flow_in"]) \
                      + (subG.nodes[n]["flow_gen"].value if isinstance(subG.nodes[n]["flow_gen"], cp.Variable) else
                         subG.nodes[n]["flow_gen"])
                rhs = sum([subG.edges[n, s]["flow"].value for s in successors]) \
                      + (subG.nodes[n]["flow_out"].value if isinstance(subG.nodes[n]["flow_out"], cp.Variable) else
                         subG.nodes[n]["flow_out"]) \
                      + (subG.nodes[n]["flow_con"].value if isinstance(subG.nodes[n]["flow_con"], cp.Variable) else
                         subG.nodes[n]["flow_con"])
                try:
                    if abs(rhs - lhs) / max(lhs, rhs) > 0.1 and (rhs > 0.1 or lhs > 0.1):
                        print(
                            "Unbalanced node: " + n + "(" + G.nodes[n]["type"] + ") : in=" + str(lhs) + ", out=" + str(
                                rhs))
                except RuntimeWarning as e:
                    print(n, lhs, rhs, e)

            subGlhs = sum([(subG.nodes[n]["flow_in"].value if isinstance(subG.nodes[n]["flow_in"], cp.Variable) else
                            subG.nodes[n][
                                "flow_in"])
                           + (subG.nodes[n]["flow_gen"].value if isinstance(subG.nodes[n]["flow_gen"], cp.Variable) else
                              subG.nodes[n]["flow_gen"])
                           for n in subG.nodes()])
            subGrhs = sum([(subG.nodes[n]["flow_out"].value if isinstance(subG.nodes[n]["flow_out"], cp.Variable) else
                            subG.nodes[n][
                                "flow_out"])
                           + (subG.nodes[n]["flow_con"].value if isinstance(subG.nodes[n]["flow_con"], cp.Variable) else
                              subG.nodes[n]["flow_con"])
                           for n in subG.nodes()])
            print("Subgrid overall: in=", subGlhs, ", out=", subGrhs)

        # P.U.2.4 Assign flows back to original graph
        for u, v in subG.edges():
            G.edges[u, v]["flow"] = subG.edges[u, v]["flow"].value \
                if isinstance(subG.edges[u, v]["flow"], cp.Variable) \
                else subG.edges[u, v]["flow"] if isinstance(subG.edges[u, v]["flow"], (float, int)) else 0.
            G.edges[u, v]["pct_flow_cap"] = G.edges[u, v]["flow"] / G.edges[u, v]["flow_cap"]
        for n in subG.nodes():
            predecessors = list(G.predecessors(n))
            successors = list(G.successors(n))
            G.nodes[n]["thruflow"] = max(sum([G.edges[p, n]["flow"] for p in predecessors]),
                                         sum([G.edges[n, s]["flow"] for s in successors]))
            G.nodes[n]["pct_thruflow_cap"] = G.nodes[n]["thruflow"] / G.nodes[n]["thruflow_cap"]

    return G


# P.U.5 Flow check
def flow_check(G):
    for u, v in G.edges():
        if (G.edges[u, v]["flow"] > G.edges[u, v]["flow_cap"]) \
                or (G.edges[u, v]["flow"] == 0) or (G.edges[u, v]["flow_cap"] == 0):
            print("Link Warning: ", u, v, G.edges[u, v]["flow"], G.edges[u, v]["flow_cap"])
    for n in G.nodes():
        if (G.nodes[n]["thruflow"] > G.nodes[n]["thruflow_cap"]) \
                or (G.nodes[n]["thruflow"] == 0) or (G.nodes[n]["thruflow_cap"] == 0):
            print("Node Warning: ", n, G.nodes[n]["thruflow"], G.nodes[n]["thruflow_cap"])


if __name__ == "__main__":

    # P.U.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/power_ukpn/out/power_ukpn_G.pkl', "rb"))
    except IOError:
        gisnode, circuit, trans2w, trans3w, loads, geners = load_power_ukpn_data()
        G = create_power_ukpn_graph(gisnode, circuit, trans2w, trans3w, loads, geners)
        try:
            pickle.dump(G, open(r'data/power_ukpn/out/power_ukpn_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # P.U.2 Calculate baseline flows
    try:
        G_flow = pickle.load(open(r'data/power_ukpn/out/power_ukpn_G_flow.pkl', "rb"))
    except IOError:
        G_flow = flowcalc_power_ukpn_graph(G)
        try:
            pickle.dump(G_flow, open(r'data/power_ukpn/out/power_ukpn_G_flow.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # P.U.3 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G_flow.nodes(data=True)],
                                         index=[i[0] for i in G_flow.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/power_ukpn/out/power_ukpn_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # P.U.4 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G_flow, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/power_ukpn/out/power_ukpn_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # P.U.5 Export graph edgelist
    try:
        edgelist = pd.DataFrame(columns=["From", "To", "flow", "flow_cap", "pct_flow_cap"])
        for u, v in G_flow.edges():
            series_obj = pd.Series([u, v,
                                    G_flow.edges[u, v]["flow"], G_flow.edges[u, v]["flow_cap"],
                                    G_flow.edges[u, v]["pct_flow_cap"]],
                                   index=edgelist.columns)
            edgelist = edgelist.append(series_obj, ignore_index=True)
        edgelist.to_excel(r'data/power_ukpn/out/power_ukpn_edgelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # P.U.5 Flow check
    flow_check(G_flow)

    # P.U.6 Add colours and export map plot
    node_colors = [POWER_COLORS.get(G_flow.nodes[node]["type"], "#000000") for node in G_flow.nodes()]
    edge_colors = [POWER_COLORS.get("cable", "#000000") for u, v in G_flow.edges()]  # FYI assumes underground cable

    nx.draw(G_flow, pos=nx.get_node_attributes(G_flow, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/power_ukpn/img/power_UKPN.png")
    plt.savefig("data/power_ukpn/img/power_UKPN.svg")
    plt.show()

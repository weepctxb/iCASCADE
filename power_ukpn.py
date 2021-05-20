import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import json
import numpy as np


# P.U.1.1 Load nodes and edges data
from globalparams import POWER_COLORS


def load_power_ukpn_data():

    # P.U.1.1.1 Load node data
    # TODO To test
    try:
        gisnode = pd.read_pickle("data/power_ukpn/in/power_ukpn_gisnode.pkl")
    except IOError:
        gisnode = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                                sheet_name="GIS_node", header=0)
        gisnode.to_pickle("data/power_ukpn/in/power_ukpn_gisnode.pkl")

    # P.U.1.1.2 Load circuit data
    # TODO To test
    try:
        circuit = pd.read_pickle("data/power_ukpn/in/power_ukpn_circuit.pkl")
    except IOError:
        circuit = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 1 - Circuit Data", header=0)
        circuit.to_pickle("data/power_ukpn/in/power_ukpn_circuit.pkl")

    # P.U.1.1.3 Load transformer 2W data
    # TODO To test
    try:
        trans2w = pd.read_pickle("data/power_ukpn/in/power_ukpn_trans2w.pkl")
    except IOError:
        trans2w = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 2A - Transformer Data 2W", header=0)
        trans2w.to_pickle("data/power_ukpn/in/power_ukpn_trans2w.pkl")

    # P.U.1.1.4 Load transformer 3W data
    # TODO To test
    try:
        trans3w = pd.read_pickle("data/power_ukpn/in/power_ukpn_trans3w.pkl")
    except IOError:
        trans3w = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 2B - Transformer Data 3W", header=0)
        trans3w.to_pickle("data/power_ukpn/in/power_ukpn_trans3w.pkl")

    # P.U.1.1.5 Load load data
    # TODO To test
    try:
        loads = pd.read_pickle("data/power_ukpn/in/power_ukpn_load.pkl")
    except IOError:
        loads = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 3 - Load Data", header=0)
        loads.to_pickle("data/power_ukpn/in/power_ukpn_load.pkl")

    # P.U.1.1.6 Load generator data
    # TODO To test
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
    # TODO To test
    G_c = nx.from_pandas_edgelist(circuit, source="From Node", target="To Node",
                                edge_attr=["Line Name", "GSP",
                                           "From Substation", "From x", "From y",
                                           "To Substation", "To x", "To y",
                                           "Circuit Length km", "Operating Voltage kV",
                                           "Rating Amps Summer", "Rating Amps Winter"])
    for u, v in G_c.edges():
        G_c.nodes[u]["Location"] = G_c.edges[u, v]["From Substation"]
        G_c.nodes[v]["Location"] = G_c.edges[u, v]["To Substation"]
        G_c.nodes[u]["pos"] = (G_c.edges[u, v]["From y"], G_c.edges[u, v]["From x"])
        G_c.nodes[v]["pos"] = (G_c.edges[u, v]["To y"], G_c.edges[u, v]["To x"])

    # P.U.1.2.2 Create graph from edges dataframe - 2W transformers
    # TODO To test
    G_t2 = nx.from_pandas_edgelist(trans2w, source="HV Node", target="LV Node",
                                edge_attr=["GSP",
                                           "HV Substation", "HV x", "HV y",
                                           "LV Substation", "LV x", "LV y",
                                           "Voltage HV kV", "Voltage LV kV",
                                           "Transformer Rating Summer MVA", "Transformer Rating Winter MVA"])
    for u, v in G_t2.edges():
        G_t2.nodes[u]["Location"] = G_t2.edges[u, v]["HV Substation"]
        G_t2.nodes[v]["Location"] = G_t2.edges[u, v]["LV Substation"]
        G_t2.nodes[u]["pos"] = (G_t2.edges[u, v]["HV y"], G_t2.edges[u, v]["HV x"])
        G_t2.nodes[v]["pos"] = (G_t2.edges[u, v]["LV y"], G_t2.edges[u, v]["LV x"])
        G_t2.nodes[u]["Transformer Rating Summer MVA"] = G_t2.edges[u, v]["Transformer Rating Summer MVA"]
        G_t2.nodes[v]["Transformer Rating Summer MVA"] = G_t2.edges[u, v]["Transformer Rating Summer MVA"]
        G_t2.nodes[u]["Transformer Rating Winter MVA"] = G_t2.edges[u, v]["Transformer Rating Winter MVA"]
        G_t2.nodes[v]["Transformer Rating Winter MVA"] = G_t2.edges[u, v]["Transformer Rating Winter MVA"]

    # P.U.1.2.3 Create graph from edges dataframe - 3W transformers
    # TODO To test
    # HV -> LV1
    G_t3_1 = nx.from_pandas_edgelist(trans3w, source="HV Node", target="LV Node 1",
                                   edge_attr=["GSP",
                                              "HV Substation", "HV x", "HV y",
                                              "LV1 Substation", "LV1 x", "LV1 y",
                                              "Voltage HV kV", "Voltage LV1 kV",
                                              "Transformer Rating MVA Summer HV",
                                              "Transformer Rating MVA Winter HV",
                                              "Transformer Rating MVA Summer LV1",
                                              "Transformer Rating MVA Winter LV1"])
    for u, v in G_t3_1.edges():
        G_t3_1.nodes[u]["Location"] = G_t3_1.edges[u, v]["HV Substation"]
        G_t3_1.nodes[v]["Location"] = G_t3_1.edges[u, v]["LV1 Substation"]
        G_t3_1.nodes[u]["pos"] = (G_t3_1.edges[u, v]["HV y"], G_t3_1.edges[u, v]["HV x"])
        G_t3_1.nodes[v]["pos"] = (G_t3_1.edges[u, v]["LV1 y"], G_t3_1.edges[u, v]["LV1 x"])
        G_t3_1.nodes[u]["Transformer Rating Summer MVA"] = G_t3_1.edges[u, v]["Transformer Rating MVA Summer HV"]
        G_t3_1.nodes[v]["Transformer Rating Summer MVA"] = G_t3_1.edges[u, v]["Transformer Rating MVA Summer LV1"]
        G_t3_1.nodes[u]["Transformer Rating Winter MVA"] = G_t3_1.edges[u, v]["Transformer Rating MVA Summer HV"]
        G_t3_1.nodes[v]["Transformer Rating Winter MVA"] = G_t3_1.edges[u, v]["Transformer Rating MVA Winter LV1"]

    # HV -> LV2
    G_t3_2 = nx.from_pandas_edgelist(trans3w, source="HV Node", target="LV Node 2",
                                     edge_attr=["GSP",
                                                "HV Substation", "HV x", "HV y",
                                                "LV2 Substation", "LV2 x", "LV2 y",
                                                "Voltage HV kV", "Voltage LV2 kV",
                                                "Transformer Rating MVA Summer HV",
                                                "Transformer Rating MVA Winter HV",
                                                "Transformer Rating MVA Summer LV2",
                                                "Transformer Rating MVA Winter LV2"])
    for u, v in G_t3_2.edges():
        G_t3_2.nodes[u]["Location"] = G_t3_2.edges[u, v]["HV Substation"]
        G_t3_2.nodes[v]["Location"] = G_t3_2.edges[u, v]["LV2 Substation"]
        G_t3_2.nodes[u]["pos"] = (G_t3_2.edges[u, v]["HV y"], G_t3_2.edges[u, v]["HV x"])
        G_t3_2.nodes[v]["pos"] = (G_t3_2.edges[u, v]["LV2 y"], G_t3_2.edges[u, v]["LV2 x"])
        G_t3_2.nodes[u]["Transformer Rating Summer MVA"] = G_t3_2.edges[u, v]["Transformer Rating MVA Summer HV"]
        G_t3_2.nodes[v]["Transformer Rating Summer MVA"] = G_t3_2.edges[u, v]["Transformer Rating MVA Summer LV2"]
        G_t3_2.nodes[u]["Transformer Rating Winter MVA"] = G_t3_2.edges[u, v]["Transformer Rating MVA Summer HV"]
        G_t3_2.nodes[v]["Transformer Rating Winter MVA"] = G_t3_2.edges[u, v]["Transformer Rating MVA Winter LV2"]

    G = nx.compose(G_c, nx.compose(G_t2, nx.compose(G_t3_1, G_t3_2)))

    # P.U.1.2.4 Create graph from edges dataframe - loads
    # TODO To test
    # Summer
    loads_summer = loads.loc[loads["Season"] == "Summer"]
    for index, row in loads_summer.iterrows():
        for n in G.nodes():
            if G.nodes[n]["Location"] == row["Substation"]:
                load_name = str(row["Substation"]) + " Load"
                # FYI This new node is the load with the full substation name, not the node ID!
                G.add_node(load_name,
                           pos=(row["y"], row["x"]),
                           Maximum_Demand_1920_MW=row["Maximum Demand 19/20 MW"],
                           Maximum_Demand_1920_PF=row["Maximum Demand 19/20 PF"],
                           Firm_Capacity_MW=row["Firm Capacity MW"],
                           Minimum_Load_Scaling_Factor=row["Minimum Load Scaling Factor %"])
                G.add_edge(n, load_name)
    # Winter
    loads_winter = loads.loc[loads["Season"] == "Winter"]
    for index, row in loads_winter.iterrows():
        for n in G.nodes():
            if G.nodes[n]["Location"] == row["Substation"]:
                load_name = str(row["Substation"]) + " Load"
                # FYI This new node is the load with the full substation name, not the node ID!
                G.add_node(load_name,
                           pos=(row["y"], row["x"]),
                           Maximum_Demand_1920_MW=row["Maximum Demand 19/20 MW"],
                           Maximum_Demand_1920_PF=row["Maximum Demand 19/20 PF"],
                           Firm_Capacity_MW=row["Firm Capacity MW"],
                           Minimum_Load_Scaling_Factor=row["Minimum Load Scaling Factor %"])
                G.add_edge(n, load_name)  # Electricity flows *TO* load

    # TODO how to resolve Substations to Nodes? Or cluster Nodes into Substations at the beginning?

    # P.U.1.2.5 Create graph from edges dataframe - generators
    # TODO To test
    geners = geners.loc[geners["Connected / Accepted"] == "Connected"]  # only consider connected generators
    geners = geners.loc[geners["Installed Capacity MW"] >= 1.]  # only consider generators above 1 MW
    for index, row in geners.iterrows():
        for n in G.nodes():
            if G.nodes[n]["Location"] == row["Substation"]:
                gener_name = str(row["Substation"]) + " Gen"
                # FYI This new node is the generator with the full substation name, not the node ID!
                G.add_node(gener_name,
                           pos=(row["y"], row["x"]),
                           Connection_Voltage=row["Connection Voltage kV"],
                           Installed_Capacity=row["Installed Capacity MW"])
                G.add_edge(gener_name, n)  # Electricity flows *FROM* generator

    # TODO Assign flows
    #  But test the above code first (plot the network, make sure everything is correct first)

    return G


if __name__ == "__main__":
    # TODO To Test

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

    # P.U.2 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/power_ukpn/out/power_ukpn_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # P.U.3 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/power_ukpn/out/power_ukpn_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # P.U.4 Add colours and export map plot
    node_colors = [POWER_COLORS.get("substation", "#000000") for node in G.nodes()]
    edge_colors = [POWER_COLORS.get("cable", "#000000") for u, v in G.edges()]  # FYI assumes underground cable

    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/power_osm/img/power_UKPN.png")
    plt.savefig("data/power_osm/img/power_UKPN.svg")
    plt.show()

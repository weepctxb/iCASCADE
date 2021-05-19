import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import json
import numpy as np


# P.U.1.1 Load nodes and edges data
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
        gener = pd.read_pickle("data/power_ukpn/in/power_ukpn_gen.pkl")
    except IOError:
        gener = pd.read_excel("data/power_ukpn/raw/power_ukpn.xlsx",
                              sheet_name="Table 5 - Generation", header=0)
        gener.to_pickle("data/power_ukpn/in/power_ukpn_gen.pkl")

    return gisnode, circuit, trans2w, trans3w, loads, gener


# P.U.1.2 Create networkx graph
def create_power_ukpn_graph(gisnode, circuit, trans2w, trans3w, loads, gener):

    # P.U.1.2.1 Create graph from edges dataframe - circuits & transformers
    # TODO To test
    G_c = nx.from_pandas_edgelist(circuit, source="From Node", target="To Node",
                                edge_attr=["Line Name", "GSP",
                                           "From Substation", "From x", "From y",
                                           "To Substation", "To x", "To y",
                                           "Circuit Length km", "Operating Voltage kV",
                                           "Rating Amps Summer", "Rating Amps Winter"])

    G_t2 = nx.from_pandas_edgelist(trans2w, source="HV Node", target="LV Node",
                                edge_attr=["GSP",
                                           "HV Substation", "HV x", "HV y",
                                           "LV Substation", "LV x", "LV y",
                                           "Voltage HV kV", "Voltage LV kV",
                                           "Transformer Rating Summer MVA", "Transformer Rating Winter MVA"])

    G_t3 = nx.from_pandas_edgelist(trans3w, source="HV Node", target="LV Node 1",
                                   edge_attr=["GSP",
                                              "HV Substation", "HV x", "HV y",
                                              "LV1 Substation", "LV1 x", "LV1 y",
                                              "LV2 Substation", "LV2 x", "LV2 y",
                                              "Voltage HV kV", "Voltage LV1 kV", "Voltage LV2 kV",
                                              "Transformer Rating MVA Summer HV", "Transformer Rating MVA Winter HV",
                                              "Transformer Rating MVA Summer LV1", "Transformer Rating MVA Winter LV1",
                                              "Transformer Rating MVA Summer LV2", "Transformer Rating MVA Winter LV2",
                                              ""])

    G = nx.compose(G_c, nx.compose(G_t2, G_t3))

    # TODO P.U.1.2.2 Create graph from edges dataframe - loads
    loads = loads.loc[loads["Season"] == "Summer"]
    # TODO continue
    # TODO how to resolve Substations to Nodes? Or cluster Nodes into Substations at the beginning?

    # TODO P.U.1.2.3 Create graph from edges dataframe - generators
    gener = gener.loc[gener["Connected / Accepted"] == "Connected"]  # only consider connected generators
    gener = gener.loc[gener["Installed Capacity MW"] >= 1.]  # only consider generators above 1 MW
    # TODO continue

    # P.U.1.2.4 Create and assign node attributes
    pos = dict()
    for index, row in gisnode.iterrows():
        pos[row["Node"]] = (row["y"], row["x"])

    nx.set_node_attributes(G, pos, 'pos')

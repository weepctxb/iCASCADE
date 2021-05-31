import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import json
import numpy as np

from globalparams import TRANSPORT_COLORS, LOAD_CAP


# T.M.1.1 Load nodes and edges data
def load_transport_multiplex_data():

    # T.M.1.1.1 Load nodes
    try:
        nodes = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_nodes.pkl")
    except IOError:
        nodes = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                              sheet_name="london_transport_nodes", header=0)
        nodes.to_pickle("data/transport_multiplex/in/transport_multiplex_nodes.pkl")

    # T.M.1.1.2 Load edges
    try:
        edges = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_edges.pkl")
    except IOError:
        edges = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                              sheet_name="london_transport_raw", header=0)
        edges.to_pickle("data/transport_multiplex/in/transport_multiplex_edges.pkl")

    # T.M.1.1.3 Load O-D matrix
    try:
        odmat = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_odmat.pkl")
    except IOError:
        odmat = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                              sheet_name="OD_matrix", header=0)
        odmat.to_pickle("data/transport_multiplex/in/transport_multiplex_odmat.pkl")

    # T.M.1.1.4 Load capacity matrix
    try:
        capac = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_capac.pkl")
    except IOError:
        capac = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                              sheet_name="capacity", header=0)
        capac.to_pickle("data/transport_multiplex/in/transport_multiplex_capac.pkl")

    return nodes, edges, odmat, capac


# T.M.1.2 Create networkx graph
def create_transport_multiplex_graph(nodes, edges):

    # T.M.1.2.1 Create graph from edges dataframe
    G = nx.from_pandas_edgelist(edges, source="StationA_ID", target="StationB_ID",
                                edge_attr=["Line", "StationA", "StationB",
                                           "Distance", "running_time_min"])

    # T.M.1.2.2 Create and assign node attributes
    pos = dict()
    nodeLabel = dict()
    interchange = dict()
    line = dict()
    NLC = dict()
    for index, row in nodes.iterrows():
        pos[row["nodeID"]] = (row["nodeLong"], row["nodeLat"])
        nodeLabel[row["nodeID"]] = row["nodeLabel"]
        interchange[row["nodeID"]] = row["interchange"]
        line[row["nodeID"]] = row["line"]
        NLC[row["nodeID"]] = row["NLC"]

    nx.set_node_attributes(G, pos, 'pos')
    nx.set_node_attributes(G, nodeLabel, 'nodeLabel')
    nx.set_node_attributes(G, interchange, 'interchange')
    nx.set_node_attributes(G, line, 'line')
    nx.set_node_attributes(G, NLC, 'NLC')

    nx.set_node_attributes(G, "station", 'railway')

    # T.M.1.2.3 Calculate and assign centralities
    dd = nx.degree_centrality(G)
    bb = nx.betweenness_centrality(G, normalized=True, weight=True)
    cc = nx.closeness_centrality(G, distance="running_time_min")  # previously "Distance"
    ee = nx.eigenvector_centrality(G, max_iter=1000)

    nx.set_node_attributes(G, dd, 'degree')
    nx.set_node_attributes(G, bb, 'betweenness')
    nx.set_node_attributes(G, cc, 'closeness')
    nx.set_node_attributes(G, ee, 'eigenvector')

    nodes['degree'] = None
    nodes['betweenness'] = None
    nodes['closeness'] = None
    nodes['eigenvector'] = None
    for node in dd:
        nodes.at[node, 'degree'] = dd[node]
    for node in bb:
        nodes.at[node, 'betweenness'] = bb[node]
    for node in cc:
        nodes.at[node, 'closeness'] = cc[node]
    for node in ee:
        nodes.at[node, 'eigenvector'] = ee[node]

    # T.M.1.2.4 Calculate node sizes
    node_sizes = [int(bb[node] * 1000) for node in bb]

    return G, pos, node_sizes


# T.M.2.1 Calculate flows for networkx graph
def flowcalc_transport_multiplex_graph(G, odmat, capac):

    # T.M.2.1.1 Calculate shortest paths
    try:
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'r') as json_file:
            shortest_paths = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        shortest_paths = dict()
        for u in G.nodes():
            if u not in shortest_paths:
                shortest_paths[u] = dict()
            for v in G.nodes():
                if u != v:
                    if v not in shortest_paths[u]:
                        shortest_paths[u][v] = dict()
                    try:
                        shortest_paths[u][v]["path"] = nx.dijkstra_path(G, u, v,
                                                                        weight='running_time_min')  # 'distance'
                        # shortest_paths[u][v]["path_names"] = [G.nodes[n]["nodeLabel"] for n in
                        #                                       shortest_paths[u][v]["path"]]
                        shortest_paths[u][v]["travel_time"] = nx.dijkstra_path_length(G, u, v,
                                                                                      weight='running_time_min')
                        shortest_paths[u][v]["length"] = len(shortest_paths[u][v]["path"])
                    except nx.NodeNotFound:
                        shortest_paths[u][v]["path"] = None
                        # shortest_paths[u][v]["path_names"] = None
                        shortest_paths[u][v]["travel_time"] = -1
                        shortest_paths[u][v]["length"] = -1
                    except nx.NetworkXNoPath:
                        shortest_paths[u][v]["path"] = None
                        # shortest_paths[u][v]["path_names"] = None
                        shortest_paths[u][v]["travel_time"] = np.inf
                        shortest_paths[u][v]["length"] = np.inf
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'w') as json_file:
            json.dump(shortest_paths, json_file)  # indent=2, cls=util.CustomEncoder

    # T.M.2.1.2 Assign baseline flows

    # Initialise flows
    for u, v in G.edges():
        G.edges[u, v]["flow"] = 0

    # Assign flows
    for u in shortest_paths:
        # FIXME patch - keys in json are in strings for some reason
        unlc = G.nodes[int(u)]["NLC"]
        for v in shortest_paths[u]:
            vnlc = G.nodes[int(v)]["NLC"]
            odmat_filtered = odmat.loc[odmat["mnlc_o"] == unlc]
            odmat_filtered = odmat_filtered.loc[odmat["mnlc_d"] == vnlc]
            flow_uv = sum(odmat_filtered["od_tb_3_perhour"])
            for n1, n2 in zip(shortest_paths[u][v]["path"][:-1], shortest_paths[u][v]["path"][1:]):
                G.edges[n1, n2]["flow"] += flow_uv
                # print(n1, n2, flow_uv)  # DEBUG

    # T.M.2.1.3 Assign total node through flows
    # TODO To test
    for n in G.nodes():
        # Initialise through flows
        G.nodes[n]["thruflow"] = 0
        # Calculate through flows
        for ne in G.neighbors(n):
            G.nodes[n]["thruflow"] += G.edges[n, ne]["flow"]
        # Initialise inflows
        unlc = G.nodes[int(n)]["NLC"]
        odmat_filtered = odmat.loc[odmat["mnlc_o"] == unlc]
        G.nodes[n]["flow_in"] = sum(odmat_filtered["od_tb_3_perhour"])
        # Initialise outflows
        vnlc = G.nodes[int(n)]["NLC"]
        odmat_filtered = odmat.loc[odmat["mnlc_d"] == vnlc]
        G.nodes[n]["flow_out"] = sum(odmat_filtered["od_tb_3_perhour"])

    # T.M.2.1.4 Assign flow capacity (and % capacity utilised)
    # TODO To test
    for u, v in G.edges():
        unlc = G.nodes[u]["NLC"]
        vnlc = G.nodes[v]["NLC"]
        capac_filtered_1 = pd.concat([capac.loc[capac["StationA_NLC"] == unlc],
                                    capac.loc[capac["StationA_NLC"] == vnlc]], join="outer")
        capac_filtered = pd.concat([capac_filtered_1.loc[capac_filtered_1["StationB_NLC"] == vnlc],
                                    capac_filtered_1.loc[capac_filtered_1["StationB_NLC"] == unlc]], join="outer")
        flow_cap_uv = np.dot(capac_filtered["train_capacity"],
                             capac_filtered["train_freq_tb_3_perhour"])
        G.edges[u, v]["flow_cap"] = flow_cap_uv
        G.edges[u, v]["pct_flow_cap"] = G.edges[u, v]["flow"] / G.edges[u, v]["flow_cap"]
        # print(u, v, flow_cap_uv)  # DEBUG
        # Previously: G.edges[u, v]["flow_cap"] = LOAD_CAP * G.edges[u, v]["flow"]

    # T.M.2.1.5 Assign node capacity (and % capacity utilised)
    # TODO To test
    for n in G.nodes():
        # Initialise capacities
        G.nodes[n]["thruflow_cap"] = 0
        # Calculate capacities as sum of flow capacities of neighbouring links
        for ne in G.neighbors(n):
            G.nodes[n]["thruflow_cap"] += G.edges[n, ne]["flow_cap"]
    for n in G.nodes():
        G.nodes[n]["pct_thruflow_cap"] = G.nodes[n]["thruflow"] / G.nodes[n]["thruflow_cap"]

    return G


def flow_check(G):
    for u, v in G.edges():
        if (G.edges[u, v]["flow"] > G.edges[u, v]["flow_cap"]) \
                or (G.edges[u, v]["flow"] == 0) or (G.edges[u, v]["flow_cap"] == 0):
            print("Link Warning: ", u, v, G.edges[u, v]["flow"], G.edges[u, v]["flow_cap"])
    for n in G.nodes():
        if (G.nodes[n]["thruflow"] > G.nodes[n]["thruflow_cap"]) \
                or (G.nodes[n]["thruflow"] == 0) or (G.nodes[n]["thruflow_cap"] == 0):
            print("Node Warning: ", n, G.nodes[n]["thruflow"], G.nodes[n]["thruflow_cap"])


def plot_degree_histogram(G):

    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.loglog(degrees[:], degree_freq[:], 'go-')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

    weighted_degrees = []
    for n in G.nodes():
        dist = sum(edges.loc[edges["StationA_ID"] == n]["Distance"]) + sum(
            edges.loc[edges["StationB_ID"] == n]["Distance"])
        weighted_degrees.append(round(dist * 1000))
    n, bins, patches = plt.hist(weighted_degrees, bins=100)
    plt.clf()
    plt.loglog(bins[:-1], n[:], ".")
    plt.show()


if __name__ == "__main__":

    # T.M.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', "rb"))
    except IOError:
        nodes, edges, _, _ = load_transport_multiplex_data()
        G, pos, node_sizes = create_transport_multiplex_graph(nodes, edges)
        try:
            pickle.dump(G, open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.2 Calculate baseline flows and flow capacities
    try:
        G_flow = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', "rb"))
    except IOError:
        _, _, odmat, capac = load_transport_multiplex_data()
        G_flow = flowcalc_transport_multiplex_graph(G, odmat, capac)
        try:
            pickle.dump(G_flow, open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.3 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # T.M.4 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/transport_multiplex/out/transport_multiplex_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # T.M.5 Export graph edgelist
    try:
        edgelist = pd.DataFrame(columns=["StationA_ID", "StationA", "StationB_ID", "StationB", "flow", "flow_cap",
                                         "pct_flow_cap"])
        for u, v in G.edges():
            series_obj = pd.Series([u, G.edges[u, v]["StationA"], v, G.edges[u, v]["StationB"],
                                    G.edges[u, v]["flow"], G.edges[u, v]["flow_cap"],
                                    G.edges[u, v]["pct_flow_cap"]],
                                   index=edgelist.columns)
            edgelist = edgelist.append(series_obj, ignore_index=True)
        edgelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_edgelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # T.M.6 Check flows
    flow_check(G)

    # T.M.7 Add colours and export map plot
    node_colors = [TRANSPORT_COLORS.get(G.nodes[node]["line"], "#FF0000") for node in G.nodes()]
    edge_lines = nx.get_edge_attributes(G, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G.edges()]

    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/transport_multiplex/img/transport_multiplex.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex.svg")
    plt.show()

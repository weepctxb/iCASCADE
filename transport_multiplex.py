import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import json
import numpy as np

from globalparams import TRANSPORT_COLORS, LOAD_CAP


# T.M.1.1 Load nodes and edges data
from util import linear_cluster


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
    # PATCH Ignore interchanges with walk links first
    edges = edges.loc[edges["Line"] != "walk-link"]

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
def create_transport_multiplex_graph(nodes, edges, odmat, capac):

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
    # dd = nx.degree_centrality(G)
    bb = nx.betweenness_centrality(G, normalized=True, weight="running_time_min")
    cc = nx.closeness_centrality(G, distance="running_time_min")  # previously "Distance"
    # ee = nx.eigenvector_centrality(G, max_iter=1000)

    # nx.set_node_attributes(G, dd, 'degree')
    nx.set_node_attributes(G, bb, 'betweenness')
    nx.set_node_attributes(G, cc, 'closeness')
    # nx.set_node_attributes(G, ee, 'eigenvector')

    # nodes['degree'] = None
    # nodes['betweenness'] = None
    # nodes['closeness'] = None
    # nodes['eigenvector'] = None
    # for node in dd:
    #     nodes.at[node, 'degree'] = dd[node]
    for node in bb:
        nodes.at[node, 'betweenness'] = bb[node]
    for node in cc:
        nodes.at[node, 'closeness'] = cc[node]
    # for node in ee:
    #     nodes.at[node, 'eigenvector'] = ee[node]

    # T.M.1.2.4 Calculate node sizes
    node_sizes = [int(bb[node] * 1000) for node in bb]

    # T.M.1.2.x Assign node inflows and outflpws
    for n in G.nodes():
        # FIXME patch - keys in json are in strings for some reason (JSON encoder issue)
        nnlc = G.nodes[n]["NLC"]
        odmat_filtered_in = odmat.loc[odmat["mnlc_o"] == nnlc]
        G.nodes[n]["flow_in"] = sum(odmat_filtered_in["od_tb_3_perhour"])
        odmat_filtered_out = odmat.loc[odmat["mnlc_d"] == nnlc]
        G.nodes[n]["flow_out"] = sum(odmat_filtered_out["od_tb_3_perhour"])

    # T.M.1.2.5 Assign link flow capacities
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

    # T.M.1.2.6 Assign node thruflow capacities
    for n in G.nodes():
        # Initialise capacities
        G.nodes[n]["thruflow_cap"] = 0
        # Calculate capacities as sum of flow capacities of neighbouring links
        for ne in G.neighbors(n):
            G.nodes[n]["thruflow_cap"] += G.edges[n, ne]["flow_cap"]

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

    # T.M.2.1.2 Assign baseline link flows
    for u, v in G.edges():
        G.edges[u, v]["flow"] = 0
    for u in shortest_paths:
        # FIXME patch - keys in json are in strings for some reason (JSON encoder issue)
        unlc = G.nodes[int(u)]["NLC"]
        for v in shortest_paths[u]:
            vnlc = G.nodes[int(v)]["NLC"]
            odmat_filtered = odmat.loc[odmat["mnlc_o"] == unlc]
            odmat_filtered = odmat_filtered.loc[odmat["mnlc_d"] == vnlc]
            flow_uv = sum(odmat_filtered["od_tb_3_perhour"])
            for n1, n2 in zip(shortest_paths[u][v]["path"][:-1], shortest_paths[u][v]["path"][1:]):
                G.edges[n1, n2]["flow"] += flow_uv
                # print(n1, n2, flow_uv)  # DEBUG

    # T.M.2.1.3 Assign baseline node thruflows
    for n in G.nodes():
        # Initialise through flows
        G.nodes[n]["thruflow"] = 0
        # Calculate through flows
        for ne in G.neighbors(n):
            G.nodes[n]["thruflow"] += G.edges[n, ne]["flow"]

    # T.M.2.1.4 Assign baseline link flow % capacity utilised
    for u, v in G.edges():
        G.edges[u, v]["pct_flow_cap"] = G.edges[u, v]["flow"] / G.edges[u, v]["flow_cap"]

    # T.M.2.1.5 Assign baseline node thruflow % capacity utilised
    for n in G.nodes():
        G.nodes[n]["pct_thruflow_cap"] = G.nodes[n]["thruflow"] / G.nodes[n]["thruflow_cap"]

    return G


def flowcalc_transport_multiplex_graph_capscal(G):
    G1 = G.to_directed()
    for n in G1.nodes():
        G1.nodes[n]["flow_in"] = round(G1.nodes[n]["flow_in"])
        G1.nodes[n]["flow_out"] = round(G1.nodes[n]["flow_out"])
    total_supply = sum([G1.nodes[n]["flow_in"] for n in G1.nodes()])
    total_demand = sum([G1.nodes[n]["flow_out"] for n in G1.nodes()])
    for n in G1.nodes():
        G1.nodes[n]["demand"] = round(G1.nodes[n]["flow_out"] - G1.nodes[n]["flow_in"] * total_demand / total_supply)
    G1.nodes[n]["demand"] -= sum(G1.nodes[u].get("demand", 0) for u in G1)  # PATCH FIXME
    for u, v in G1.edges():
        G1.edges[u, v]["capacity"] = round(G1.edges[u, v]["flow_cap"])
        G1.edges[u, v]["weight"] = round(G1.edges[u, v]["Distance"] * 1000)
    flowCost, flowDict = nx.capacity_scaling(G1, demand='demand', capacity='capacity', weight='weight')
    for u, v in G1.edges():
        G.edges[u, v]["flow_capscal"] = flowDict[u][v] + flowDict[v][u]
    return G


# T.M.x Simplify network
def simplify_transport_multiplex_graph(G):
    # Current rule: Cluster 3 of:
    # - adjacent & consecutive stations with degree 2; and excluding
    # - line terminals
    # into a single hypernode
    adj_nonitc_path_list = list()
    for u, v in G.edges():
        if (G.degree[u] == 2) and (G.degree[v] == 2):
            u_nb = [n for n in G.neighbors(u) if n != v and G.degree[n] == 2]
            v_nb = [n for n in G.neighbors(v) if n != u and G.degree[n] == 2]
            if len([u, v] + v_nb) == 3:
                adj_nonitc_path_list.extend([[u, v] + v_nb])
            elif len(u_nb + [u, v]) == 3:
                adj_nonitc_path_list.extend([u_nb + [u, v]])
    for tri in adj_nonitc_path_list:
        try:
            G = linear_cluster(G, tri)
        except nx.NetworkXError:  # because nodes may have already been clustered from previous iteration
            continue
    return G


# T.M.x Check flows
def flow_check(G):
    for u, v in G.edges():
        if (G.edges[u, v]["flow"] > G.edges[u, v]["flow_cap"]) \
                or (G.edges[u, v]["flow"] == 0) or (G.edges[u, v]["flow_cap"] == 0):
            print("Link Warning: ", u, v, G.edges[u, v]["flow"], G.edges[u, v]["flow_cap"])
    for n in G.nodes():
        if (G.nodes[n]["thruflow"] > G.nodes[n]["thruflow_cap"]) \
                or (G.nodes[n]["thruflow"] == 0) or (G.nodes[n]["thruflow_cap"] == 0):
            print("Node Warning: ", n, G.nodes[n]["thruflow"], G.nodes[n]["thruflow_cap"])


# T.M.x Graph stats summary
def stats_summary(G):
    print("Stats summary:")
    print("sigma:", nx.sigma(G, niter=1, nrand=5), "(small world if >1)")
    print("omega:", nx.omega(G, niter=1, nrand=5), "(lattice if ~-1, small world if ~0, random if ~1)")

# def plot_degree_histogram(G):
#
#     degree_freq = nx.degree_histogram(G)
#     degrees = range(len(degree_freq))
#     plt.loglog(degrees[:], degree_freq[:], 'go-')
#     plt.xlabel('Degree')
#     plt.ylabel('Frequency')
#     plt.show()
#
#     weighted_degrees = []
#     for n in G.nodes():
#         dist = sum(edges.loc[edges["StationA_ID"] == n]["Distance"]) + sum(
#             edges.loc[edges["StationB_ID"] == n]["Distance"])
#         weighted_degrees.append(round(dist * 1000))
#     n, bins, patches = plt.hist(weighted_degrees, bins=100)
#     plt.clf()
#     plt.loglog(bins[:-1], n[:], ".")
#     plt.show()


if __name__ == "__main__":

    # T.M.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', "rb"))
    except IOError:
        nodes, edges, odmat, capac = load_transport_multiplex_data()
        G, pos, node_sizes = create_transport_multiplex_graph(nodes, edges, odmat, capac)
        try:
            pickle.dump(G, open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.2 Calculate baseline flows
    try:
        G_flow = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', "rb"))
    except IOError:
        _, _, odmat, capac = load_transport_multiplex_data()
        G_flow = flowcalc_transport_multiplex_graph(G, odmat, capac)
        G_flow = flowcalc_transport_multiplex_graph_capscal(G_flow)  # DEBUG alternative flow calc
        try:
            pickle.dump(G_flow, open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.x Simplify network
    try:
        G_skele = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', "rb"))
    except IOError:
        G_skele = simplify_transport_multiplex_graph(G_flow)
        G_skele = simplify_transport_multiplex_graph(G_skele)  # Run it again!
        # Beyond this the graph can no longer be further simplified using the current rules
        try:
            pickle.dump(G_skele, open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.x Check flows
    flow_check(G_skele)

    # T.M.x Graph stats summary
    # stats_summary(G)
    # stats_summary(G_skele)

    # T.M.3 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G_skele.nodes(data=True)], index=[i[0] for i in G_skele.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # T.M.4 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G_skele, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/transport_multiplex/out/transport_multiplex_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # T.M.5 Export graph edgelist
    try:
        edgelist = pd.DataFrame(columns=["StationA_ID", "StationA", "StationA_NLC",
                                         "StationB_ID", "StationB", "StationB_NLC",
                                         "flow", "flow_capscal", "flow_cap", "pct_flow_cap"])
        for u, v in G_skele.edges():
            series_obj = pd.Series([u, G_skele.nodes[u]["nodeLabel"], G_skele.nodes[u]["NLC"],
                                    v, G_skele.nodes[v]["nodeLabel"], G_skele.nodes[v]["NLC"],
                                    G_skele.edges[u, v]["flow"], G_skele.edges[u, v]["flow_capscal"],
                                    G_skele.edges[u, v]["flow_cap"], G_skele.edges[u, v]["pct_flow_cap"]],
                                   index=edgelist.columns)
            edgelist = edgelist.append(series_obj, ignore_index=True)
        edgelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_edgelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # T.M.7 Add colours and export map plot
    node_colors = [TRANSPORT_COLORS.get(G_skele.nodes[node]["line"], "#FF0000") for node in G_skele.nodes()]
    edge_lines = nx.get_edge_attributes(G_skele, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G_skele.edges()]

    nx.draw(G_skele, pos=nx.get_node_attributes(G_skele, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/transport_multiplex/img/transport_multiplex.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex.svg")
    plt.show()

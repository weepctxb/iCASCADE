import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import json
import numpy as np

from globalparams import TRANSPORT_COLORS, TRAIN_DUR_THRESHOLD_MIN
from util import linear_cluster, transport_calc_centrality, transport_to_undirected, populate_shortest_paths, \
    populate_shortest_paths_skele


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
    G = nx.DiGraph()  # Create directional graph first
    for index, row in edges.iterrows():
        G.add_edge(row["StationA_ID"], row["StationB_ID"],
                   Line=row["Line"], StationA=row["StationA"], StationB=row["StationB"],
                   Distance=row["Distance"], running_time_min=row["running_time_min"],
                   recip_running_time_min=1. / max(1., row["running_time_min"]))
        G.add_edge(row["StationB_ID"], row["StationA_ID"],
                   Line=row["Line"], StationA=row["StationB"], StationB=row["StationA"],
                   Distance=row["Distance"], running_time_min=row["running_time_min"],
                   recip_running_time_min=1. / max(1., row["running_time_min"]))
    # FYI Catch & remove the one-way train links: Hatton Cross -> Heathrow Terminal 4 is one-way
    G.remove_edge(235, 234)
    # FYI other removed segments (removed directly from dataset):
    #  Earl's Court <-> Kensington (Olympia) - District line branch only runs on weekends
    #  Camden Town <-> King's Cross & St. Pancras International - parallel branch on Northern line

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

    # T.M.1.2.4 Assign baseline node inflows and outflows
    for n in G.nodes():
        nnlc = G.nodes[n]["NLC"]
        odmat_filtered_in = odmat.loc[odmat["mnlc_o"] == nnlc]
        G.nodes[n]["flow_in"] = sum(odmat_filtered_in["od_tb_3_perhour"])
        odmat_filtered_out = odmat.loc[odmat["mnlc_d"] == nnlc]
        G.nodes[n]["flow_out"] = sum(odmat_filtered_out["od_tb_3_perhour"])

    # T.M.1.2.5 Assign link flow capacities
    for u, v in G.edges():
        unlc = G.nodes[u]["NLC"]
        vnlc = G.nodes[v]["NLC"]
        # Capacities are defined to be directional in TfL RODS dataset
        capac_filtered_1 = capac.loc[capac["StationA_NLC"] == unlc]
        capac_filtered = capac_filtered_1.loc[capac_filtered_1["StationB_NLC"] == vnlc]
        flow_cap_uv = np.dot(capac_filtered["train_capacity"],
                             capac_filtered["train_freq_tb_3_perhour"])
        G.edges[u, v]["flow_cap"] = flow_cap_uv

    # T.M.1.2.6 Assign node thruflow capacities
    for n in G.nodes():
        # Calculate capacities as sum of max(flow_in + incoming flows, flow_out + outgoing flows)
        predecessors = list(G.predecessors(n))
        successors = list(G.successors(n))
        G.nodes[n]["thruflow_cap"] = max(G.nodes[n]["flow_in"] + sum([G.edges[p, n]["flow_cap"] for p in predecessors]),
                                     G.nodes[n]["flow_out"] + sum([G.edges[n, s]["flow_cap"] for s in successors]))

    # T.M.1.2.7 Clean attribute data
    for n in G.nodes():
        G.nodes[n]["type"] = "interchange" if G.nodes[n]["interchange"] else "station"

    return G


# T.M.2.1 Calculate flows for networkx graph
def flow_cen_calc_transport_multiplex_graph(G, odmat):

    # T.M.2.1.1 Calculate shortest paths
    try:
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'r') as json_file:
            shortest_paths = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        shortest_paths = populate_shortest_paths(G, odmat)
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'w') as json_file:
            json.dump(shortest_paths, json_file)  # indent=2, cls=util.CustomEncoder

    # T.M.2.1.2 Assign baseline link flows
    for u, v in G.edges():
        G.edges[u, v]["sp_flow"] = 0
    for s in shortest_paths:
        for t in shortest_paths[s]:
            if shortest_paths[s][t]["path"] is not None:  # if shortest path exists
                for u, v in zip(shortest_paths[s][t]["path"][:-1], shortest_paths[s][t]["path"][1:]):
                    G.edges[u, v]["sp_flow"] += shortest_paths[s][t]["flow"]

    # T.M.2.1.3 Calculate centralities
    G = transport_calc_centrality(G)

    # T.M.2.1.4 Combine into hybrid measure for link flow,
    #  and Assign baseline % capacity utilised
    for u, v in G.edges():
        # G.edges[u, v]["flow"] = (float(G.edges[u, v]["sp_flow"]) ** 0.64) * \
        #                         (float(G.edges[u, v]["edge_current_flow_betweenness"]) ** 0.39)
        G.edges[u, v]["flow"] = G.edges[u, v]["sp_flow"]
        G.edges[u, v]["pct_flow_cap"] = G.edges[u, v]["flow"] / (G.edges[u, v]["flow_cap"] + np.spacing(1.0))

    # T.M.2.1.3 Assign baseline node thruflows,
    #  and Assign baseline % capacity utilised
    for n in G.nodes():
        # Calculate capacities as sum of max(flow_in + incoming flows, flow_out + outgoing flows)
        predecessors = list(G.predecessors(n))
        successors = list(G.successors(n))
        G.nodes[n]["thruflow"] = max(G.nodes[n]["flow_in"] + sum([G.edges[p, n]["flow"] for p in predecessors]),
                                         G.nodes[n]["flow_out"] + sum([G.edges[n, s]["flow"] for s in successors]))
        G.nodes[n]["pct_thruflow_cap"] = G.nodes[n]["thruflow"] / (G.nodes[n]["thruflow_cap"] + np.spacing(1.0))

    return G


# T.M.3 Simplify network
def simplify_transport_multiplex_graph(G):
    # Current rule: Cluster 3 of:
    # - adjacent & consecutive stations with degree 2; and excluding
    # - line terminals
    # into a single hypernode
    adj_nonitc_path_list = list()
    for u, v in G.edges():
        if (G.in_degree[u] == 2) and (G.out_degree[u] == 2) and (G.in_degree[v] == 2) and (G.out_degree[v] == 2) \
                and u != 362 and v != 362:  # To manually ignore Upminster - i.e. the only line terminus of degree 2
            u_nb = [n for n in G.neighbors(u) if n != v and G.in_degree[n] == 2 and G.out_degree[n] == 2]
            v_nb = [n for n in G.neighbors(v) if n != u and G.in_degree[n] == 2 and G.out_degree[n] == 2]
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


# T.M.4.1 Check flows
def flow_check(G):
    for u, v in G.edges():
        if (G.edges[u, v]["flow"] > G.edges[u, v]["flow_cap"]) \
                or (G.edges[u, v]["flow"] == 0) or (G.edges[u, v]["flow_cap"] == 0):
            print("Link Warning: ", u, v, G.edges[u, v]["flow"], G.edges[u, v]["flow_cap"])
    for n in G.nodes():
        if (G.nodes[n]["thruflow"] > G.nodes[n]["thruflow_cap"]) \
                or (G.nodes[n]["thruflow"] == 0) or (G.nodes[n]["thruflow_cap"] == 0):
            print("Node Warning: ", n, G.nodes[n]["thruflow"], G.nodes[n]["thruflow_cap"])


# def plot_degree_histogram(G, edges):
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

    odmat = None

    # T.M.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', "rb"))
    except IOError:
        nodes, edges, odmat, capac = load_transport_multiplex_data()
        G = create_transport_multiplex_graph(nodes, edges, odmat, capac)
        try:
            pickle.dump(G, open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.2 Calculate baseline flows
    try:
        G_flow = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', "rb"))
    except IOError:
        if odmat is None:
            odmat = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_odmat.pkl")
        G_flow = flow_cen_calc_transport_multiplex_graph(G, odmat)
        try:
            pickle.dump(G_flow, open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.3 Simplify network
    try:
        G_skele = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', "rb"))
    except (IOError, FileNotFoundError, json.decoder.JSONDecodeError):
        G_skele = simplify_transport_multiplex_graph(G_flow)
        G_skele = simplify_transport_multiplex_graph(G_skele)  # Run it again!
        G_skele = transport_calc_centrality(G_skele)
        if odmat is None:
            odmat = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_odmat.pkl")
        shortest_paths_skele = populate_shortest_paths_skele(G_skele, odmat)
        # Beyond this the graph can no longer be further simplified using the current rules
        try:
            pickle.dump(G_skele, open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)
        with open(r'data/transport_multiplex/flow/shortest_paths_skele.json', 'w') as json_file:
            json.dump(shortest_paths_skele, json_file)  # indent=2, cls=util.CustomEncoder

    flow_check(G_skele)

    # T.M.5 Export graph nodelist
    # try:
    #     combined_nodelist = pd.DataFrame([i[1] for i in G_skele.nodes(data=True)], index=[i[0] for i in G_skele.nodes(data=True)])
    #     combined_nodelist = combined_nodelist.rename_axis('full_id')
    #     combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist.xlsx', index=True)
    # except Exception as e:
    #     print(e)
    # try:
    #     combined_nodelist = pd.DataFrame([i[1] for i in G_flow.nodes(data=True)], index=[i[0] for i in G_flow.nodes(data=True)])
    #     combined_nodelist = combined_nodelist.rename_axis('full_id')
    #     combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist_unsimpl.xlsx', index=True)
    # except Exception as e:
    #     print(e)

    # T.M.6 Export adjacency matrix
    # try:
    #     combined_adjmat = nx.adjacency_matrix(G_skele, nodelist=None, weight="weight")  # gives scipy sparse matrix
    #     sparse.save_npz(r'data/transport_multiplex/out/transport_multiplex_adjmat.npz', combined_adjmat)
    # except Exception as e:
    #     print(e)
    # try:
    #     combined_adjmat = nx.adjacency_matrix(G_flow, nodelist=None, weight="weight")  # gives scipy sparse matrix
    #     sparse.save_npz(r'data/transport_multiplex/out/transport_multiplex_adjmat_unsimpl.npz', combined_adjmat)
    # except Exception as e:
    #     print(e)

    # T.M.7 Export graph edgelist
    # try:
    #     edgelist = pd.DataFrame(columns=["StationA_ID", "StationA", "StationA_NLC",
    #                                      "StationB_ID", "StationB", "StationB_NLC",
    #                                      "flow", "flow_cap", "pct_flow_cap",
    #                                      "actual_flow", "rel_error",
    #                                      "edge_betweenness", "edge_current_flow_betweenness"])
    #
    #     link_loads = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
    #                                       sheet_name="link_loads", header=0)
    #     for u, v in G_skele.edges():
    #         if isinstance(u, int):
    #             actual_flow = link_loads.loc[link_loads["From ID"] == u]
    #         else:
    #             u_list = [int(x) for x in u.split("-")]
    #             actual_flow = link_loads.loc[link_loads["From ID"] == u_list[-1]]
    #         if isinstance(v, int):
    #             actual_flow = actual_flow.loc[link_loads["To ID"] == v]
    #         else:
    #             v_list = [int(x) for x in v.split("-")]
    #             actual_flow = link_loads.loc[link_loads["To ID"] == v_list[0]]
    #         actual_flow = np.nansum(actual_flow["AM Peak per hour"])
    #         G_skele.edges[u, v]["actual_flow"] = actual_flow
    #         rel_error = abs(G_skele.edges[u, v]["flow"] - actual_flow) / max(1.e-5, actual_flow)
    #         G_skele.edges[u, v]["rel_error"] = rel_error
    #
    #         series_obj = pd.Series([u, G_skele.nodes[u]["nodeLabel"], G_skele.nodes[u]["NLC"],
    #                                 v, G_skele.nodes[v]["nodeLabel"], G_skele.nodes[v]["NLC"],
    #                                 G_skele.edges[u, v].get("flow", "No Data"),
    #                                 G_skele.edges[u, v].get("flow_cap", "No Data"),
    #                                 G_skele.edges[u, v].get("pct_flow_cap", "No Data"),
    #                                 actual_flow, rel_error,
    #                                 G_skele.edges[u, v].get("edge_betweenness", "No Data"),
    #                                 G_skele.edges[u, v].get("edge_current_flow_betweenness", "No Data")],
    #                                index=edgelist.columns)
    #         edgelist = edgelist.append(series_obj, ignore_index=True)
    #     edgelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_edgelist.xlsx', index=True)
    # except Exception as e:
    #     raise e
    #
    # try:
    #     edgelist = pd.DataFrame(columns=["StationA_ID", "StationA", "StationA_NLC",
    #                                      "StationB_ID", "StationB", "StationB_NLC",
    #                                      "flow", "flow_cap", "pct_flow_cap",
    #                                      "actual_flow", "rel_error",
    #                                      "edge_betweenness", "edge_current_flow_betweenness"])
    #
    #     link_loads = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
    #                                       sheet_name="link_loads", header=0)
    #     for u, v in G_flow.edges():
    #         if isinstance(u, int):
    #             actual_flow = link_loads.loc[link_loads["From ID"] == u]
    #         else:
    #             u_list = [int(x) for x in u.split("-")]
    #             actual_flow = link_loads.loc[link_loads["From ID"] == u_list[-1]]
    #         if isinstance(v, int):
    #             actual_flow = actual_flow.loc[link_loads["To ID"] == v]
    #         else:
    #             v_list = [int(x) for x in v.split("-")]
    #             actual_flow = link_loads.loc[link_loads["To ID"] == v_list[0]]
    #         actual_flow = np.nansum(actual_flow["AM Peak per hour"])
    #         G_flow.edges[u, v]["actual_flow"] = actual_flow
    #         rel_error = abs(G_flow.edges[u, v]["flow"] - actual_flow) / max(1.e-5, actual_flow)
    #         G_flow.edges[u, v]["rel_error"] = rel_error
    #
    #         series_obj = pd.Series([u, G_flow.nodes[u]["nodeLabel"], G_flow.nodes[u]["NLC"],
    #                                 v, G_flow.nodes[v]["nodeLabel"], G_flow.nodes[v]["NLC"],
    #                                 G_flow.edges[u, v].get("flow", "No Data"),
    #                                 G_flow.edges[u, v].get("flow_cap", "No Data"),
    #                                 G_flow.edges[u, v].get("pct_flow_cap", "No Data"),
    #                                 actual_flow, rel_error,
    #                                 G_flow.edges[u, v].get("edge_betweenness", "No Data"),
    #                                 G_flow.edges[u, v].get("edge_current_flow_betweenness", "No Data")],
    #                                index=edgelist.columns)
    #         edgelist = edgelist.append(series_obj, ignore_index=True)
    #     edgelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_edgelist_unsimpl.xlsx', index=True)
    # except Exception as e:
    #     raise e

    # T.M.8 Add colours and export map plot
    G_skele_undir = transport_to_undirected(G_skele)

    node_colors = [TRANSPORT_COLORS.get(G_skele_undir.nodes[node]["line"], "#FF0000") for node in G_skele_undir.nodes()]
    edge_lines = nx.get_edge_attributes(G_skele_undir, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G_skele_undir.edges()]
    # widths = [G_skele_undir.edges[u, v]["pct_flow_cap"] * 4 + 0.2 for u, v in G_skele_undir.edges()]
    # widths = [min(8, G_skele_undir.edges[u, v]["rel_error"] * 4 + 0.2) for u, v in G_skele_undir.edges()]
    widths = 1

    nx.draw(G_skele_undir, pos=nx.get_node_attributes(G_skele_undir, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors, width=widths)
    plt.savefig("data/transport_multiplex/img/transport_multiplex.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex.svg")
    plt.show()

    G_flow_undir = transport_to_undirected(G_flow)

    node_colors = [TRANSPORT_COLORS.get(G_flow_undir.nodes[node]["line"], "#FF0000") for node in G_flow_undir.nodes()]
    edge_lines = nx.get_edge_attributes(G_flow_undir, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G_flow_undir.edges()]
    # widths = [G_flow_undir.edges[u, v]["pct_flow_cap"] * 4 + 0.2 for u, v in G_flow_undir.edges()]
    # widths = [min(8, G_flow_undir.edges[u, v]["rel_error"] * 4 + 0.2) for u, v in G_flow_undir.edges()]
    widths = 1

    nx.draw(G_flow_undir, pos=nx.get_node_attributes(G_flow_undir, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors, width=widths)
    plt.savefig("data/transport_multiplex/img/transport_multiplex_unsimpl.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex_unsimpl.svg")
    plt.show()

    # _, edges, _, _ = load_transport_multiplex_data()
    # plot_degree_histogram(G_skele, edges)

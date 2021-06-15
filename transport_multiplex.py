import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import json
import numpy as np

from globalparams import TRANSPORT_COLORS, TRAIN_DUR_THRESHOLD_MIN
from util import linear_cluster, transport_calc_centrality


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
                   Distance=row["Distance"], running_time_min=row["running_time_min"])
        G.add_edge(row["StationB_ID"], row["StationA_ID"],
                   Line=row["Line"], StationA=row["StationB"], StationB=row["StationA"],
                   Distance=row["Distance"], running_time_min=row["running_time_min"])
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
        G.nodes[n]["flow_in"] = round(sum(odmat_filtered_in["od_tb_3_perhour"]))
        odmat_filtered_out = odmat.loc[odmat["mnlc_d"] == nnlc]
        G.nodes[n]["flow_out"] = round(sum(odmat_filtered_out["od_tb_3_perhour"]))

    # T.M.1.2.5 Assign link flow capacities
    for u, v in G.edges():
        unlc = G.nodes[u]["NLC"]
        vnlc = G.nodes[v]["NLC"]
        # Capacities are defined to be directional in TfL RODS dataset
        capac_filtered_1 = capac.loc[capac["StationA_NLC"] == unlc]
        capac_filtered = capac_filtered_1.loc[capac_filtered_1["StationB_NLC"] == vnlc]
        flow_cap_uv = np.dot(capac_filtered["train_capacity"],
                             capac_filtered["train_freq_tb_3_perhour"])
        G.edges[u, v]["flow_cap"] = round(flow_cap_uv)

    # T.M.1.2.6 Assign node thruflow capacities
    for n in G.nodes():
        # Initialise capacities
        G.nodes[n]["thruflow_cap"] = G.nodes[n]["flow_in"] + G.nodes[n]["flow_out"]
        # Calculate capacities as sum of flow capacities of outgoing links
        for nsucc in G.successors(n):
            G.nodes[n]["thruflow_cap"] += G.edges[n, nsucc]["flow_cap"]
        for npred in G.predecessors(n):
            G.nodes[n]["thruflow_cap"] += G.edges[npred, n]["flow_cap"]
        G.nodes[n]["thruflow_cap"] = round(G.nodes[n]["thruflow_cap"] / 2)  # to avoid double-counting

    # T.M.1.2.7 Clean attribute data
    for n in G.nodes():
        G.nodes[n]["type"] = "interchange" if G.nodes[n]["interchange"] else "station"
    for u, v in G.edges():
        G.edges[u, v].pop("StationA", None)
        G.edges[u, v].pop("StationB", None)

    return G


# T.M.2.1 Calculate flows for networkx graph
def flowcalc_transport_multiplex_graph(G, odmat):

    # T.M.2.1.1 Calculate shortest paths
    try:
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'r') as json_file:
            shortest_paths = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        shortest_paths = dict()
        for s in G.nodes():
            # FYI Things to catch:
            #  - Excessively long shortest paths (above cutoff TRAIN_DUR_THRESHOLD_MIN)
            #  - Origin node no longer exists (have to catch outside loop)
            #  - Destination node no longer exists (have to catch outside loop)
            #  - No path exists (have to catch outside loop)
            if s not in shortest_paths:
                shortest_paths[s] = dict()
            travel_time, path = nx.single_source_dijkstra(G, source=s,
                                                          cutoff=TRAIN_DUR_THRESHOLD_MIN,
                                                          weight='running_time_min')
            for t in G.nodes():
                if s != t:
                    if t not in shortest_paths[s]:
                        shortest_paths[s][t] = dict()
                    try:
                        shortest_paths[s][t]["path"] = path[t]
                        # shortest_paths[s][t]["path_names"] = [G.nodes[n]["nodeLabel"] for n in
                        #                                       shortest_paths[s][t]["path"]]
                        shortest_paths[s][t]["travel_time"] = travel_time[t]
                        shortest_paths[s][t]["length"] = len(shortest_paths[s][t]["path"])
                    except KeyError:  # Excessively long shortest paths (above cutoff TRAIN_DUR_THRESHOLD_MIN)
                        shortest_paths[s][t]["path"] = None
                        # shortest_paths[s][t]["path_names"] = None
                        shortest_paths[s][t]["travel_time"] = np.inf
                        shortest_paths[s][t]["length"] = np.inf
        for s in shortest_paths:
            # PATCH - keys in json are in strings for some reason (JSON encoder issue)
            snlc = G.nodes[int(s)]["NLC"]
            for t in shortest_paths[s]:
                tnlc = G.nodes[int(t)]["NLC"]
                odmat_filtered = odmat.loc[odmat["mnlc_o"] == snlc]
                odmat_filtered = odmat_filtered.loc[odmat["mnlc_d"] == tnlc]
                flow_st = sum(odmat_filtered["od_tb_3_perhour"])
                shortest_paths[s][t]["flow"] = flow_st
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'w') as json_file:
            json.dump(shortest_paths, json_file)  # indent=2, cls=util.CustomEncoder

    # T.M.2.1.2 Assign baseline link flows
    for u, v in G.edges():
        G.edges[u, v]["flow"] = 0
    for s in shortest_paths:
        for t in shortest_paths[s]:
            if shortest_paths[s][t]["path"] is not None:  # if shortest path exists
                for u, v in zip(shortest_paths[s][t]["path"][:-1], shortest_paths[s][t]["path"][1:]):
                    G.edges[u, v]["flow"] += round(shortest_paths[s][t]["flow"])

    # T.M.2.1.3 Assign baseline node thruflows
    for n in G.nodes():
        # Initialise through flows
        G.nodes[n]["thruflow"] = 0
        # Calculate through flows
        for ne in G.neighbors(n):  # this considers only successors for digraph
            G.nodes[n]["thruflow"] += G.edges[n, ne]["flow"]

    # T.M.2.1.4 Assign baseline link flow % capacity utilised
    for u, v in G.edges():
        G.edges[u, v]["pct_flow_cap"] = G.edges[u, v]["flow"] / G.edges[u, v]["flow_cap"]

    # T.M.2.1.5 Assign baseline node thruflow % capacity utilised
    for n in G.nodes():
        G.nodes[n]["pct_thruflow_cap"] = G.nodes[n]["thruflow"] / G.nodes[n]["thruflow_cap"]

    return G


# T.M.2.2 Alternative method to calculate flows for networkx graph, using capacity scaling
# def flowcalc_transport_multiplex_graph_capscal(G):
#     total_supply = sum([G.nodes[n]["flow_in"] for n in G.nodes()])
#     total_demand = sum([G.nodes[n]["flow_out"] for n in G.nodes()])
#     for n in G.nodes():
#         G.nodes[n]["demand"] = round(G.nodes[n]["flow_out"] - G.nodes[n]["flow_in"] * total_demand / total_supply)
#     G.nodes[n]["demand"] -= sum(G.nodes[u].get("demand", 0) for u in G)  # PATCH FIXME
#     for u, v in G.edges():
#         G.edges[u, v]["capacity"] = round(G.edges[u, v]["flow_cap"])
#         G.edges[u, v]["weight"] = round(G.edges[u, v]["running_time_min"] * 60)  # convert to int seconds (must be int)
#     flowCost, flowDict = nx.capacity_scaling(G, demand='demand', capacity='capacity', weight='weight')
#     for u, v in G.edges():
#         G.edges[u, v]["flow_capscal"] = flowDict[u][v]
#     return G


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


# T.M.x Graph stats summary
def stats_summary(G):
    print("Stats summary:")
    print("sigma:", nx.sigma(G, niter=1, nrand=5), "(small world if >1)")
    print("omega:", nx.omega(G, niter=1, nrand=5), "(lattice if ~-1, small world if ~0, random if ~1)")


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
        odmat = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_odmat.pkl")
        G_flow = flowcalc_transport_multiplex_graph(G, odmat)
        # G_flow = flowcalc_transport_multiplex_graph_capscal(G_flow)  # DEBUG alternative flow calc
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
        # Beyond this the graph can no longer be further simplified using the current rules
        try:
            pickle.dump(G_skele, open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.4 Check flows and calculate centrality
    # flow_check(G_skele)
    G_skele = transport_calc_centrality(G_skele)
    try:
        pickle.dump(G_skele, open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', 'wb+'))
    except FileNotFoundError as e:
        print(e)
    # flow_check(G_flow)
    G_flow = transport_calc_centrality(G_flow)
    try:
        pickle.dump(G_flow, open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', 'wb+'))
    except FileNotFoundError as e:
        print(e)
    # Graph stats summary
    # stats_summary(G)
    # stats_summary(G_flow)

    # T.M.5 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G_skele.nodes(data=True)], index=[i[0] for i in G_skele.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G_flow.nodes(data=True)], index=[i[0] for i in G_flow.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist_unsimpl.xlsx', index=True)
    except Exception as e:
        print(e)

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
    try:
        # edgelist = pd.DataFrame(columns=["StationA_ID", "StationA", "StationA_NLC",
        #                                  "StationB_ID", "StationB", "StationB_NLC",
        #                                  "flow",
        #                                  # "flow_capscal",
        #                                  "flow_cap", "pct_flow_cap",
        #                                  "actual_flow", "rel_error",
        #                                  "edge_betweenness", "edge_current_flow_betweenness"])

        link_loads = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                                          sheet_name="link_loads", header=0)
        for u, v in G_skele.edges():
            if isinstance(u, int):
                actual_flow = link_loads.loc[link_loads["From ID"] == u]
            else:
                u_list = [int(x) for x in u.split("-")]
                actual_flow = link_loads.loc[link_loads["From ID"] == u_list[-1]]
            if isinstance(v, int):
                actual_flow = actual_flow.loc[link_loads["To ID"] == v]
            else:
                v_list = [int(x) for x in v.split("-")]
                actual_flow = link_loads.loc[link_loads["To ID"] == v_list[0]]
            actual_flow = np.nansum(actual_flow["AM Peak per hour"])
            rel_error = abs(G_skele.edges[u, v]["flow"] - actual_flow) / max(1.e-5, actual_flow)
            G_skele.edges[u, v]["rel_error"] = rel_error

        #     series_obj = pd.Series([u, G_skele.nodes[u]["nodeLabel"], G_skele.nodes[u]["NLC"],
        #                             v, G_skele.nodes[v]["nodeLabel"], G_skele.nodes[v]["NLC"],
        #                             G_skele.edges[u, v]["flow"],
        #                             # G_skele.edges[u, v]["flow_capscal"],
        #                             G_skele.edges[u, v]["flow_cap"], G_skele.edges[u, v]["pct_flow_cap"],
        #                             actual_flow, rel_error,
        #                             G_skele.edges[u, v]["edge_betweenness"],
        #                             G_skele.edges[u, v]["edge_current_flow_betweenness"]],
        #                            index=edgelist.columns)
        #     edgelist = edgelist.append(series_obj, ignore_index=True)
        # edgelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_edgelist.xlsx', index=True)
    except Exception as e:
        raise e

    try:
        # edgelist = pd.DataFrame(columns=["StationA_ID", "StationA", "StationA_NLC",
        #                                  "StationB_ID", "StationB", "StationB_NLC",
        #                                  "flow",
        #                                  # "flow_capscal",
        #                                  "flow_cap", "pct_flow_cap",
        #                                  "actual_flow", "rel_error",
        #                                  "edge_betweenness", "edge_current_flow_betweenness"])

        link_loads = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                                          sheet_name="link_loads", header=0)
        for u, v in G_flow.edges():
            if isinstance(u, int):
                actual_flow = link_loads.loc[link_loads["From ID"] == u]
            else:
                u_list = [int(x) for x in u.split("-")]
                actual_flow = link_loads.loc[link_loads["From ID"] == u_list[-1]]
            if isinstance(v, int):
                actual_flow = actual_flow.loc[link_loads["To ID"] == v]
            else:
                v_list = [int(x) for x in v.split("-")]
                actual_flow = link_loads.loc[link_loads["To ID"] == v_list[0]]
            actual_flow = np.nansum(actual_flow["AM Peak per hour"])
            rel_error = abs(G_flow.edges[u, v]["flow"] - actual_flow) / max(1.e-5, actual_flow)
            G_flow.edges[u, v]["rel_error"] = rel_error

        #     series_obj = pd.Series([u, G_flow.nodes[u]["nodeLabel"], G_flow.nodes[u]["NLC"],
        #                             v, G_flow.nodes[v]["nodeLabel"], G_flow.nodes[v]["NLC"],
        #                             G_flow.edges[u, v]["flow"],
        #                             # G_flow.edges[u, v]["flow_capscal"],
        #                             G_flow.edges[u, v]["flow_cap"], G_flow.edges[u, v]["pct_flow_cap"],
        #                             actual_flow, rel_error,
        #                             G_flow.edges[u, v]["edge_betweenness"],
        #                             G_flow.edges[u, v]["edge_current_flow_betweenness"]],
        #                            index=edgelist.columns)
        #     edgelist = edgelist.append(series_obj, ignore_index=True)
        # edgelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_edgelist_unsimpl.xlsx', index=True)
    except Exception as e:
        raise e

    # T.M.8 Add colours and export map plot
    G_skele_undir = G_skele.to_undirected()

    node_colors = [TRANSPORT_COLORS.get(G_skele_undir.nodes[node]["line"], "#FF0000") for node in G_skele_undir.nodes()]
    edge_lines = nx.get_edge_attributes(G_skele_undir, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G_skele_undir.edges()]
    # widths = [G_skele_undir.edges[u, v]["pct_flow_cap"] * 4 + 0.2 for u, v in G_skele_undir.edges()]
    widths = [min(8, G_skele_undir.edges[u, v]["rel_error"] * 4 + 0.2) for u, v in G_skele_undir.edges()]

    nx.draw(G_skele_undir, pos=nx.get_node_attributes(G_skele_undir, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors, width=widths)
    plt.savefig("data/transport_multiplex/img/transport_multiplex.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex.svg")
    plt.show()

    G_flow_undir = G_flow.to_undirected()

    node_colors = [TRANSPORT_COLORS.get(G_flow_undir.nodes[node]["line"], "#FF0000") for node in G_flow_undir.nodes()]
    edge_lines = nx.get_edge_attributes(G_flow_undir, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G_flow_undir.edges()]
    # widths = [G_flow_undir.edges[u, v]["pct_flow_cap"] * 4 + 0.2 for u, v in G_flow_undir.edges()]
    widths = [min(8, G_flow_undir.edges[u, v]["rel_error"] * 4 + 0.2) for u, v in G_flow_undir.edges()]

    nx.draw(G_flow_undir, pos=nx.get_node_attributes(G_flow_undir, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors, width=widths)
    plt.savefig("data/transport_multiplex/img/transport_multiplex_unsimpl.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex_unsimpl.svg")
    plt.show()

    # _, edges, _, _ = load_transport_multiplex_data()
    # plot_degree_histogram(G_skele, edges)

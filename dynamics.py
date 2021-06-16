import networkx as nx
import numpy as np
import pandas as pd
import random

from util import weakly_connected_component_subgraphs, transport_to_undirected, transport_calc_centrality, \
    power_calc_centrality


def get_node(G, network, mode="random", top=1):
    """Returns a list of nodes in infrastructure node G, to be failed"""

    assert network in ["power", "transport"]
    assert mode in ["random", "degree", "closeness", "betweenness", "thruflow", "thruflow_cap", "pct_thruflow_cap"]
    assert 1 <= top <= G.number_of_nodes()

    if network in ["power", "transport"]:
        G = G.subgraph([n for n in G.nodes() if G.nodes[n]["network"] == network]).copy()

    if mode == "random":
        node_list = [n for n in G.nodes()]
        ids = random.choices(node_list, k=top)
        labels = [G.nodes[n]["nodeLabel"] for n in ids]
        return ids, labels
    elif mode in ["degree", "closeness", "betweenness"]:
        if mode == "degree":
            cc = nx.degree_centrality(G)
        elif mode == "closeness":
            cc = nx.closeness_centrality(G, distance="resistance" if network == "power" else "running_time_min")
        else:
            cc = nx.betweenness_centrality(G, normalized=True,
                                           weight="resistance" if network == "power" else "running_time_min")
        df = pd.DataFrame.from_dict({
            'node': list(cc.keys()),
            'centrality': list(cc.values())
        })
        df = df.sort_values('centrality', ascending=False)
        l = df["node"].tolist()
        ids = l[0:top]
        labels = [G.nodes[n]["nodeLabel"] for n in ids]
        return ids, labels
    elif mode in ["thruflow", "thruflow_cap", "pct_thruflow_cap"]:
        df = pd.DataFrame.from_dict({
            'node': [n for n in G.nodes()],
            mode: [G.nodes[n][mode] for n in G.nodes()]
        })
        df = df.sort_values(mode, ascending=False)
        l = df["node"].tolist()
        ids = l[0:top]
        labels = [G.nodes[n]["nodeLabel"] for n in ids]
        return ids, labels


def get_link(G, network, mode="random", top=1):
    """Returns a list of links in infrastructure node G, to be failed. This ignores all interdependencies."""

    assert network in ["power", "transport"]
    assert mode in ["random", "betweenness", "flow", "flow_cap", "pct_flow_cap"]
    assert 1 <= top <= G.number_of_edges()

    if network in ["power", "transport"]:
        G = G.subgraph([n for n in G.nodes() if G.nodes[n]["network"] == network]).copy()

    if mode == "random":
        link_list = [(u, v) for u, v in G.edges()]
        ids = random.choices(link_list, k=top)
        labels = [(G.nodes[u]["nodeLabel"], G.nodes[v]["nodeLabel"]) for (u, v) in ids]
        return ids, labels
    elif mode in ["betweenness"]:
        if mode == "betweenness":
            cc = nx.edge_betweenness_centrality(G)
        df = pd.DataFrame.from_dict({
            'link': list(cc.keys()),
            'centrality': list(cc.values())
        })
        df = df.sort_values('centrality', ascending=False)
        l = df["link"].tolist()
        ids = l[0:top]
        labels = [(G.nodes[u]["nodeLabel"], G.nodes[v]["nodeLabel"]) for (u, v) in ids]
        return ids, labels
    elif mode in ["flow", "flow_cap", "pct_flow_cap"]:
        df = pd.DataFrame.from_dict({
            'link': [(u, v) for u, v in G.edges()],
            mode: [G.edges[u, v][mode] for u, v in G.edges()]
        })
        df = df.sort_values(mode, ascending=False)
        l = df["link"].tolist()
        ids = l[0:top]
        labels = [(G.nodes[u]["nodeLabel"], G.nodes[v]["nodeLabel"]) for (u, v) in ids]
        return ids, labels
    else:
        return list()


def percolate_nodes(G, failed_nodes):
    # TODO TO TEST
    """Returns the network after failing the selected nodes. This does not do centrality or flow recalculation."""
    for fn in failed_nodes:
        assert fn in G.nodes()
        G.nodes[fn]["state"] = 0
        if G.is_directed():  # Also fail all links adjacent to fn to ensure a closed network
            for su in G.successors(fn):
                G.edges[fn, su]["state"] = 0
            for pr in G.predecessors(fn):
                G.edges[pr, fn]["state"] = 0
        else:
            for ne in G.neighbours(fn):
                G.edges[fn, ne]["state"] = 0
    return G


def percolate_links(G, failed_links, reversible=True):
    # TODO TO TEST
    """Returns the network after failing the selected links. This does not do centrality or flow recalculation."""
    for fl in failed_links:
        assert fl in G.edges()
        if G.is_directed():
            G.edges[fl]["state"] = 0
            if reversible:
                if (fl[1], fl[0]) in G.edges:  # Also fail the link going in opposite direction
                    G.edges[fl[1], fl[0]]["state"] = 0
        else:
            G.edges[fl]["state"] = 0
    return G


def recompute_flows(G, newly_failed_nodes, newly_failed_links, shortest_paths):
    """Recompute all flows for network given node and link failures.
    Here, the network contains ONLY nodes/links that are still working (i.e. not failed).
    It is assumed that newly_failed_nodes and newly_failed_links are consistent with each other,
    i.e. both ends of failed links are in failed nodes, and all links adjacent to failed nodes are failed links"""

    # Split networks
    GT = G.subgraph([n for n in G.nodes() if G.nodes[n]["network"] == "transport"]).copy()
    GP = G.subgraph([n for n in G.nodes() if G.nodes[n]["network"] == "power"]).copy()

    # Create new copy of overall network
    new_G = G.deepcopy()

    # TRANSPORT NETWORK RECOMPUTATION

    # Populate affected shortest paths
    affected_shortest_paths = list()
    for fl in newly_failed_links:
        for s in shortest_paths:
            for t in shortest_paths[s]:  # iterate through all shortest paths
                for u, v in zip(shortest_paths[s][t]["path"][:-1], shortest_paths[s][t]["path"][1:]):
                    if fl == (u, v):  # if newly-failed link is along shortest path
                        affected_shortest_paths.append((s, t))

    # Create new copy of shortest_paths dict
    new_shortest_paths = shortest_paths.deepcopy()

    # Track unfulfilled trips and/or calculate new shortest paths
    for sp in affected_shortest_paths:
        (s, t) = sp
        if s in newly_failed_nodes:
            print(s, t, "Unfulfilled trips because s failed")
            new_shortest_paths[s][t]["path"] = None
            new_shortest_paths[s][t]["travel_time"] = np.inf
            new_shortest_paths[s][t]["length"] = np.inf
            new_shortest_paths[s][t]["flow"] = shortest_paths[s][t]["flow"]  # O-D matrix remains the same
        elif t in newly_failed_nodes:
            print(s, t, "Unfulfilled trips because t failed")
            new_shortest_paths[s][t]["path"] = None
            new_shortest_paths[s][t]["travel_time"] = np.inf
            new_shortest_paths[s][t]["length"] = np.inf
            new_shortest_paths[s][t]["flow"] = shortest_paths[s][t]["flow"]  # O-D matrix remains the same
        else:
            try:
                # This implementation reduces the number of Dijkstra runs significantly (only run for affected st-pairs)
                travel_time, path = nx.dijkstra_path(GT, source=s, target=t, weight='running_time_min')
                new_shortest_paths[s][t]["path"] = path
                new_shortest_paths[s][t]["travel_time"] = travel_time
                new_shortest_paths[s][t]["length"] = len(new_shortest_paths[s][t]["path"])
                print(s, t, "Rerouted successfully")
            except nx.NetworkXNoPath:  # No path exists between source and target
                print(s, t, "Unfulfilled trips because no SP exists between s and t")
                new_shortest_paths[s][t]["path"] = None
                new_shortest_paths[s][t]["travel_time"] = np.inf
                new_shortest_paths[s][t]["length"] = np.inf

    # Adjust flows accordingly
    for sp in affected_shortest_paths:
        (s, t) = sp
        # Remove flows from all links in affected SP
        for u, v in zip(shortest_paths[s][t]["path"][:-1], shortest_paths[s][t]["path"][1:]):
            new_G.edges[u, v]["sp_flow"] -= shortest_paths[s][t]["flow"]
        # Add flows from all links in new SP
        for u, v in zip(new_shortest_paths[s][t]["path"][:-1], new_shortest_paths[s][t]["path"][1:]):
            new_G.edges[u, v]["sp_flow"] += new_shortest_paths[s][t]["flow"]

    for subGT in weakly_connected_component_subgraphs(GT, copy=True):
        subGT = transport_calc_centrality(subGT)
        for n in subGT.nodes():
            new_G.nodes[n]["betweenness"] = subGT.nodes[n]["betweenness"]
            new_G.nodes[n]["closeness"] = subGT.nodes[n]["closeness"]
            new_G.nodes[n]["current_flow_betweenness"] = subGT.nodes[n]["current_flow_betweenness"]
        for u, v in subGT.edges():
            new_G.edges[u, v]["edge_betweenness"] = subGT.edges[u, v]["edge_betweenness"]
            new_G.edges[u, v]["edge_current_flow_betweenness"] = subGT.edges[u, v]["edge_current_flow_betweenness"]

    # Combine into hybrid measure for link flow,
    #  and Assign baseline % capacity utilised
    for u, v in GT.edges():
        new_G.edges[u, v]["flow"] = (float(new_G.edges[u, v]["sp_flow"]) ** 0.64) * \
                                    (float(new_G.edges[u, v]["edge_current_flow_betweenness"]) ** 0.39)
        new_G.edges[u, v]["pct_flow_cap"] = new_G.edges[u, v]["flow"] / new_G.edges[u, v]["flow_cap"]

    # Assign baseline node thruflows,
    #  and Assign baseline % capacity utilised
    for n in GT.nodes():
        # Calculate capacities as sum of max(flow_in + incoming flows, flow_out + outgoing flows)
        predecessors = list(GT.predecessors(n))
        successors = list(GT.successors(n))
        new_G.nodes[n]["thruflow"] = max(new_G.nodes[n]["flow_in"] + sum([new_G.edges[p, n]["flow"] for p in predecessors]),
                                     new_G.nodes[n]["flow_out"] + sum([new_G.edges[n, s]["flow"] for s in successors]))
        new_G.nodes[n]["pct_thruflow_cap"] = new_G.nodes[n]["thruflow"] / new_G.nodes[n]["thruflow_cap"]

    # POWER NETWORK RECOMPUTATION
    if not GP.is_weakly_connected():
        for subGP in weakly_connected_component_subgraphs(GP, copy=True):
            # If total supply and inflow cannot meet demand, whole subgrid fails
            total_supply = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if G.nodes[n]["type"] in ["GSP_transmission"]])
            total_inflow = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if G.nodes[n]["type"] in ["generator"]])
            total_demand = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if G.nodes[n]["type"] in ["load"]])
            if total_demand > total_supply + total_inflow:
                # Failed everything here - TODO anything else? - TODO should move this elsewhere
                print("Subgrid with", subGP.nodes[0], "failed because Total demand",
                      total_demand, "> Total inflow", total_inflow, "+ Total supply", total_supply)
                for n in subGP.nodes():
                    new_G.edges[n]["thruflow"] = 0  # TODO flag failure state later
                for u, v in subGP.edges():
                    new_G.edges[u, v]["thruflow"] = 0  # TODO flag failure state later
            else:
                subGP = power_calc_centrality(subGP)
                for n in subGP.nodes():
                    new_G.nodes[n]["current_flow_betweenness"] = subGP.nodes[n]["current_flow_betweenness"]
                for u, v in subGP.edges():
                    new_G.edges[u, v]["edge_betweenness"] = subGP.edges[u, v]["edge_betweenness"]
                    new_G.edges[u, v]["edge_current_flow_betweenness"] \
                        = subGP.edges[u, v]["edge_current_flow_betweenness"]
    else:
        GP = power_calc_centrality(GP)
        for n in GP.nodes():
            new_G.nodes[n]["current_flow_betweenness"] = GP.nodes[n]["current_flow_betweenness"]
        for u, v in GP.edges():
            new_G.edges[u, v]["edge_betweenness"] = GP.edges[u, v]["edge_betweenness"]
            new_G.edges[u, v]["edge_current_flow_betweenness"] \
                = GP.edges[u, v]["edge_current_flow_betweenness"]

    return new_G, new_shortest_paths


def filter_functional_network(G):
    return G.subgraph([n for n in G.nodes() if G.nodes[n]["state"] == 1]).copy()

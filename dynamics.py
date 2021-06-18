import copy

import networkx as nx
import numpy as np
import pandas as pd
import random

from util import weakly_connected_component_subgraphs, transport_calc_centrality, power_calc_centrality


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
        print("Node(s)", str(ids), "initial failure")
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
        print("Node(s)", str(ids), "initial failure")
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
        print("Node(s)", str(ids), "initial failure")
        return ids, labels
    else:
        return list()


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
        print("Link(s)", str(ids), "initial failure")
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
        print("Link(s)", str(ids), "initial failure")
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
        print("Link(s)", str(ids), "initial failure")
        return ids, labels
    else:
        return list()


def percolate_nodes(g, failed_nodes):
    """Returns the network after failing the selected nodes. This does not do centrality or flow recalculation."""
    assert isinstance(failed_nodes, list)
    G = g.copy()
    failed_links = list()
    for fn in failed_nodes:
        assert fn in list(G.nodes())
        G.nodes[fn]["state"] = 0
        if G.is_directed():  # Also fail all links adjacent to fn to ensure a closed network
            for su in G.successors(fn):
                G.edges[fn, su]["state"] = 0
                failed_links.append((fn, su))
                print("Link", fn, "-", su, "failed by definition due to connecting node", fn, "failure")
            for pr in G.predecessors(fn):
                G.edges[pr, fn]["state"] = 0
                failed_links.append((pr, fn))
                print("Link", pr, "-", fn, "failed by definition due to connecting node", fn, "failure")
        else:
            for ne in G.neighbours(fn):
                G.edges[fn, ne]["state"] = 0
                failed_links.append((fn, ne))
                print("Link", fn, "-", ne, "failed by definition due to connecting node", fn, "failure")
    return G, failed_links


def percolate_links(g, failed_links, reversible=True):
    """Returns the network after failing the selected links. This does not do centrality or flow recalculation."""
    assert isinstance(failed_links, list)
    G = g.copy()
    for fl in failed_links:
        assert fl in list(G.edges())
        if G.is_directed():
            G.edges[fl]["state"] = 0
            if reversible:
                if (fl[1], fl[0]) in G.edges:  # Also fail the link going in opposite direction
                    G.edges[fl[1], fl[0]]["state"] = 0
                    print("Link", fl[1], "-", fl[0], "failed by definition due to reversed link", fl, "failure")
        else:
            G.edges[fl]["state"] = 0
    return G


def recompute_flows(G, newly_failed_nodes, newly_failed_links, shortest_paths):
    """Recompute all flows for network given node and link failures.
    Here, the network contains ONLY nodes/links that are still working (i.e. not failed).
    It is assumed that newly_failed_nodes and newly_failed_links are consistent with each other,
    i.e. both ends of failed links are in failed nodes, and all links adjacent to failed nodes are failed links"""

    # Filter G - we are only recomputing flows for the filtered network (that is still functional)
    Gf = filter_functional_network(G)

    # Split the transport and power parts of the FILTERED network
    GT = G.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "transport"]).copy()
    GP = G.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "power"]).copy()

    # Create new copy of overall network
    new_G = G.copy()

    # TRANSPORT NETWORK RECOMPUTATION

    # Populate affected shortest paths
    affected_shortest_paths = list()
    for fl in newly_failed_links:
        for s in shortest_paths:
            for t in shortest_paths[s]:  # iterate through all shortest paths
                if shortest_paths[s][t]["path"] is not None:  # only if it exists
                    for u, v in zip(shortest_paths[s][t]["path"][:-1], shortest_paths[s][t]["path"][1:]):
                        if fl == (u, v):  # if newly-failed link is along shortest path
                            affected_shortest_paths.append((s, t))

    # Create new copy of shortest_paths dict
    new_shortest_paths = copy.deepcopy(shortest_paths)

    # Track unfulfilled trips and/or calculate new shortest paths
    for sp in affected_shortest_paths:
        (s, t) = sp
        if s in newly_failed_nodes:
            print("Trips between", s, t, "unfulfilled because s failed")
            new_shortest_paths[s][t]["path"] = None
            new_shortest_paths[s][t]["travel_time"] = np.inf
            new_shortest_paths[s][t]["length"] = np.inf
            new_shortest_paths[s][t]["flow"] = 0  # Since all these trips are unfulfilled
        elif t in newly_failed_nodes:
            print("Trips between", s, t, "unfulfilled because t failed")
            new_shortest_paths[s][t]["path"] = None
            new_shortest_paths[s][t]["travel_time"] = np.inf
            new_shortest_paths[s][t]["length"] = np.inf
            new_shortest_paths[s][t]["flow"] = 0  # Since all these trips are unfulfilled
        else:
            try:
                # This implementation reduces the number of Dijkstra runs significantly (only run for affected st-pairs)
                travel_time, path = nx.single_source_dijkstra(GT, source=s, target=t, weight='running_time_min')
                new_shortest_paths[s][t]["path"] = path
                new_shortest_paths[s][t]["travel_time"] = travel_time
                new_shortest_paths[s][t]["length"] = len(new_shortest_paths[s][t]["path"])
                new_shortest_paths[s][t]["flow"] = shortest_paths[s][t]["flow"]  # O-D matrix remains the same
                # print("Trips between", s, t, "rerouted successfully")
            except nx.NetworkXNoPath:  # No path exists between source and target
                print("Trips between", s, t, "unfulfilled because no SP exists between s and t (network disconnected)")
                new_shortest_paths[s][t]["path"] = None
                new_shortest_paths[s][t]["travel_time"] = np.inf
                new_shortest_paths[s][t]["length"] = np.inf
                new_shortest_paths[s][t]["flow"] = 0  # Since all these trips are unfulfilled

    # Adjust flows accordingly
    for sp in affected_shortest_paths:
        (s, t) = sp
        # Remove flows from all links in affected SP
        for u, v in zip(shortest_paths[s][t]["path"][:-1], shortest_paths[s][t]["path"][1:]):
            new_G.edges[u, v]["sp_flow"] -= shortest_paths[s][t]["flow"]
            if new_G.edges[u, v]["sp_flow"] < 0:
                new_G.edges[u, v]["sp_flow"] = 0  # PATCH floating point error when sp_flow subtracted to zero
        # Add flows from all links in new SP, but only if it exists
        if new_shortest_paths[s][t]["path"] is not None:
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
        # PATCH floating point error when sp_flow subtracted to zero, or when edge_current_flow_betweenness < 0
        new_G.edges[u, v]["flow"] = (float(max(new_G.edges[u, v]["sp_flow"], 0)) ** 0.64) * \
                                    (float(max(new_G.edges[u, v]["edge_current_flow_betweenness"], 0)) ** 0.39)
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
    subseq_failed_nodes = list()
    subseq_failed_links = list()

    if not nx.is_weakly_connected(GP):
        for subGP in weakly_connected_component_subgraphs(GP, copy=True):
            # If total supply and inflow cannot meet demand, whole subgrid fails
            total_supply = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if subGP.nodes[n]["type"] in ["GSP_transmission"]])
            total_inflow = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if subGP.nodes[n]["type"] in ["generator"]])
            total_demand = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if subGP.nodes[n]["type"] in ["load"]])
            if total_demand > total_supply + total_inflow:
                # Failed everything here - TODO anything else?
                print("Subgrid with", str(list(subGP.nodes())), "failed deterministically because total demand",
                      total_demand, "> total inflow", total_inflow, "+ total supply", total_supply)
                for n in subGP.nodes():
                    subseq_failed_nodes.append(n)
                    print("Node", n, "failed deterministically due to subgrid supply collapse")
                for u, v in subGP.edges():
                    subseq_failed_links.append((u, v))
                    print("Link", u, "-", v, "failed deterministically due to subgrid supply collapse")
            # If total supply and inflow can meet or exceeds demand, re-balance and redistribute flows
            # Advantage of CFB is that rebalancing the subgrid is equivalent to recalculating CFB
            else:
                subGP = power_calc_centrality(subGP)
                for n in subGP.nodes():
                    new_G.nodes[n]["current_flow_betweenness"] = subGP.nodes[n]["current_flow_betweenness"]
                for u, v in subGP.edges():
                    new_G.edges[u, v]["edge_betweenness"] = subGP.edges[u, v]["edge_betweenness"]
                    new_G.edges[u, v]["edge_current_flow_betweenness"] \
                        = subGP.edges[u, v]["edge_current_flow_betweenness"]
    else:
        # If whole network is still connected, just re-balance and redistribute flows
        GP = power_calc_centrality(GP)
        for n in GP.nodes():
            new_G.nodes[n]["current_flow_betweenness"] = GP.nodes[n]["current_flow_betweenness"]
        for u, v in GP.edges():
            new_G.edges[u, v]["edge_betweenness"] = GP.edges[u, v]["edge_betweenness"]
            new_G.edges[u, v]["edge_current_flow_betweenness"] \
                = GP.edges[u, v]["edge_current_flow_betweenness"]

    # Set flows and thruflows to zero for all failed nodes & links
    for n in newly_failed_nodes:
        new_G.nodes[n]["thruflow"] = 0
    for (u, v) in newly_failed_links:
        new_G.edges[u, v]["flow"] = 0

    return new_G, new_shortest_paths, subseq_failed_nodes, subseq_failed_links


def filter_functional_network(G):
    """Returns a deepcopied functinal subnetwork of the overall network"""
    return G.subgraph([n for n in G.nodes() if G.nodes[n]["state"] == 1]).copy()


def fail_flow(Gn, Gp, cap_lwr_threshold=0.9, cap_upp_threshold=1.5, ratio_lwr_threshold=1.5, ratio_upp_threshold=2.0):
    # Filter Gn & Gp - we are only recomputing flows for the filtered network (that is still functional)
    Gn = filter_functional_network(Gn)
    Gp = filter_functional_network(Gp)

    newly_failed_links_flow = list()
    # Fail stochastically if links are close to or over capacity
    for u, v in Gn.edges():
        if Gn.edges[u, v]["network"] != "physical_interdependency":
            assert (u, v) in Gp.edges()
            if Gn.edges[u, v]["state"] == 1:
                if cap_lwr_threshold <= Gn.edges[u, v]["pct_flow_cap"] < cap_upp_threshold:
                    p = (Gn.edges[u, v]["pct_flow_cap"] - cap_lwr_threshold) / (cap_upp_threshold - cap_lwr_threshold)
                    if random.random() <= p:
                        newly_failed_links_flow.append((u, v))
                        print("Link", u, "-", v, "failed stochastically due to flow being close to capacity",
                              Gn.edges[u, v]["pct_flow_cap"], Gn.edges[u, v]["flow"], Gn.edges[u, v]["flow_cap"])
                elif Gn.edges[u, v]["pct_flow_cap"] >= cap_upp_threshold:
                    newly_failed_links_flow.append((u, v))
                    print("Link", u, "-", v, "failed deterministically due to flow being over capacity",
                          Gn.edges[u, v]["pct_flow_cap"], Gn.edges[u, v]["flow"], Gn.edges[u, v]["flow_cap"])
    # Otherwise, Fail stochastically if there are flow surges
    for u, v in Gn.edges():
        if Gn.edges[u, v]["network"] != "physical_interdependency":
            assert (u, v) in Gp.edges()
            if Gn.edges[u, v]["state"] == 1:
                if Gp.edges[u, v]["state"] == 1:
                    surge_ratio = Gn.edges[u, v]["flow"] / (Gp.edges[u, v]["flow"] + np.spacing(1))
                    if ratio_lwr_threshold <= surge_ratio < ratio_upp_threshold:
                        p = (surge_ratio - ratio_lwr_threshold) / (ratio_upp_threshold - ratio_lwr_threshold)
                        if random.random() <= p:
                            newly_failed_links_flow.append((u, v))
                            print("Link", u, "-", v, "failed stochastically due to flow surge")
                    elif surge_ratio >= ratio_upp_threshold:
                        newly_failed_links_flow.append((u, v))
                        print("Link", u, "-", v, "failed deterministically due to flow surge")

    # TODO for nodes, fail stochastically if thruflows are close to or over capacity
    # TODO if all incoming or all outoging links fail, node fails

    return newly_failed_links_flow


def fail_SIS(Gn, Gp, infection_probability=0.05, recovery_probability=0.01):
    # TODO Not Implemented
    newly_failed_nodes_dif = list()
    return newly_failed_nodes_dif

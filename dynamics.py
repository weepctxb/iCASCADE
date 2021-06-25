import copy

import networkx as nx
import numpy as np
import pandas as pd
import random

from util import weakly_connected_component_subgraphs, transport_calc_centrality, power_calc_centrality, haversine


def get_node(G, network, type=None, mode="random", top=1):
    """Returns a list of nodes in infrastructure node G, to be failed"""

    assert network in ["power", "transport"]
    assert mode in ["random", "degree", "closeness", "betweenness", "thruflow",
                    "thruflow_cap", "thruflow_max", "pct_thruflow_cap"]
    assert 1 <= top <= G.number_of_nodes()

    if network in ["power", "transport"]:
        G = G.subgraph([n for n in G.nodes() if G.nodes[n]["network"] == network]).copy()

    if type is not None:
        if type == "all_substations":
            G = G.subgraph([n for n in G.nodes() if G.nodes[n]["type"] in
                            ["GSP", "substation", "GSP_transmission", "substation_transmission",
                             "substation_traction", "sub_station"]]).copy()
        else:
            G = G.subgraph([n for n in G.nodes() if G.nodes[n]["type"] == type]).copy()

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
    elif mode in ["thruflow", "thruflow_cap", "thruflow_max", "pct_thruflow_cap"]:
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
    assert mode in ["random", "betweenness", "flow", "flow_cap", "flow_max", "pct_flow_cap"]
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
    elif mode in ["flow", "flow_cap", "flow_max", "pct_flow_cap"]:
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
                if (fl[1], fl[0]) in G.edges and G.edges[fl[1], fl[0]]["state"] == 1:
                    # Also fail the link going in opposite direction, ONLY if it is still functional
                    G.edges[fl[1], fl[0]]["state"] = 0
                    failed_links.append((fl[1], fl[0]))
                    print("Link", fl[1], "-", fl[0], "failed by definition due to reversed link", fl, "failure")
        else:
            G.edges[fl]["state"] = 0
    return G, failed_links


def recompute_flows(G, newly_failed_nodes, newly_failed_links, shortest_paths):
    """Recompute all flows for network given node and link failures.
    Here, the network contains ONLY nodes/links that are still working (i.e. not failed).
    It is assumed that newly_failed_nodes and newly_failed_links are consistent with each other,
    i.e. both ends of failed links are in failed nodes, and all links adjacent to failed nodes are failed links"""

    # Filter G - we are only recomputing flows for the filtered network (that is still functional)
    Gf = filter_functional_network(G)

    # Split the transport and power parts of the FILTERED network
    GT = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "transport"]).copy()
    GP = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "power"]).copy()

    # Create new copy of overall network
    new_G = G.copy()

    # Initialise list of nodes and links to subsequently fail
    subseq_failed_nodes = list()
    subseq_failed_links = list()

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
            # print("Trips starting at", s, "unfulfilled because s failed")
            new_shortest_paths[s][t]["path"] = None
            new_shortest_paths[s][t]["travel_time"] = np.inf
            new_shortest_paths[s][t]["length"] = np.inf
            new_shortest_paths[s][t]["flow"] = 0  # Since all these trips are unfulfilled
        elif t in newly_failed_nodes:
            # print("Trips ending at", t, "unfulfilled because t failed")
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
                # print("Trips between", s, t, "unfulfilled because no SP exists between s and t (network disconnected)")
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

    if len(GT) == 0:  # Skip if GP has failed completely
        print("Transport network has failed completely - no functional nodes left")
    else:
        for subGT in weakly_connected_component_subgraphs(GT, copy=True):
            if subGT.number_of_nodes() <= 2:
                # Fail all subnetworks with <= 2 stations
                for n in subGT.nodes():
                    subseq_failed_nodes.append(n)
                    print("Node", n, "failed deterministically due to subnetwork collapse")
                for u, v in subGT.edges():
                    subseq_failed_links.append((u, v))
                    print("Link", u, "-", v, "failed deterministically due to subnetwork collapse")
            else:
                subGT = transport_calc_centrality(subGT, skip=True)  # PATCH SKIP
                for n in subGT.nodes():
                    new_G.nodes[n]["betweenness"] = subGT.nodes[n]["betweenness"]
                    new_G.nodes[n]["closeness"] = subGT.nodes[n]["closeness"]
                    if "current_flow_betweenness" in subGT.nodes[n]:
                        new_G.nodes[n]["current_flow_betweenness"] = subGT.nodes[n]["current_flow_betweenness"]
                for u, v in subGT.edges():
                    if "edge_betweenness" in subGT.edges[u, v]:
                        new_G.edges[u, v]["edge_betweenness"] = subGT.edges[u, v]["edge_betweenness"]
                    if "edge_current_flow_betweenness" in subGT.edges[u, v]:
                        new_G.edges[u, v]["edge_current_flow_betweenness"] = subGT.edges[u, v]["edge_current_flow_betweenness"]

    # Combine into hybrid measure for link flow,
    #  and Assign baseline % capacity utilised
    for u, v in GT.edges():
        # PATCH floating point error when sp_flow subtracted to zero, or when edge_current_flow_betweenness < 0
        # new_G.edges[u, v]["flow"] = (float(max(new_G.edges[u, v]["sp_flow"], 0.)) ** 0.64) * \
        #                             (float(max(new_G.edges[u, v]["edge_current_flow_betweenness"], 0.)) ** 0.39)
        new_G.edges[u, v]["flow"] = max(new_G.edges[u, v]["sp_flow"], 0.)
        new_G.edges[u, v]["pct_flow_cap"] = new_G.edges[u, v]["flow"] / (new_G.edges[u, v]["flow_cap"] + np.spacing(1.0))

    # Assign baseline node thruflows,
    #  and Assign baseline % capacity utilised
    for n in GT.nodes():
        # Calculate capacities as sum of max(flow_in + incoming flows, flow_out + outgoing flows)
        predecessors = list(GT.predecessors(n))
        successors = list(GT.successors(n))
        new_G.nodes[n]["thruflow"] = max(new_G.nodes[n]["flow_in"] + sum([new_G.edges[p, n]["flow"] for p in predecessors]),
                                     new_G.nodes[n]["flow_out"] + sum([new_G.edges[n, s]["flow"] for s in successors]))
        new_G.nodes[n]["pct_thruflow_cap"] = new_G.nodes[n]["thruflow"] / (new_G.nodes[n]["thruflow_cap"] + np.spacing(1.0))

    # POWER NETWORK RECOMPUTATION

    if len(GP) == 0:  # Skip if GP has failed completely
        print("Power network has failed completely - no functional nodes left")
    elif not nx.is_weakly_connected(GP):
        for subGP in weakly_connected_component_subgraphs(GP, copy=True):
            # If subgrid only has one node, it fails too
            subGP_nodes = [n for n in subGP.nodes()]
            if len(subGP_nodes) == 1:
                subseq_failed_nodes.append(subGP_nodes[0])
                print("Node", subGP_nodes[0], "failed deterministically due to subgrid collapse (only node)")
                continue
            # If total supply and inflow cannot meet demand, whole subgrid fails
            total_supply = sum([subGP.nodes[n]["thruflow_max"] for n in subGP.nodes()
                                if subGP.nodes[n]["type"] in ["GSP_transmission"]])
            total_inflow = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if subGP.nodes[n]["type"] in ["generator"]])
            total_demand = sum([subGP.nodes[n]["thruflow"] for n in subGP.nodes()
                                if subGP.nodes[n]["type"] in ["load"]])
            if total_demand > total_supply + total_inflow:
                load_list = [n for n in subGP.nodes() if subGP.nodes[n]["type"] in ["load"]]
                # Load shedding operations - keep failing loads randomly until demand can be met
                while len(load_list) > 0:
                    n_loadshed = load_list.pop(random.randint(0, len(load_list) - 1))
                    subseq_failed_nodes.append(n_loadshed)
                    print("Node", n_loadshed, "failed stochastically due to load shedding",
                          total_demand, "> total inflow", total_inflow, "+ total supply", total_supply)
                    total_demand -= subGP.nodes[n_loadshed]["thruflow"]
                    if total_demand <= total_supply + total_inflow:
                        break
                # PREVIOUSLY - just fail the whole subgrid
                # for n in subGP.nodes():
                #     subseq_failed_nodes.append(n)
                #     print("Node", n, "failed deterministically due to subgrid supply collapse",
                #           "Subgrid =", str(list(subGP.nodes())),
                #           total_demand, "> total inflow", total_inflow, "+ total supply", total_supply)
                # for u, v in subGP.edges():
                #     subseq_failed_links.append((u, v))
                #     print("Link", u, "-", v, "failed deterministically due to subgrid supply collapse",
                #           "Subgrid =", str(list(subGP.nodes())),
                #           total_demand, "> total inflow", total_inflow, "+ total supply", total_supply)
            # If total supply and inflow can meet or exceeds demand, re-balance and redistribute flows
            # Advantage of CFB is that rebalancing the subgrid is equivalent to recalculating CFB
            else:
                subGP = power_calc_centrality(subGP, skip=True)
                for n in subGP.nodes():
                    new_G.nodes[n]["current_flow_betweenness"] = subGP.nodes[n]["current_flow_betweenness"]
                for u, v in subGP.edges():
                    new_G.edges[u, v]["edge_betweenness"] = subGP.edges[u, v]["edge_betweenness"]
                    new_G.edges[u, v]["edge_current_flow_betweenness"] \
                        = subGP.edges[u, v]["edge_current_flow_betweenness"]
    else:
        # If whole network is still connected, just re-balance and redistribute flows
        GP = power_calc_centrality(GP, skip=True)
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
    """Returns a deepcopied functional subnetwork of the overall network"""
    return G.subgraph([n for n in G.nodes() if G.nodes[n]["state"] == 1]).copy()


def fail_flow(Gn, Gp, shortest_paths,
              pow_cap_lwr_threshold=0.6, pow_cap_upp_threshold=1.0, pow_pmax = 1.0,
              trans_cap_lwr_threshold=0.9, trans_cap_upp_threshold=1.0, trans_pmax = 0.2,
              ratio_lwr_threshold=1.5, ratio_upp_threshold=2.0):

    # Filter Gn & Gp - we are only recomputing flows for the filtered network (that is still functional)
    Gn = filter_functional_network(Gn)
    Gp = filter_functional_network(Gp)

    updated_shortest_paths = shortest_paths  # no need to deepcopy since we are just updating within the same iteration

    newly_failed_links_flow = list()
    for u, v in Gn.edges():

        # For power: Fail if links are close to or over capacity
        if Gn.edges[u, v]["network"] == "power":
            assert (u, v) in Gp.edges()
            if Gn.edges[u, v]["state"] == 1:
                if pow_cap_lwr_threshold <= Gn.edges[u, v]["pct_flow_cap"] < pow_cap_upp_threshold:
                    p = pow_pmax * (Gn.edges[u, v]["pct_flow_cap"] - pow_cap_lwr_threshold) \
                        / (pow_cap_upp_threshold - pow_cap_lwr_threshold)
                    if random.random() <= p:
                        newly_failed_links_flow.append((u, v))
                        print("Link", u, "-", v, "failed stochastically due to flow being close to capacity :",
                              "pct_flow_cap =", Gn.edges[u, v]["pct_flow_cap"],
                              "=", Gn.edges[u, v]["flow"], "/", Gn.edges[u, v]["flow_cap"],
                              "old_flow =", Gp.edges[u, v]["flow"],
                              "p =", p)
                elif Gn.edges[u, v]["pct_flow_cap"] >= pow_cap_upp_threshold:
                    p = pow_pmax
                    if random.random() <= p:
                        newly_failed_links_flow.append((u, v))
                        print("Link", u, "-", v, "failed stochastically due to flow being over capacity :",
                              "pct_flow_cap =", Gn.edges[u, v]["pct_flow_cap"],
                              "=", Gn.edges[u, v]["flow"], "/", Gn.edges[u, v]["flow_cap"],
                              "old_flow =", Gp.edges[u, v]["flow"],
                              "p =", p)

        # For transport: Physical fail at lower prob. if links are close to or over capacity
        elif Gn.edges[u, v]["network"] == "transport":
            assert (u, v) in Gp.edges()
            if Gn.edges[u, v]["state"] == 1:
                if trans_cap_lwr_threshold <= Gn.edges[u, v]["pct_flow_cap"] < trans_cap_upp_threshold:
                    p = trans_pmax * (Gn.edges[u, v]["pct_flow_cap"] - trans_cap_lwr_threshold) \
                        / (trans_cap_upp_threshold - trans_cap_lwr_threshold)
                    if random.random() <= p:
                        newly_failed_links_flow.append((u, v))
                        print("Link", u, "-", v, "failed stochastically due to flow being close to capacity :",
                              "pct_flow_cap =", Gn.edges[u, v]["pct_flow_cap"],
                              "=", Gn.edges[u, v]["flow"], "/", Gn.edges[u, v]["flow_cap"],
                              "old_flow =", Gp.edges[u, v]["flow"],
                              "p =", p)
                elif Gn.edges[u, v]["pct_flow_cap"] >= trans_cap_upp_threshold:
                    p = trans_pmax
                    if random.random() <= p:
                        newly_failed_links_flow.append((u, v))
                        print("Link", u, "-", v, "failed stochastically due to flow being over capacity :",
                              "pct_flow_cap =", Gn.edges[u, v]["pct_flow_cap"],
                              "=", Gn.edges[u, v]["flow"], "/", Gn.edges[u, v]["flow_cap"],
                              "old_flow =", Gp.edges[u, v]["flow"],
                              "p =", p)
                    else:
                        # Scale down passenger flows - Any excess passengers will just have their trips unfulfilled
                        # But balance it evenly across all shortest paths containing (u, v)
                        scale_factor = trans_cap_upp_threshold / Gn.edges[u, v]["pct_flow_cap"]
                        for s in updated_shortest_paths:
                            for t in updated_shortest_paths[s]:  # iterate through all shortest paths
                                if updated_shortest_paths[s][t]["path"] is not None:  # only if it exists
                                    for u_, v_ in zip(updated_shortest_paths[s][t]["path"][:-1],
                                                      updated_shortest_paths[s][t]["path"][1:]):
                                        if (u_, v_) == (u, v):  # if overloaded link is along shortest path
                                            updated_shortest_paths[s][t]["flow"] *= scale_factor
                        # Scale down passenger flows and thruflows for G too
                        Gn.edges[u, v]["flow"] = trans_cap_upp_threshold * Gn.edges[u, v]["flow_cap"]

    # Otherwise, Fail stochastically if there are flow surges
    # for u, v in Gn.edges():
    #     if Gn.edges[u, v]["network"] != "physical_interdependency":  # Ignore physical interdependencies
    #         assert (u, v) in Gp.edges()
    #         if Gn.edges[u, v]["state"] == 1:
    #             if Gp.edges[u, v]["state"] == 1 and Gp.edges[u, v]["flow"] > 0.:
    #                 surge_ratio = Gn.edges[u, v]["flow"] / (Gp.edges[u, v]["flow"] + np.spacing(1.0))
    #                 if ratio_lwr_threshold <= surge_ratio < ratio_upp_threshold:
    #                     p = (surge_ratio - ratio_lwr_threshold) / (ratio_upp_threshold - ratio_lwr_threshold)
    #                     if random.random() <= p:
    #                         newly_failed_links_flow.append((u, v))
    #                         print("Link", u, "-", v, "failed stochastically due to flow surge: ",
    #                               "surge ratio =", surge_ratio, "=", Gn.edges[u, v]["flow"], "/", Gp.edges[u, v]["flow"], "p =", p)
    #                 elif surge_ratio >= ratio_upp_threshold:
    #                     newly_failed_links_flow.append((u, v))
    #                     print("Link", u, "-", v, "failed deterministically due to flow surge: ",
    #                           "surge ratio =", surge_ratio, "=", Gn.edges[u, v]["flow"], "/", Gp.edges[u, v]["flow"])

    newly_failed_nodes_flow = list()
    # Fail stochastically if nodes are close to or over capacity
    for n in Gn.nodes():
        if Gn.nodes[n]["network"] == "power":
            if Gn.nodes[n]["state"] == 1:
                if pow_cap_lwr_threshold <= Gn.nodes[n]["pct_thruflow_cap"] < pow_cap_upp_threshold:
                    p = pow_pmax * (Gn.nodes[n]["pct_thruflow_cap"] - pow_cap_lwr_threshold) \
                        / (pow_cap_upp_threshold - pow_cap_lwr_threshold)
                    if random.random() <= p:
                        newly_failed_nodes_flow.append(n)
                        print("Node", n, "failed stochastically due to thruflow being close to capacity :",
                              "thruflow =", Gn.nodes[n]["thruflow"],
                              "pct_thruflow_cap =", Gn.nodes[n]["pct_thruflow_cap"],
                              "p =", p)
                elif Gn.nodes[n]["pct_thruflow_cap"] >= pow_cap_upp_threshold:
                    p = pow_pmax
                    if random.random() <= p:
                        newly_failed_nodes_flow.append(n)
                        print("Node", n, "failed stochastically due to thruflow being over capacity :",
                              "thruflow =", Gn.nodes[n]["thruflow"],
                              "pct_thruflow_cap =", Gn.nodes[n]["pct_thruflow_cap"],
                              "p =", p)

        elif Gn.nodes[n]["network"] == "transport":
            if Gn.nodes[n]["state"] == 1:
                if trans_cap_lwr_threshold <= Gn.nodes[n]["pct_thruflow_cap"] < trans_cap_upp_threshold:
                    p = trans_pmax * (Gn.nodes[n]["pct_thruflow_cap"] - trans_cap_lwr_threshold) \
                        / (trans_cap_upp_threshold - trans_cap_lwr_threshold)
                    if random.random() <= p:
                        newly_failed_nodes_flow.append(n)
                        print("Node", n, "failed stochastically due to thruflow being close to capacity :",
                              "thruflow =", Gn.nodes[n]["thruflow"],
                              "pct_thruflow_cap =", Gn.nodes[n]["pct_thruflow_cap"],
                              "p =", p)
                elif Gn.nodes[n]["pct_thruflow_cap"] >= trans_cap_upp_threshold:
                    p = trans_pmax
                    if random.random() <= p:
                        newly_failed_nodes_flow.append(n)
                        print("Node", n, "failed deterministically due to thruflow being over capacity :",
                              "thruflow =", Gn.nodes[n]["thruflow"],
                              "pct_thruflow_cap =", Gn.nodes[n]["pct_thruflow_cap"],
                              "p =", p)
                    else:
                        Gn.nodes[n]["thruflow"] = trans_cap_upp_threshold * Gn.nodes[n]["thruflow_cap"]
                        # Any excess passengers will have their trips unfulfilled
                        # TODO anything else to updated_shortest_paths?

    # Otherwise, Fail stochastically if there are thruflow surges
    # for n in Gn.nodes():
    #     if Gn.nodes[n]["state"] == 1:
    #         if Gp.nodes[n]["state"] == 1 and Gp.nodes[n]["thruflow"] > 0.:
    #             surge_ratio = Gn.nodes[n]["thruflow"] / (Gp.nodes[n]["thruflow"] + np.spacing(1.0))
    #             if ratio_lwr_threshold <= surge_ratio < ratio_upp_threshold:
    #                 p = (surge_ratio - ratio_lwr_threshold) / (ratio_upp_threshold - ratio_lwr_threshold)
    #                 if random.random() <= p:
    #                     newly_failed_nodes_flow.append(n)
    #                     print("Node", n, "failed stochastically due to thruflow surge: ",
    #                           "surge ratio =", surge_ratio,
    #                           "=", Gn.nodes[n]["thruflow"], "/", Gp.nodes[n]["thruflow"], "p =", p)
    #             elif surge_ratio >= ratio_upp_threshold:
    #                 newly_failed_nodes_flow.append(n)
    #                 print("Node", n, "failed deterministically due to thruflow surge: ",
    #                       "surge ratio =", surge_ratio,
    #                       "=", Gn.nodes[n]["thruflow"], "/", Gp.nodes[n]["thruflow"])

    return newly_failed_nodes_flow, newly_failed_links_flow, updated_shortest_paths


def fail_SI(Gn, infection_probability=0.05, recovery_probability=0.01, geo_threshold=0.1, geo_probability=0.05):
    newly_failed_nodes_dif = list()
    for n in Gn.nodes():
        if Gn.nodes[n]["state"] == 1:  # If node is working
            neighbours = set([p for p in Gn.predecessors(n)] + [s for s in Gn.successors(n)])
            failed_ne = [ne for ne in neighbours if Gn.nodes[ne]["state"] == 0]
            if len(failed_ne) > 0:  # only if there are neighbours
                p = 1. - (1. - infection_probability) ** len(failed_ne)  # assuming parallel hazards
                if random.random() <= p:
                    newly_failed_nodes_dif.append(n)
                    print("Node", n, "failed stochastically due to failure diffusion from", str(failed_ne), "p =", p)

    # Physical interdependencies - directional
    for n in Gn.nodes():
        if Gn.nodes[n]["state"] == 0:  # If node is not working
            for s in Gn.successors(n):
                if Gn.edges[n, s]["network"] == "physical_interdependency":
                    intdp = len([ps for ps in Gn.predecessors(s)
                                 if Gn.edges[ps, s]["network"] == "physical_interdependency"])
                    p = 1. / max(1., intdp)
                    if random.random() <= p:
                        newly_failed_nodes_dif.append(s)
                        print("Node", s, "failed stochastically due to interdependency with", n, "p =", p)

    # Geographical interdependencies - non-directional
    for n in Gn.nodes():
        if Gn.nodes[n]["state"] == 0:  # If node is not working
            for m in Gn.nodes():
                if m != n and Gn.nodes[n]["network"] != Gn.nodes[m]["network"]:
                    d = haversine(Gn.nodes[n]["pos"][1], Gn.nodes[n]["pos"][0],
                                  Gn.nodes[m]["pos"][1], Gn.nodes[m]["pos"][0])
                    if d < geo_threshold:
                        # We assume a inverse relationship with proximity: p \propto 1/(1+d)
                        #  - Naturally, when d -> \inf, p -> 0, and when d -> 0, p -> pgeo
                        p = geo_probability / (1. + d)
                        if random.random() <= p:
                            newly_failed_nodes_dif.append(m)
                            print("Node", m, "failed stochastically due to co-located failure diffusion from", str(n),
                                  "d =", d, "p =", p)

    return newly_failed_nodes_dif

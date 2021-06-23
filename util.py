import networkx as nx
import numpy as np
import copy

from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

from globalparams import POWER_LINE_PP_EPS, TRAIN_DUR_THRESHOLD_MIN


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) + [255]


def parse_json(shortest_paths):
    for s in list(shortest_paths.keys()):
        if not isinstance(s, int):  # PATCH - keys in json are in strings for some reason (JSON encoder issue)
            try:
                shortest_paths[int(s)] = copy.deepcopy(shortest_paths[s])
                shortest_paths.pop(s)
            except ValueError as e:
                continue
    for s in list(shortest_paths.keys()):
        for t in list(shortest_paths[s].keys()):
            if not isinstance(t, int):  # PATCH - keys in json are in strings for some reason (JSON encoder issue)
                try:
                    shortest_paths[s][int(t)] = copy.deepcopy(shortest_paths[s][t])
                    shortest_paths[s].pop(t)
                except ValueError as e:
                    continue
    return shortest_paths


def haversine(lon1, lat1, lon2, lat2):
    """Calculates distance between two sets of coordinates"""
    # https://gist.github.com/Tofull/49fbb9f3661e376d2fe08c2e9d64320e

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r  # Returns distance in km


def perpendicular_dist(x, y, x1, y1, x2, y2):
    """Calculates perpendicular distance between a point and a line represented by two points"""
    # https://stackoverflow.com/a/6853926/5881159

    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C ** 2 + D ** 2
    param = dot / len_sq if len_sq != 0 else -1

    xx = x1 if param < 0 else x2 if param > 1 else x1 + param * C
    yy = y1 if param < 0 else y2 if param > 1 else y1 + param * D
    dx = x - xx
    dy = y - yy

    return sqrt(dx ** 2 + dy ** 2)


def douglas_peucker(linenodes, eps=POWER_LINE_PP_EPS):
    """Simplifies multi-line geometry"""
    # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm#Pseudocode

    linenodes.reset_index(inplace=True, drop=True)

    # Find point with maximum perpendicular distance from line between first & last nodes
    dmax = -1.
    index = -1
    firstnode = linenodes.head(1)
    lastnode = linenodes.tail(1)
    for i, linenode in linenodes[1:-1].iterrows():
        d = perpendicular_dist(linenode["$x"], linenode["$y"],
                               linenodes.at[0, "$x"], linenodes.at[0, "$y"],
                               linenodes.at[len(linenodes)-1, "$x"], linenodes.at[len(linenodes)-1, "$y"])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > eps:
        linenodes_res_1 = douglas_peucker(linenodes[:index], eps=eps)
        linenodes_res_2 = douglas_peucker(linenodes[index:], eps=eps)
        linenodes_res = linenodes_res_1[:-1].append(linenodes_res_2)
    else:
        linenodes_res = firstnode.append(lastnode)

    return linenodes_res


def weakly_connected_component_subgraphs(G, copy=True):
    """[Copied from deprecated version of networkx library] Generate weakly connected components as subgraphs.
    For directed graphs only. Graph, node, and edge attributes are copied to the subgraphs by default.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph.

    copy: bool (default=True)
        If True make a copy of the graph attributes

    Returns
    -------
    comp : generator
        A generator of graphs, one for each weakly connected component of G.
    """
    for comp in nx.weakly_connected_components(G):
        if copy:
            yield G.subgraph(comp).copy()
        else:
            yield G.subgraph(comp)


def populate_shortest_paths(G, odmat):
    """Populate shortest paths for train network"""

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

    return shortest_paths


def populate_shortest_paths_skele(G, odmat):
    """Populate shortest paths for skeletonised train network"""

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
        for t in shortest_paths[s]:
            shortest_paths[s][t]["flow"] = 0.
        # PATCH - keys in json are in strings for some reason (JSON encoder issue)
        s_list = s.split("-") if isinstance(s, str) else [s]
        snlc = G.nodes[s]["NLC"]
        snlc_list = snlc.split("-") if isinstance(s, str) else [snlc]
        for s_, snlc_ in zip(s_list, snlc_list):
            for t in shortest_paths[s]:
                t_list = t.split("-") if isinstance(t, str) else [t]
                tnlc = G.nodes[t]["NLC"]
                tnlc_list = tnlc.split("-") if isinstance(t, str) else [tnlc]
                for t_, tnlc_ in zip(t_list, tnlc_list):
                    odmat_filtered = odmat.loc[odmat["mnlc_o"] == int(snlc_)]
                    odmat_filtered = odmat_filtered.loc[odmat["mnlc_d"] == int(tnlc_)]
                    flow_st = sum(odmat_filtered["od_tb_3_perhour"])
                    shortest_paths[s][t]["flow"] += flow_st

    return shortest_paths


def linear_cluster(G, nodes_to_cluster):
    # Nomenclature: (rest of network-x)-a-b-c-(y-rest of network)

    G1 = G.copy()  # deepcopy
    nnew = "-".join([str(n) for n in nodes_to_cluster])

    nb_left = [n for n in G1.neighbors(nodes_to_cluster[0]) if n != nodes_to_cluster[1]]
    nb_right = [n for n in G1.neighbors(nodes_to_cluster[-1]) if n != nodes_to_cluster[-2]]

    G1.add_node(nnew,
                pos=(np.average([G1.nodes[n]["pos"][0] for n in nodes_to_cluster]),
                     np.average([G1.nodes[n]["pos"][1] for n in nodes_to_cluster])),
                nodeLabel="-".join([G1.nodes[n]["nodeLabel"] for n in nodes_to_cluster]),
                interchange=False,
                line="-".join(np.unique([G1.nodes[n]["line"] for n in nodes_to_cluster])),
                NLC="-".join([str(G1.nodes[n]["NLC"]) for n in nodes_to_cluster]),
                type="station",
                flow_in=sum([G1.nodes[n]["flow_in"] for n in nodes_to_cluster]),
                flow_out=sum([G1.nodes[n]["flow_out"] for n in nodes_to_cluster]),
                thruflow=G1.edges[nb_left[-1], nodes_to_cluster[0]]["flow"] +
                         G1.edges[nodes_to_cluster[-1], nb_right[0]]["flow"],
                thruflow_cap=sum([G1.nodes[n]["flow_in"] for n in nodes_to_cluster]) +
                             sum([G1.nodes[n]["flow_out"] for n in nodes_to_cluster]) +
                             (G1.edges[nb_left[-1], nodes_to_cluster[0]]["flow_cap"] +
                             G1.edges[nodes_to_cluster[0], nb_left[-1]]["flow_cap"] +
                             G1.edges[nodes_to_cluster[-1], nb_right[0]]["flow_cap"] +
                             G1.edges[nb_right[0], nodes_to_cluster[-1]]["flow_cap"]) / 2,
                pct_thruflow_cap=(G1.edges[nb_left[-1], nodes_to_cluster[0]]["flow"] +
                                  G1.edges[nodes_to_cluster[-1], nb_right[0]]["flow"]) /
                                 (G1.edges[nb_left[-1], nodes_to_cluster[0]]["flow_cap"] +
                                  G1.edges[nodes_to_cluster[-1], nb_right[0]]["flow_cap"])
                )

    for nleft in nb_left:  # actually it will always be the case that len(nb_left) == 1
        G1.add_edge(nleft, nnew,
                    Line="-".join(np.unique([G1.edges[n1, n2]["Line"]
                                             for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])])),
                    StationA=G1.nodes[nleft]["nodeLabel"], StationB=G1.nodes[nnew]["nodeLabel"],
                    Distance=G1.edges[nleft, nodes_to_cluster[0]]["Distance"] +
                             sum([G1.edges[n1, n2]["Distance"]
                                  for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    running_time_min=G1.edges[nleft, nodes_to_cluster[0]]["running_time_min"] +
                                     sum([G1.edges[n1, n2]["running_time_min"]
                                          for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    flow=G1.edges[nleft, nodes_to_cluster[0]]["flow"],
                    sp_flow=G1.edges[nleft, nodes_to_cluster[0]]["sp_flow"],
                    # flow_capscal=G1.edges[nleft, nodes_to_cluster[0]]["flow_capscal"],
                    flow_cap=G1.edges[nleft, nodes_to_cluster[0]]["flow_cap"],
                    pct_flow_cap=G1.edges[nleft, nodes_to_cluster[0]]["flow"] /
                                 G1.edges[nleft, nodes_to_cluster[0]]["flow_cap"]
                    )
        G1.add_edge(nnew, nleft,
                    Line="-".join(np.unique([G1.edges[n1, n2]["Line"]
                                             for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])])),
                    StationA=G1.nodes[nnew]["nodeLabel"], StationB=G1.nodes[nleft]["nodeLabel"],
                    Distance=G1.edges[nodes_to_cluster[0], nleft]["Distance"] +
                             sum([G1.edges[n1, n2]["Distance"]
                                  for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    running_time_min=G1.edges[nodes_to_cluster[0], nleft]["running_time_min"] +
                                     sum([G1.edges[n1, n2]["running_time_min"]
                                          for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    flow=G1.edges[nodes_to_cluster[0], nleft]["flow"],
                    sp_flow=G1.edges[nodes_to_cluster[0], nleft]["sp_flow"],
                    # flow_capscal=G1.edges[nodes_to_cluster[0], nleft]["flow_capscal"],
                    flow_cap=G1.edges[nodes_to_cluster[0], nleft]["flow_cap"],
                    pct_flow_cap=G1.edges[nodes_to_cluster[0], nleft]["flow"] /
                                 G1.edges[nodes_to_cluster[0], nleft]["flow_cap"]
                    )
    for nright in nb_right:  # actually it will always be the case that len(nb_right) == 1
        G1.add_edge(nnew, nright,
                    Line="-".join(np.unique([G1.edges[n1, n2]["Line"]
                                             for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])])),
                    StationA=G1.nodes[nnew]["nodeLabel"], StationB=G1.nodes[nright]["nodeLabel"],
                    Distance=G1.edges[nodes_to_cluster[-1], nright]["Distance"] +
                             sum([G1.edges[n1, n2]["Distance"]
                                  for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    running_time_min=G1.edges[nodes_to_cluster[-1], nright]["running_time_min"] +
                                     sum([G1.edges[n1, n2]["running_time_min"]
                                          for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    flow=G1.edges[nodes_to_cluster[-1], nright]["flow"],
                    sp_flow=G1.edges[nodes_to_cluster[-1], nright]["sp_flow"],
                    # flow_capscal=G1.edges[nodes_to_cluster[-1], nright]["flow_capscal"],
                    flow_cap=G1.edges[nodes_to_cluster[-1], nright]["flow_cap"],
                    pct_flow_cap=G1.edges[nodes_to_cluster[-1], nright]["flow"] /
                                 G1.edges[nodes_to_cluster[-1], nright]["flow_cap"]
                    )
        G1.add_edge(nright, nnew,
                    Line="-".join(np.unique([G1.edges[n1, n2]["Line"]
                                             for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])])),
                    StationA=G1.nodes[nright]["nodeLabel"], StationB=G1.nodes[nnew]["nodeLabel"],
                    Distance=G1.edges[nright, nodes_to_cluster[-1]]["Distance"] +
                             sum([G1.edges[n1, n2]["Distance"]
                                  for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    running_time_min=G1.edges[nright, nodes_to_cluster[-1]]["running_time_min"] +
                                     sum([G1.edges[n1, n2]["running_time_min"]
                                          for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:])]) / 2,
                    flow=G1.edges[nright, nodes_to_cluster[-1]]["flow"],
                    sp_flow=G1.edges[nright, nodes_to_cluster[-1]]["sp_flow"],
                    # flow_capscal=G1.edges[nright, nodes_to_cluster[-1]]["flow_capscal"],
                    flow_cap=G1.edges[nright, nodes_to_cluster[-1]]["flow_cap"],
                    pct_flow_cap=G1.edges[nright, nodes_to_cluster[-1]]["flow"] /
                                 G1.edges[nright, nodes_to_cluster[-1]]["flow_cap"]
                    )
    for n1, n2 in zip(nodes_to_cluster[:-1], nodes_to_cluster[1:]):
        G1.remove_edge(n1, n2)
        G1.remove_edge(n2, n1)
    for n in nodes_to_cluster:
        G1.remove_node(n)

    return G1


def transport_calc_centrality(G, skip=False):
    """Calculate and assign transport network centralities by running time"""
    assert G.is_directed()

    bb = dict()
    cc = dict()
    cfb = dict()
    eb = dict()
    ecfb = dict()

    for subG in weakly_connected_component_subgraphs(G, copy=True):
        subG_undir = transport_to_undirected(subG)

        bb.update(nx.betweenness_centrality(subG, normalized=True, weight="running_time_min"))
        cc.update(nx.closeness_centrality(subG, distance="running_time_min"))
        eb.update(nx.edge_betweenness_centrality(subG, normalized=True, weight="running_time_min"))

        # Do NOT normalise because network may break up into disconnected components -
        #  but surrogate flows can still be estimated

        # cfb.update(nx.current_flow_betweenness_centrality(
        #     subG_undir, normalized=False, weight="recip_running_time_min"))

        if not skip:  # PATCH SKIP
            cfb.update(nx.approximate_current_flow_betweenness_centrality(
                subG_undir, normalized=False, weight="recip_running_time_min", epsilon=0.5, kmax=10000))
            # Do NOT normalise because network may break up into disconnected components -
            #  but surrogate flows can still be estimated
            ecfb.update(nx.edge_current_flow_betweenness_centrality(
                subG_undir, normalized=False, weight="recip_running_time_min"))

    nx.set_node_attributes(G, bb, 'betweenness')
    nx.set_node_attributes(G, cc, 'closeness')
    nx.set_edge_attributes(G, eb, 'edge_betweenness')

    if not skip:  # PATCH SKIP
        nx.set_node_attributes(G, cfb, 'current_flow_betweenness')
        for u, v in G.edges():
            try:
                G.edges[u, v]["edge_current_flow_betweenness"] = ecfb[(u, v)]
            except Exception as e:
                G.edges[u, v]["edge_current_flow_betweenness"] = ecfb[(v, u)]  # PATCH

    return G


def transport_to_undirected(G):
    """Further simplifies directed transport network to undirected network"""

    if not nx.is_directed(G):
        return G

    G_undir = nx.create_empty_copy(G, with_data=True).to_undirected()
    for u, v in G.edges():
        G_undir.add_edge(u, v)

        for rev_attr in ["StationA", "StationB", "Line", "Distance",
                         "edge_betweenness", "edge_current_flow_betweenness"]:
            if rev_attr in G.edges[u, v]:
                G_undir.edges[u, v][rev_attr] = G.edges[u, v][rev_attr]

        for sum_attr in ["sp_flow", "flow", "flow_cap"]:
            if sum_attr in G.edges[u, v]:
                if G.has_edge(v, u):  # i.e. both directions exist
                    G_undir.edges[u, v][sum_attr] = G.edges[u, v][sum_attr] + G.edges[v, u][sum_attr]
                else:  # only one direction exists
                    G_undir.edges[u, v][sum_attr] = G.edges[u, v][sum_attr]

        for avg_attr in ["running_time_min", "recip_running_time_min", "Distance", "rel_error"]:
            if avg_attr in G.edges[u, v]:
                if G.has_edge(v, u):  # i.e. both directions exist
                    G_undir.edges[u, v][avg_attr] = np.nanmean([G.edges[u, v][avg_attr], G.edges[v, u][avg_attr]])
                else:  # only one direction exists
                    G_undir.edges[u, v][avg_attr] = G.edges[u, v][avg_attr]

    for u, v in G.edges():
        # Only if flow has already been calculated
        if "flow" in G_undir.edges[u, v] and "flow_cap" in G_undir.edges[u, v]:
            G_undir.edges[u, v]["pct_flow_cap"] = G_undir.edges[u, v]["flow"] / G_undir.edges[u, v]["flow_cap"]

    # TODO Run transport_calc_centrality later on at each iteration of the simulation

    return G_undir


def transport_compare_flow_dur_distribution():
    """Check fit with power law, and fit between estimated & actual flows & travel durations"""

    # FYI Distribution of trip durations won't match exactly because we are comparing (only) the
    #  train travel timings with stopping (w/o transfer or waiting times) vs. actual measured durations
    #  But this is not a major cause of concern because we only use the trip durations as a weight measure
    #  for shortest path calculations.

    import json
    import pandas as pd
    import matplotlib.pyplot as plt

    from globalparams import TRAIN_DUR_THRESHOLD_MIN

    # Estimated flows - all s-t pairs
    with open(r'data/transport_multiplex/flow/shortest_paths.json', 'r') as json_file:
        shortest_paths = json.load(json_file)

    est_flow_dist = list()

    for s in shortest_paths:
        for t in shortest_paths[s]:
            for _ in range(round(shortest_paths[s][t]["flow"])):
                if shortest_paths[s][t]["travel_time"] <= TRAIN_DUR_THRESHOLD_MIN:
                    est_flow_dist.append(shortest_paths[s][t]["travel_time"])

    # Actual flows - all s-t pairs
    travel_times = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx",
                                 sheet_name="power_law_check", header=0)
    actual_flow_dist = list()

    for index, row in travel_times.iterrows():
        for _ in range(round(row["AM_peak_perhour"])):
            try:
                actual_flow_dist.append(row["nonzero_median_MIN"])
            except Exception:
                continue

    an, abins, apatches = plt.hist(actual_flow_dist, bins=100)
    en, ebins, epatches = plt.hist(est_flow_dist, bins=100)
    plt.clf()
    plt.loglog(abins[:-1], an[:], "b.")
    plt.loglog(ebins[:-1], en[:], "r.")
    plt.title("Distribution of trip duration (actual and estimated) within London")
    plt.xlabel('Trip duration (min)')
    plt.ylabel('No. of trips (unique)')
    plt.show()


def power_calc_centrality(G, skip=False):
    """Calculate and assign centralities by conductance"""
    assert G.is_directed()

    cfb = dict()
    cfb_max = dict()
    eb = dict()
    ecfb = dict()
    ecfb_max = dict()

    source_nodes = [n for n in G.nodes() if
                    (G.nodes[n]["type"] == "generator" or G.nodes[n]["type"] == "GSP_transmission")]
    target_nodes = [n for n in G.nodes() if
                    G.nodes[n]["type"] == "load"]

    for subG in weakly_connected_component_subgraphs(G, copy=True):
        subG_undir = power_to_undirected(subG)

        if not skip:  # PATCH SKIP
            cfb.update(custom_cfb(
                subG_undir, normalized=False, weight="conductance",
                sources=source_nodes, targets=target_nodes, solver="full", max=False))

        if not skip:  # PATCH SKIP
            cfb_max.update(custom_cfb(
                subG_undir, normalized=False, weight="conductance",
                sources=source_nodes, targets=target_nodes, solver="full", max=True))

        eb.update(nx.edge_betweenness_centrality(subG, normalized=True, weight="resistance"))

        ecfb.update(custom_ecfb(
            subG_undir, normalized=False, weight="conductance",
            sources=source_nodes, targets=target_nodes, solver="full", max=False))

        if not skip:  # PATCH SKIP
            ecfb_max.update(custom_ecfb(
                subG_undir, normalized=False, weight="conductance",
                sources=source_nodes, targets=target_nodes, solver="full", max=True))

    if not skip:  # PATCH SKIP
        nx.set_node_attributes(G, cfb, 'current_flow_betweenness')
        nx.set_node_attributes(G, cfb_max, 'current_flow_betweenness_max')

    nx.set_edge_attributes(G, eb, 'edge_betweenness')
    for u, v in G.edges():
        try:
            G.edges[u, v]["edge_current_flow_betweenness"] = max(np.spacing(1.0), ecfb[(u, v)])
        except Exception as e:
            G.edges[u, v]["edge_current_flow_betweenness"] = max(np.spacing(1.0), ecfb[(v, u)])  # PATCH reversible links
    if not skip:  # PATCH SKIP
        for u, v in G.edges():
            try:
                G.edges[u, v]["edge_current_flow_betweenness_max"] = max(np.spacing(1.0), ecfb_max[(u, v)])
            except Exception as e:
                G.edges[u, v]["edge_current_flow_betweenness_max"] = max(np.spacing(1.0), ecfb_max[(v, u)])  # PATCH reversible links

    return G


def custom_ecfb(G, sources, targets, normalized=True, weight=None, dtype=float, solver="lu", max=False, mode="slow"):
    """Custom implementation of nx.edge_current_flow_betweenness_centrality_subset"""

    if mode == "slow":

        from networkx.utils import reverse_cuthill_mckee_ordering
        from networkx.algorithms.centrality.flow_matrix import flow_matrix_row

        if not nx.is_connected(G):
            raise nx.NetworkXError("Graph not connected.")
        n = G.number_of_nodes()
        ordering = list(reverse_cuthill_mckee_ordering(G))
        # make a copy with integer labels according to rcm ordering
        # this could be done without a copy if we really wanted to
        mapping = dict(zip(ordering, range(n)))
        H = nx.relabel_nodes(G, mapping)
        edges = (tuple(sorted((u, v))) for u, v in H.edges())
        betweenness = dict.fromkeys(edges, 0.0)
        if normalized:
            nb = (n - 1.0) * (n - 2.0)  # normalization factor
        else:
            nb = 2.0
        for row, (e) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
            for ss in sources:
                i = mapping[ss]
                for tt in targets:
                    j = mapping[tt]
                    if max:  # generator at nameplate capacity, transmission GSP at max, substation at maximum demand
                        betweenness[e] += 0.5 * np.abs(row[i] - row[j]) * \
                                          sqrt(G.nodes[ss]["thruflow_cap"] * G.nodes[tt]["thruflow_cap"])
                    else:  # generator at average capacity, transmission GSP at max, substation at average demand
                        betweenness[e] += 0.5 * np.abs(row[i] - row[j]) * \
                                          sqrt(G.nodes[ss]["thruflow_cap"] * \
                                               (G.nodes[ss]["Capacity_Factor"]
                                                if G.nodes[ss]["type"] == "generator" else 1.) * \
                                               G.nodes[tt]["thruflow_cap"] * \
                                               (1 + G.nodes[tt]["Minimum_Load_Scaling_Factor_winter"]) / 2)
            betweenness[e] /= nb
        return {(ordering[s], ordering[t]): v for (s, t), v in betweenness.items()}

    else:
        raise NotImplementedError


def custom_cfb(G, sources, targets, normalized=True, weight=None, dtype=float, solver="lu", max=False, mode="slow"):
    """Custom implementation of nx.current_flow_betweenness_centrality_subset"""

    if mode == "slow":

        from networkx.utils import reverse_cuthill_mckee_ordering
        from networkx.algorithms.centrality.flow_matrix import flow_matrix_row

        if not nx.is_connected(G):
            raise nx.NetworkXError("Graph not connected.")
        n = G.number_of_nodes()
        ordering = list(reverse_cuthill_mckee_ordering(G))
        # make a copy with integer labels according to rcm ordering
        # this could be done without a copy if we really wanted to
        mapping = dict(zip(ordering, range(n)))
        H = nx.relabel_nodes(G, mapping)
        betweenness = dict.fromkeys(H, 0.0)  # b[v]=0 for v in H
        for row, (s, t) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
            for ss in sources:
                i = mapping[ss]
                for tt in targets:
                    j = mapping[tt]
                    if max:  # generator at nameplate capacity, transmission GSP at max, substation at maximum demand
                        betweenness[s] += 0.5 * np.abs(row[i] - row[j]) * \
                                          sqrt(G.nodes[ss]["thruflow_cap"] * G.nodes[tt]["thruflow_cap"])
                        betweenness[t] += 0.5 * np.abs(row[i] - row[j]) * \
                                          sqrt(G.nodes[ss]["thruflow_cap"] * G.nodes[tt]["thruflow_cap"])
                    else:  # generator at average capacity, transmission GSP at max, substation at average demand
                        betweenness[s] += 0.5 * np.abs(row[i] - row[j]) * \
                                          sqrt(G.nodes[ss]["thruflow_cap"] *
                                               (G.nodes[ss]["Capacity_Factor"]
                                                if G.nodes[ss]["type"] == "generator" else 1.) * \
                                               G.nodes[tt]["thruflow_cap"] * \
                                               (1 + G.nodes[tt]["Minimum_Load_Scaling_Factor_winter"]) / 2)
                        betweenness[t] += 0.5 * np.abs(row[i] - row[j]) * \
                                          sqrt(G.nodes[ss]["thruflow_cap"] *
                                               (G.nodes[ss]["Capacity_Factor"]
                                                if G.nodes[ss]["type"] == "generator" else 1.) * \
                                               G.nodes[tt]["thruflow_cap"] * \
                                               (1 + G.nodes[tt]["Minimum_Load_Scaling_Factor_winter"]) / 2)
        if normalized:
            nb = (n - 1.0) * (n - 2.0)  # normalization factor
        else:
            nb = 2.0
        for v in H:
            if n > 2:  # PATCH
                betweenness[v] = betweenness[v] / nb + 1.0 / (2 - n)
            elif n == 2:  # PATCH
                betweenness[v] = betweenness[v] / nb
        return {ordering[k]: v for k, v in betweenness.items()}

    else:
        raise NotImplementedError


def power_to_undirected(G):
    """Further simplifies directed power network to undirected network"""

    if not nx.is_directed(G):
        return G

    G_undir = nx.create_empty_copy(G, with_data=True).to_undirected()
    for u, v in G.edges():
        G_undir.add_edge(u, v)

        for rev_attr in ["Line Name", "GSP", "Circuit Length km", "Operating Voltage kV",
                         "Rating Amps Summer", "Rating Amps Winter",
                         "G_base", "G_base_HV_LV1", "G_base_HV_LV2",
                         "Voltage HV kV", "Voltage LV kV", "Voltage LV1 kV", "Voltage LV2 kV",
                         "Transformer Rating Summer MVA", "Transformer Rating Winter MVA",
                         "Transformer Rating MVA Summer HV",
                         "Transformer Rating MVA Summer LV1", "Transformer Rating MVA Summer LV2",
                         "Transformer Rating MVA Winter HV",
                         "Transformer Rating MVA Winter LV1", "Transformer Rating MVA Winter LV2",
                         "edge_betweenness", "edge_current_flow_betweenness"]:
            if rev_attr in G.edges[u, v]:
                G_undir.edges[u, v][rev_attr] = G.edges[u, v][rev_attr]

        for sum_attr in ["flow", "flow_cap", "flow_max"]:
            if sum_attr in G.edges[u, v]:
                if G.has_edge(v, u):  # i.e. both directions exist
                    G_undir.edges[u, v][sum_attr] = G.edges[u, v][sum_attr] + G.edges[v, u][sum_attr]
                else:  # only one direction exists
                    G_undir.edges[u, v][sum_attr] = G.edges[u, v][sum_attr]

        for avg_attr in ["conductance", "resistance"]:
            if avg_attr in G.edges[u, v]:
                if G.has_edge(v, u):  # i.e. both directions exist
                    G_undir.edges[u, v][avg_attr] = np.nanmean([G.edges[u, v][avg_attr], G.edges[v, u][avg_attr]])
                else:  # only one direction exists
                    G_undir.edges[u, v][avg_attr] = G.edges[u, v][avg_attr]

    for u, v in G.edges():
        # Only if flow has already been calculated
        if "flow" in G_undir.edges[u, v] and "flow_max" in G_undir.edges[u, v]:
            G_undir.edges[u, v]["pct_flow_cap"] = G_undir.edges[u, v]["flow"] / G_undir.edges[u, v]["flow_max"]

    # TODO Run power_calc_centrality later on at each iteration of the simulation

    return G_undir


def network_plot_3D(G, angle):

    from mpl_toolkits.mplot3d import Axes3D

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(n) for n in G.nodes()])
    # Define color range proportional to number of edges adjacent to a single node
    colors = {n: plt.cm.plasma(G.degree(n) / edge_max) for n in G.nodes()}
    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, color=colors[key], s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, color='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    plt.show()

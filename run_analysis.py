import json
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dynamics import filter_functional_network
from globalparams import STATE_COLORS
from util import weakly_connected_component_subgraphs

# TODO Augment code to support combined plotting of multiple runs
G = pickle.load(open('data/combined_network/cobourg_example/infra_dynG_06232021_1621_[\'Cobourg Street Electricity Substation\'].pkl', "rb"))

with open('data/combined_network/cobourg_example/shortest_paths_06232021_1621_[\'Cobourg Street Electricity Substation\'].json', 'rb') as json_file:
    shortest_paths = json.load(json_file)

nodes_t_functional = list()  # TODO convert to fraction of nodes still functional
nodes_p_functional = list()
links_t_functional = list()  # TODO convert to fraction of links still functional
links_p_functional = list()
num_t_sub = list()
size_t_sub = list()
num_p_sub = list()
size_p_sub = list()
wei_avg_time = list()
fulfilled_trips = list()
unfulfilled_trips = list()
fulfilled_power_demand = list()
unfulfilled_power_demand = list()

for i in range(len(G)):
    # Track functional nodes & links
    Gf = filter_functional_network(G[i])
    GT = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "transport"]).copy()
    GP = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "power"]).copy()
    nodes_t_functional.append(len(GT.nodes()))
    nodes_p_functional.append(len(GP.nodes()))
    links_t_functional.append(len(GT.edges()))
    links_p_functional.append(len(GP.edges()))

    # Track functional components & GCC size
    num_t_sub.append(len([subG for subG in weakly_connected_component_subgraphs(GT, copy=True)]))
    size_t_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(GT, copy=True)])
    num_p_sub.append(len([subG for subG in weakly_connected_component_subgraphs(GP, copy=True)]))
    size_p_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(GP, copy=True)])

    # Track unfulfilled power demand
    fd = 0.
    for n in G[i].nodes():
        if G[i].nodes[n].get("type", None) in ["load"]:
            if G[i].nodes[n]["state"] == 1:
                fd += G[i].nodes[n]["thruflow"]
    fulfilled_power_demand.append(fd)
    unfulfilled_power_demand.append((fulfilled_power_demand[0] - fd) / fulfilled_power_demand[0])

# Track weighted avg. time and unfulfilled trips
for i in range(len(shortest_paths)):  # FYI len(shortest_paths) = len(G) - 1
    i_ = str(i)  # PATCH - keys in json are in strings for some reason (JSON encoder issue)
    numer = 0.
    denom = 0.
    tft = 0.
    # for s in shortest_paths[str(len(shortest_paths) - 1)]:
    #     for t in shortest_paths[str(len(shortest_paths) - 1)][s]:
    #         if shortest_paths[str(len(shortest_paths) - 1)][s][t]["travel_time"] not in [np.inf, np.nan] \
    #                 and shortest_paths[str(len(shortest_paths) - 1)][s][t]["flow"] > 0.:
    #             numer += shortest_paths[i_][s][t]["travel_time"] * shortest_paths[i_][s][t]["flow"]
    #             denom += shortest_paths[i_][s][t]["flow"]
    for s in shortest_paths[i_]:
        for t in shortest_paths[i_][s]:
            if shortest_paths[i_][s][t]["travel_time"] not in [np.inf, np.nan] \
                    and shortest_paths[i_][s][t]["flow"] > 0.:
                numer += shortest_paths[i_][s][t]["travel_time"] * shortest_paths[i_][s][t]["flow"]
                denom += shortest_paths[i_][s][t]["flow"]
                tft += shortest_paths[i_][s][t]["flow"]
    wei_avg_time.append(numer / (denom + np.spacing(1)))
    fulfilled_trips.append(tft)
    unfulfilled_trips.append((fulfilled_trips[0] - tft) / fulfilled_trips[0])

for i in range(len(G)):
    print("ITERATION", i)
    print("Nodes functional:", "Transport:", nodes_t_functional[i], "Power:", nodes_p_functional[i])
    print("Links functional:", "Transport:", links_t_functional[i], "Power:", links_p_functional[i])
    print("Transport subnetworks: ", num_t_sub[i])
    print("Transport subnetworks sizes: ", size_t_sub[i])
    print("Power subnetworks: ", num_p_sub[i])
    print("Power subnetworks sizes: ", size_p_sub[i])

    if i < len(G) - 1:
        print("Weighted average time:", wei_avg_time[i])
        print("Total unfulfilled trips:", unfulfilled_trips[i])
        print("Total unfulfilled power demand:", unfulfilled_power_demand[i])

# Plot results
# TODO maybe instead of plotting by iterations, plot by number of remaining functional nodes
f, axs = plt.subplots(3, 2, figsize=(6, 4))

# axs[0, 0].plot(range(len(G)), np.array(nodes_t_functional) + np.array(nodes_p_functional))
axs[0, 0].plot(range(len(G)), nodes_t_functional, range(len(G)), nodes_p_functional)
axs[0, 0].set_xlabel("Iteration")
axs[0, 0].set_ylabel("Functional\nnodes")
axs[0, 0].set_yticks(range(0, 750, 250))
axs[0, 0].legend(["Transport", "Power"])

# axs[0, 1].plot(range(len(G)), np.array(links_t_functional) + np.array(links_p_functional))
axs[0, 1].plot(range(len(G)), links_t_functional, range(len(G)), links_p_functional)
axs[0, 1].set_xlabel("Iteration")
axs[0, 1].set_ylabel("Functional\nlinks")
axs[0, 1].set_yticks(range(0, 750, 250))
axs[0, 1].legend(["Transport", "Power"])

axs[1, 0].plot(range(len(G)), num_t_sub, range(len(G)), num_p_sub)
axs[1, 0].set_xlabel("Iteration")
axs[1, 0].set_ylabel("Functional\nconnected components")
axs[1, 0].legend(["Transport", "Power"])

axs[1, 1].plot(range(len(G)), [max(s) for s in size_t_sub], range(len(G)), [max(s) for s in size_p_sub])
axs[1, 1].set_xlabel("Iteration")
axs[1, 1].set_ylabel("Giant connected\ncomponent size")
axs[1, 1].legend(["Transport", "Power"])

axs[2, 0].plot(range(len(G) - 1), unfulfilled_trips)
axs[2, 0].set_xlabel("Iteration")
axs[2, 0].set_ylabel("Fraction unfulfilled\ntrips")

axs[2, 1].plot(range(len(G)), unfulfilled_power_demand)
axs[2, 1].set_xlabel("Iteration")
axs[2, 1].set_ylabel("Fraction unfulfilled\npower demand")

plt.show()

# Plot networks to show failure propagation
for i in range(len(G)):
    Gi_undir = G[i].to_undirected()
    GT = Gi_undir.subgraph([n for n in Gi_undir.nodes() if Gi_undir.nodes[n]["network"] == "transport"]).copy()
    GP = Gi_undir.subgraph([n for n in Gi_undir.nodes() if Gi_undir.nodes[n]["network"] == "power"]).copy()
    f, axs = plt.subplots(1, 2, figsize=(12, 4))
    for j, g in enumerate([GT, GP]):
        node_colors = [STATE_COLORS.get(g.nodes[n]["state"], "#FF0000") for n in g.nodes()]
        pos2D = {n: g.nodes[n]["pos"][0:2] for n in g.nodes()}
        edge_colors = [STATE_COLORS.get(g.edges[u, v]["state"], "#808080") for u, v in g.edges()]
        nx.draw(g, ax=axs[j], pos=pos2D, node_size=5,
                node_color=node_colors, edge_color=edge_colors)
    # plt.savefig("data/combined_network/infra_dynG_06202021_1824_"+str(i)+".png")
    # plt.savefig("data/combined_network/infra_dynG_06202021_1824_"+str(i)+".svg")
    plt.show()

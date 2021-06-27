import json
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dynamics import filter_functional_network
from globalparams import STATE_COLORS, COST_POWER, COST_TRANSPORT
from util import weakly_connected_component_subgraphs

G = pickle.load(open('data/combined_network/brimsdown132gen/SB_06242021_1234.pkl', "rb"))

with open('data/combined_network/brimsdown132gen/SB_06242021_1234.json', 'rb') as json_file:
    shortest_paths = json.load(json_file)

# Initialisation
nodes_t_functional = list()
nodes_p_functional = list()
links_t_functional = list()
links_p_functional = list()
links_i_functional = list()
node_robustness = list()
link_robustness = list()
GCC_robustness = list()
num_t_sub = list()
size_t_sub = list()
num_p_sub = list()
size_p_sub = list()
size_sub = list()
fulfilled_train_time_inc = list()
fulfilled_train_time = list()
unfulfilled_train_time = list()
cost_impact_train = list()
fulfilled_power_demand = list()
unfulfilled_power_demand = list()
cost_impact_power = list()

for i in range(len(G)):
    # Track functional nodes, links & interdependencies
    Gf = filter_functional_network(G[i])
    GT = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "transport"]).copy()
    GP = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "power"]).copy()
    nodes_t_functional.append(len(GT.nodes()))
    nodes_p_functional.append(len(GP.nodes()))
    links_t_functional.append(len(GT.edges()))
    links_p_functional.append(len(GP.edges()))
    links_i_functional.append(len([e for e in G[i].edges()
                                   if (G[i].edges[e]["network"] == "physical_interdependency"
                                   and G[i].edges[e]["state"] == 1)]))

    # Track functional components & GCC size
    num_t_sub.append(len([subG for subG in weakly_connected_component_subgraphs(GT, copy=True)]))
    size_t_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(GT, copy=True)])
    num_p_sub.append(len([subG for subG in weakly_connected_component_subgraphs(GP, copy=True)]))
    size_p_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(GP, copy=True)])
    size_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(Gf, copy=True)])

    # Calculate node, link & GCC robustness (Liu, Yin, Chen, Lv, Zhang - 2021)
    node_robustness.append((nodes_t_functional[-1] + nodes_p_functional[-1]) /
                           (nodes_t_functional[0] + nodes_p_functional[0]))
    link_robustness.append((links_t_functional[-1] + links_p_functional[-1] + links_i_functional[-1]) /
                           (links_t_functional[0] + links_p_functional[0] + links_i_functional[0]))
    GCC_robustness.append((max(size_sub[-1]) if len(size_sub[-1]) > 0 else 0) / max(size_sub[0]))

    # Track unfulfilled power demand
    fd = 0.
    for n in G[i].nodes():
        if G[i].nodes[n].get("type", None) in ["load"]:
            if G[i].nodes[n]["state"] == 1:
                fd += G[i].nodes[n]["thruflow"]
    fulfilled_power_demand.append(fd)
    unfulfilled_power_demand.append(fulfilled_power_demand[0] - fd)
    # Units: (MW demand) * (1h period) * (GBP / MWh) = (MW demand in 1h period) * (GBP / MWh) = GBP over 1h period
    cost_impact_power.append((fulfilled_power_demand[0] - fd) * COST_POWER)

# Track unfulfilled trips, and increase in travel time for fulfilled trips
for i in range(len(shortest_paths)):  # FYI len(shortest_paths) = len(G) - 1
    i_ = str(i)  # PATCH - keys in json are in strings for some reason (JSON encoder issue)
    total_train_demand = 0.
    ftt_inc = 0.
    ftt = 0.
    utt = 0.

    for s in shortest_paths[i_]:
        for t in shortest_paths[i_][s]:
            if shortest_paths[i_][s][t]["travel_time"] not in [np.inf, np.nan] \
                    and shortest_paths[i_][s][t]["flow"] > 0.:
                # Calculate increase in train time for all unaffected trips, from baseline
                ftt_inc += max(0., shortest_paths[i_][s][t]["flow"] * \
                         (shortest_paths[i_][s][t]["travel_time"] - shortest_paths["0"][s][t]["travel_time"]))  # PATCH
                ftt += shortest_paths[i_][s][t]["flow"] * shortest_paths[i_][s][t]["travel_time"]
            else:
                # Calculate total train time for all affected trips, from baseline
                utt += shortest_paths["0"][s][t]["flow"] * shortest_paths["0"][s][t]["travel_time"]

    fulfilled_train_time_inc.append(ftt_inc)
    fulfilled_train_time.append(ftt)
    unfulfilled_train_time.append(utt)
    # Units: (trip-min demand in 1h period) * (GBP / trip-min) = GBP over 1h period
    cost_impact_train.append(unfulfilled_train_time[-1] * COST_TRANSPORT
                             + fulfilled_train_time_inc[-1] * COST_TRANSPORT)

for i in range(len(G)):
    print("ITERATION", i)
    print("Node robustness:", node_robustness[i])
    print("Link robustness:", link_robustness[i])
    print("GCC robustness: ", GCC_robustness[i])

    if i < len(G) - 1:
        print("Train time increase for fulfilled trips:", fulfilled_train_time_inc[i])
        print("Total unfulfilled power demand:", unfulfilled_power_demand[i])
        print("Train time for unfulfilled trips:", unfulfilled_train_time[i])
        print("Train time for fulfilled trips:", fulfilled_train_time[i])

# Plot results
f, axs = plt.subplots(2, 3, figsize=(4, 6))

# TODO secondary axis?

axs[0, 0].plot(range(len(G)), node_robustness)
axs[0, 0].set_xlabel("Iteration")
axs[0, 0].set_ylabel("Node robustness")

axs[0, 1].plot(range(len(G)), link_robustness)
axs[0, 1].set_xlabel("Iteration")
axs[0, 1].set_ylabel("Link robustness")

axs[0, 2].plot(range(len(G)), GCC_robustness)
axs[0, 2].set_xlabel("Iteration")
axs[0, 2].set_ylabel("GCC robustness")

axs[1, 0].plot(range(len(G)), np.array(unfulfilled_power_demand) / fulfilled_power_demand[0])
axs[1, 0].set_xlabel("Iteration")
axs[1, 0].set_ylabel("Fraction unfulfilled \npower demand [MW]")

# FYI len(shortest_paths) = len(G) - 1
axs[1, 1].plot(range(len(G) - 1), np.array(unfulfilled_train_time) / fulfilled_train_time[0])
axs[1, 1].set_xlabel("Iteration")
axs[1, 1].set_ylabel("Fraction unfulfilled trips \ntrain time [trip-min]")

axs[1, 2].plot(range(len(G) - 1), (np.array(cost_impact_power[:-1]) + np.array(cost_impact_train)) /
               (fulfilled_power_demand[0] * COST_POWER + fulfilled_train_time[0] * COST_TRANSPORT))
axs[1, 2].set_xlabel("Iteration")
axs[1, 2].set_ylabel("Aggregated economic impact \n(fraction baseline revenue) [mil GBP/hr]")

plt.show()

# Test plot for node robustness vs. economic impact
plt.plot(node_robustness[:-1], np.array(cost_impact_power[:-1]) + np.array(cost_impact_train))
plt.xlabel("Node robustness")
plt.ylabel("Aggregated economic impact [mil GBP/hr]")
plt.show()

# Plot networks to show failure propagation
# for i in range(len(G)):
#     Gi_undir = G[i].to_undirected()
#     GT = Gi_undir.subgraph([n for n in Gi_undir.nodes() if Gi_undir.nodes[n]["network"] == "transport"]).copy()
#     GP = Gi_undir.subgraph([n for n in Gi_undir.nodes() if Gi_undir.nodes[n]["network"] == "power"]).copy()
#     f, axs = plt.subplots(1, 2, figsize=(12, 4))
#     for j, g in enumerate([GT, GP]):
#         node_colors = [STATE_COLORS.get(g.nodes[n]["state"], "#FF0000") for n in g.nodes()]
#         pos2D = {n: g.nodes[n]["pos"][0:2] for n in g.nodes()}
#         edge_colors = [STATE_COLORS.get(g.edges[u, v]["state"], "#808080") for u, v in g.edges()]
#         nx.draw(g, ax=axs[j], pos=pos2D, node_size=5,
#                 node_color=node_colors, edge_color=edge_colors)
#     # plt.savefig("data/brimsdown132gen/infra_dynG_06202021_1824_"+str(i)+".png")
#     # plt.savefig("data/brimsdown132gen/infra_dynG_06202021_1824_"+str(i)+".svg")
#     # plt.show()

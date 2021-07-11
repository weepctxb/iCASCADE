import os
import json
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dynamics import filter_functional_network
from globalparams import STATE_COLORS, COST_POWER, COST_TRANSPORT
from util import weakly_connected_component_subgraphs

GG = list()
sp = list()
directory = 'data/combined_network/brimsdown132gen'
filelist = sorted(os.listdir(directory))
for filename in filelist:
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and f.endswith('.pkl'):
        GG.append(pickle.load(open(f, "rb")))
    if os.path.isfile(f) and f.endswith('.json'):
        with open(f, 'rb') as json_file:
            sp.append(json.load(json_file))
F = int(len(GG))

# Initialisation
names = {index: filelist[3*index] for index in range(F)}
nodes_t_functional = {index: list() for index in range(F)}
nodes_p_functional = {index: list() for index in range(F)}
links_t_functional = {index: list() for index in range(F)}
links_p_functional = {index: list() for index in range(F)}
links_i_functional = {index: list() for index in range(F)}
node_robustness = {index: list() for index in range(F)}
link_robustness = {index: list() for index in range(F)}
GCC_robustness = {index: list() for index in range(F)}
num_t_sub = {index: list() for index in range(F)}
size_t_sub = {index: list() for index in range(F)}
num_p_sub = {index: list() for index in range(F)}
size_p_sub = {index: list() for index in range(F)}
size_sub = {index: list() for index in range(F)}
fulfilled_train_time_inc = {index: list() for index in range(F)}
fulfilled_train_time = {index: list() for index in range(F)}
unfulfilled_train_time = {index: list() for index in range(F)}
cost_impact_train = {index: list() for index in range(F)}
fulfilled_power_demand = {index: list() for index in range(F)}
unfulfilled_power_demand = {index: list() for index in range(F)}
cost_impact_power = {index: list() for index in range(F)}

for index, G in enumerate(GG):
    for i in range(len(G)):
        # Track functional nodes & links
        Gf = filter_functional_network(G[i])
        GT = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "transport"]).copy()
        GP = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "power"]).copy()
        nodes_t_functional[index].append(len(GT.nodes()))
        nodes_p_functional[index].append(len(GP.nodes()))
        links_t_functional[index].append(len(GT.edges()))
        links_p_functional[index].append(len(GP.edges()))
        links_i_functional[index].append(len([e for e in G[i].edges()
                                   if (G[i].edges[e]["network"] == "physical_interdependency"
                                   and G[i].edges[e]["state"] == 1)]))

        # Calculate node, link & GCC robustness (Liu, Yin, Chen, Lv, Zhang - 2021)
        node_robustness[index].append((nodes_t_functional[index][-1] + nodes_p_functional[index][-1]) /
                               (nodes_t_functional[index][0] + nodes_p_functional[index][0]))
        link_robustness[index].append((links_t_functional[index][-1] + links_p_functional[index][-1] + links_i_functional[index][-1]) /
                               (links_t_functional[index][0] + links_p_functional[index][0] + links_i_functional[index][0]))
        GCC_robustness[index].append((max(size_sub[index][-1]) if len(size_sub[index][-1]) > 0 else 0) / max(size_sub[index][0]))

        # Track functional components & GCC size
        num_t_sub[index].append(len([subG for subG in weakly_connected_component_subgraphs(GT, copy=True)]))
        size_t_sub[index].append([len(subG) for subG in weakly_connected_component_subgraphs(GT, copy=True)])
        num_p_sub[index].append(len([subG for subG in weakly_connected_component_subgraphs(GP, copy=True)]))
        size_p_sub[index].append([len(subG) for subG in weakly_connected_component_subgraphs(GP, copy=True)])
        size_sub[index].append([len(subG) for subG in weakly_connected_component_subgraphs(Gf, copy=True)])

        # Track unfulfilled power demand
        fd = 0.
        for n in G[i].nodes():
            if G[i].nodes[n].get("type", None) in ["load"]:
                if G[i].nodes[n]["state"] == 1:
                    fd += G[i].nodes[n]["thruflow"]
        fulfilled_power_demand[index].append(fd)
        unfulfilled_power_demand[index].append(fulfilled_power_demand[index][0] - fd)
        # Units: (MW demand) * (1h period) * (GBP / MWh) = (MW demand in 1h period) * (GBP / MWh) = GBP over 1h period
        cost_impact_power[index].append((fulfilled_power_demand[index][0] - fd) * COST_POWER)

for index, shortest_paths in enumerate(sp):
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

        fulfilled_train_time_inc[index].append(ftt_inc)
        fulfilled_train_time[index].append(ftt)
        unfulfilled_train_time[index].append(utt)

        # Units: (trip-min demand in 1h period) * (GBP / trip-min) = GBP over 1h period
        cost_impact_train[index].append(unfulfilled_train_time[index][-1] * COST_TRANSPORT
                                 + fulfilled_train_time_inc[index][-1] * COST_TRANSPORT)

# for index, G in enumerate(GG):
#     for i in range(len(G)):
#         print("ITERATION", i)
#         print("Node robustness:", node_robustness[index][i])
#         print("Link robustness:", link_robustness[index][i])
#         print("GCC robustness: ", GCC_robustness[index][i])
#
#         if i < len(G) - 1:
#             print("Train time increase for fulfilled trips:", fulfilled_train_time_inc[index][i])
#             print("Total unfulfilled power demand:", unfulfilled_power_demand[index][i])
#             print("Train time for unfulfilled trips:", unfulfilled_train_time[index][i])
#             print("Train time for fulfilled trips:", fulfilled_train_time[index][i])

# Export results
results = {names[index]: {
    "Node robustness": node_robustness[index][-1] / node_robustness[index][0],
    "Link robustness": link_robustness[index][-1] / link_robustness[index][0],
    "GCC robustness": GCC_robustness[index][-1] / GCC_robustness[index][0],
    "Fraction unfulfilled power demand": unfulfilled_power_demand[index][-1] / fulfilled_power_demand[index][0],
    "Fraction unfulfilled trips train time": unfulfilled_train_time[index][-1] / fulfilled_train_time[index][0],
    "Aggregated economic impact (fraction)": (cost_impact_power[index][-2] + cost_impact_train[index][-1]) /
                                             (fulfilled_power_demand[index][0] * COST_POWER +
                                              fulfilled_train_time[index][0] * COST_TRANSPORT)
} for index in range(F)}

with open(r'data/combined_network/brimsdown132gen_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=1)  # indent=2, cls=util.CustomEncoder

# Plot results
# f, axs = plt.subplots(2, 3, figsize=(4, 6))
#
# TODO secondary axis?
#
# for index, G in enumerate(GG):
#     axs[0, 0].plot(range(len(G)), node_robustness[index])
# axs[0, 0].set_xlabel("Iteration")
# axs[0, 0].set_ylabel("Node robustness")
#
# for index, G in enumerate(GG):
#     axs[0, 1].plot(range(len(G)), link_robustness[index])
# axs[0, 1].set_xlabel("Iteration")
# axs[0, 1].set_ylabel("Link robustness")
#
# for index, G in enumerate(GG):
#     axs[0, 2].plot(range(len(G)), GCC_robustness[index])
# axs[0, 2].set_xlabel("Iteration")
# axs[0, 2].set_ylabel("GCC robustness")
#
# for index, G in enumerate(GG):
#     axs[1, 0].plot(range(len(G)), np.array(unfulfilled_power_demand)[index] / fulfilled_power_demand[index][0])
# axs[1, 0].set_xlabel("Iteration")
# axs[1, 0].set_ylabel("Fraction unfulfilled \npower demand [MW]")
#
# FYI len(shortest_paths) = len(G) - 1
# for index, G in enumerate(GG):
#     axs[1, 1].plot(range(len(G) - 1), np.array(unfulfilled_train_time[index]) / fulfilled_train_time[index][0])
# axs[1, 1].set_xlabel("Iteration")
# axs[1, 1].set_ylabel("Fraction unfulfilled trips \ntrain time [trip-min]")
#
# for index, G in enumerate(GG):
#     axs[1, 2].plot(range(len(G) - 1), (np.array(cost_impact_power[index][:-1]) + np.array(cost_impact_train[index])) /
#                (fulfilled_power_demand[index][0] * COST_POWER + fulfilled_train_time[index][0] * COST_TRANSPORT))
# axs[1, 2].set_xlabel("Iteration")
# axs[1, 2].set_ylabel("Aggregated economic impact \n(fraction baseline revenue) [mil GBP/hr]")
#
# plt.show()

# Test plot for node robustness vs. economic impact
# for index, G in enumerate(GG):
#     plt.plot(node_robustness[index][:-1], np.array(cost_impact_power[index][:-1]) + np.array(cost_impact_train[index]))
# plt.xlabel("Node robustness")
# plt.ylabel("Aggregated economic impact [mil GBP/hr]")
# plt.show()

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

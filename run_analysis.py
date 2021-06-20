import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt

from dynamics import filter_functional_network
from globalparams import STATE_COLORS
from util import weakly_connected_component_subgraphs

G = pickle.load(open(r'data/combined_network/infra_dynG_[219]_06202021_1629.pkl', "rb"))
# G = pickle.load(open(r'data/combined_network/infra_dynG_06202021_1824.pkl', "rb"))

nodes_failed = list()
links_failed = list()
num_t_sub = list()
size_t_sub = list()
num_p_sub = list()
size_p_sub = list()

for i in range(len(G)):
    print("ITERATION", i)
    nodes_failed.append(len([n for n in G[i].nodes() if G[i].nodes[n]["state"] == 0]))
    print("Nodes failed:", nodes_failed[-1])
    links_failed.append(len([(u, v) for (u, v) in G[i].edges() if G[i].edges[u, v]["state"] == 0]))
    print("Links failed:", links_failed[-1])

    Gf = filter_functional_network(G[i])
    GT = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "transport"]).copy()
    GP = Gf.subgraph([n for n in Gf.nodes() if Gf.nodes[n]["network"] == "power"]).copy()

    num_t_sub.append(len([subG for subG in weakly_connected_component_subgraphs(GT, copy=True)]))
    print("Transport subnetworks: ", num_t_sub[-1])
    size_t_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(GT, copy=True)])
    print("Transport subnetworks sizes: ", size_t_sub[-1])
    num_p_sub.append(len([subG for subG in weakly_connected_component_subgraphs(GP, copy=True)]))
    print("Power subnetworks: ", num_p_sub[-1])
    size_p_sub.append([len(subG) for subG in weakly_connected_component_subgraphs(GP, copy=True)])
    print("Power subnetworks sizes: ", size_p_sub[-1])

f, axs = plt.subplots(2, 2, figsize=(6, 6))
axs[0, 0].plot(range(len(G)), nodes_failed)
axs[0, 1].plot(range(len(G)), links_failed)
axs[1, 0].plot(range(len(G)), num_t_sub)
axs[1, 1].plot(range(len(G)), num_p_sub)
plt.show()

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
    # plt.show()

# Track weighted average travel duration + Unfulfilled trips
# with open(r'data/transport_multiplex/flow/shortest_paths_skele.json', 'r') as json_file:
#     shortest_paths = [json.load(json_file)]
shortest_paths = pickle.load(open(r'data/combined_network/shortest_paths_id.pkl', 'wb+'))
wei_avg_time = list()
fulfilled_flows = list()
for i in range(len(shortest_paths)):
    numer = 0.
    denom = 0.
    tff = 0.
    for s in shortest_paths[i]:
        for t in shortest_paths[i][s]:
            numer += shortest_paths[i][s][t]["travel_time"] * shortest_paths[i][s][t]["flow"]
            denom += shortest_paths[i][s][t]["flow"]
            tff += shortest_paths[i][s][t]["flow"]
    wei_avg_time.append(numer / denom)
    fulfilled_flows.append(tff)
    print("Weighted average time:", wei_avg_time[-1])
    print("Total fulfilled flows:", fulfilled_flows[-1])
    # TODO Track total unfulfilled trips

# Track unfulfilled demand
unfulfilled_demand = list()
for i in range(len(G)):
    ud = 0.
    for n in G[i].nodes():
        if G[i].nodes[n]["state"] == 0 and G.nodes[n]["type"] in ["load"]:
            ud += G[i].nodes[n]["thruflow"]
    unfulfilled_demand.append(ud)

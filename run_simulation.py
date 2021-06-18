import json
import pickle

from dynamics import get_node, percolate_nodes, percolate_links, recompute_flows, fail_flow, fail_SIS
from util import parse_json

# Retrieve shortest paths for transport network
with open(r'data/transport_multiplex/flow/shortest_paths.json', 'r') as json_file:
    shortest_paths = json.load(json_file)
shortest_paths = parse_json(shortest_paths)

# Load original network, create temporal network, initialise time horizon and initialise failed nodes and links
G_original = pickle.load(open(r'data/combined_network/infra_G.pkl', "rb"))
G = [G_original]
TIME_HORIZON = 10
failed_nodes = {t: list() for t in range(TIME_HORIZON)}
failed_links = {t: list() for t in range(TIME_HORIZON)}

# Specify initial failure scenario
print("ITERATION 1")
# failed_nodes[0], _ = get_node(G[0], network="transport", mode="degree", top=1)
# failed_nodes[0] = ["Kingsway 11kV Gen"]
failed_nodes[0] = [0]

# Percolate failed nodes & links
Gn, failed_links[0] = percolate_nodes(G[0], failed_nodes=failed_nodes[0])
G.append(Gn)

# TODO Fail interdependencies

# Recompute flows
G[1], shortest_paths, failed_nodes[1], failed_links[1] = \
    recompute_flows(G[1], newly_failed_nodes=failed_nodes[0], newly_failed_links=failed_links[0],
                    shortest_paths=shortest_paths)  # shortest_paths will be overwritten

# Identify failures based on flow capacity or surges
newly_failed_links_flow = fail_flow(G[1], G[0],
                                    cap_lwr_threshold=0.9, cap_upp_threshold=1.0,
                                    ratio_lwr_threshold=1.5, ratio_upp_threshold=2.0)  # 360, 400, 1.5, 2
failed_links[1].extend(newly_failed_links_flow)

# Identify failures based on diffusion process # TODO Failure diffusion
newly_failed_nodes_dif = fail_SIS(G[1], G[0], infection_probability=0.05, recovery_probability=0.01)
failed_nodes[1].extend(newly_failed_nodes_dif)

# TODO TOO SLOW - try to remove centrality calculations, if not absolutely necessary!

for t in range(2, TIME_HORIZON):
    print("ITERATION", t)

    # Percolate failed nodes & links
    Gn, bydef_failed_links = percolate_nodes(G[t-1], failed_nodes=failed_nodes[t-1])
    failed_links[t - 1].extend(bydef_failed_links)
    Gn = percolate_links(Gn, failed_links=failed_links[t-1])
    G.append(Gn)

    # Recompute flows
    G[t], shortest_paths, failed_nodes[t], failed_links[t] = \
        recompute_flows(G[t], newly_failed_nodes=failed_nodes[t-1], newly_failed_links=failed_links[t-1],
                        shortest_paths=shortest_paths)  # shortest_paths will be overwritten

    # Identify failures based on flow capacity or surges
    newly_failed_links_flow = fail_flow(G[t], G[t-1],
                                        cap_lwr_threshold=0.9, cap_upp_threshold=1.0,
                                        ratio_lwr_threshold=1.5, ratio_upp_threshold=2.0)
    failed_links[t].extend(newly_failed_links_flow)

    # Identify failures based on diffusion process # TODO Failure diffusion
    newly_failed_nodes_dif = fail_SIS(G[t], G[t-1], infection_probability=0.05, recovery_probability=0.01)
    failed_nodes[t].extend(newly_failed_nodes_dif)

# TODO Test for transport network failures
# TODO Diffusion
# TODO Criticality calculation
# TODO Visualisation

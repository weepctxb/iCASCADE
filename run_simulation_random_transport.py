import copy
import json
import pickle
import sys
from datetime import datetime
from dynamics import get_node, percolate_nodes, percolate_links, recompute_flows, fail_flow, fail_SI
from util import parse_json


def run_simulation(id, simplified=True, time_horizon=20,
                   network="transport", type=None, mode="degree", top=1, override=None,
                   pow_cap_lwr_threshold=0.6, pow_cap_upp_threshold=1.0, pow_pmax=1.0,
                   trans_cap_lwr_threshold=0.9, trans_cap_upp_threshold=1.0, trans_pmax=0.2,
                   ratio_lwr_threshold=4.0, ratio_upp_threshold=5.0,
                   infection_probability=0.05, recovery_probability=0.0,
                   geo_threshold=0.1, geo_probability=0.2):

    # Create log
    print("SETTINGS")
    print("simplified:", simplified)
    print("time_horizon:", time_horizon)
    if override is not None:
        print("override:", override)
    else:
        print("network:", network)
        print("mode:", mode)
        print("top:", top)
    print("pow_cap_lwr_threshold:", pow_cap_lwr_threshold)
    print("pow_cap_upp_threshold:", pow_cap_upp_threshold)
    print("trans_cap_lwr_threshold:", trans_cap_lwr_threshold)
    print("trans_cap_upp_threshold:", trans_cap_upp_threshold)
    print("ratio_lwr_threshold:", ratio_lwr_threshold)
    print("ratio_upp_threshold:", ratio_upp_threshold)
    print("infection_probability:", infection_probability)
    print("recovery_probability:", recovery_probability)
    print("geo_threshold:", geo_threshold)
    print("geo_probability:", geo_probability)

    # Initialisation
    failed_nodes = {t: list() for t in range(time_horizon)}
    failed_links = {t: list() for t in range(time_horizon)}
    shortest_paths = {t: dict() for t in range(time_horizon)}

    # Retrieve shortest paths for transport network
    if simplified:
        with open(r'data/transport_multiplex/flow/shortest_paths_skele.json', 'r') as json_file:
            orig_shortest_paths = json.load(json_file)
    else:
        with open(r'data/transport_multiplex/flow/shortest_paths.json', 'r') as json_file:
            orig_shortest_paths = json.load(json_file)
    shortest_paths[0] = parse_json(orig_shortest_paths)

    # Load original network, create temporal network, initialise time horizon and initialise failed nodes and links
    if simplified:
        G_original = pickle.load(open(r'data/combined_network/infra_G_skele.pkl', "rb"))
    else:
        G_original = pickle.load(open(r'data/combined_network/infra_G.pkl', "rb"))
    G = [G_original]

    # Specify initial failure scenario
    print("ITERATION 1")
    if override is not None:
        failed_nodes[0] = override
    else:
        failed_nodes[0], _ = get_node(G[0], network=network, mode=mode, top=top)
    # failed_nodes[0] = ["Kingsway 11kV Gen"]
    # failed_nodes[0] = [0]

    # Percolate failed nodes & links
    Gn, failed_links[0] = percolate_nodes(G[0], failed_nodes=failed_nodes[0])
    G.append(Gn)

    # Recompute flows
    G[1], shortest_paths[1], failed_nodes[1], failed_links[1] = \
        recompute_flows(G[1], newly_failed_nodes=failed_nodes[0], newly_failed_links=failed_links[0],
                        shortest_paths=shortest_paths[0])

    # Identify failures based on flow capacity or surges
    newly_failed_nodes_flow, newly_failed_links_flow, shortest_paths[1] = \
        fail_flow(G[1], G[0], shortest_paths=shortest_paths[1],
                  pow_cap_lwr_threshold=pow_cap_lwr_threshold, pow_cap_upp_threshold=pow_cap_upp_threshold,
                  pow_pmax=pow_pmax,
                  trans_cap_lwr_threshold=trans_cap_lwr_threshold, trans_cap_upp_threshold=trans_cap_upp_threshold,
                  trans_pmax=trans_pmax,
                  ratio_lwr_threshold=ratio_lwr_threshold, ratio_upp_threshold=ratio_upp_threshold)
    failed_nodes[1].extend(newly_failed_nodes_flow)
    failed_links[1].extend(newly_failed_links_flow)

    # Identify failures based on diffusion process # TODO Fail deterministically for interdependencies
    newly_failed_nodes_dif = fail_SI(G[1],
                                     infection_probability=infection_probability,
                                     recovery_probability=recovery_probability,
                                     geo_threshold=geo_threshold,
                                     geo_probability=geo_probability)
    failed_nodes[1].extend(newly_failed_nodes_dif)

    failed_nodes[1] = list(set(failed_nodes[1]))
    failed_links[1] = list(set(failed_links[1]))

    for t in range(2, time_horizon):

        # Percolate failed nodes & links
        Gn, bydef_failed_links = percolate_nodes(G[t-1], failed_nodes=failed_nodes[t-1])
        failed_links[t-1].extend(bydef_failed_links)
        Gn, bydef_failed_links = percolate_links(Gn, failed_links=failed_links[t-1])
        failed_links[t-1].extend(bydef_failed_links)
        G.append(Gn)

        print("Iteration", str(t-1), "failed nodes:", str(failed_nodes[1]))
        print("Iteration", str(t-1), "failed links:", str(failed_links[1]))

        print("ITERATION", t)

        # Recompute flows, only if topology changed
        # TODO can we speed up by skipping this for minor changes in topology OR alternate iterations?
        if len(failed_nodes[t-1]) > 0 or len(failed_links[t-1]) > 0:
            G[t], shortest_paths[t], failed_nodes[t], failed_links[t] = \
                recompute_flows(G[t], newly_failed_nodes=failed_nodes[t-1], newly_failed_links=failed_links[t-1],
                                shortest_paths=shortest_paths[t-1])  # shortest_paths will be overwritten
        else:
            shortest_paths[t] = copy.deepcopy(shortest_paths[t-1])
            failed_nodes[t] = list()
            failed_links[t] = list()

        # Identify failures based on flow capacity or surges
        newly_failed_nodes_flow, newly_failed_links_flow, shortest_paths[t] = \
            fail_flow(G[t], G[t-1], shortest_paths=shortest_paths[t],
                      pow_cap_lwr_threshold=pow_cap_lwr_threshold, pow_cap_upp_threshold=pow_cap_upp_threshold,
                      pow_pmax=pow_pmax,
                      trans_cap_lwr_threshold=trans_cap_lwr_threshold, trans_cap_upp_threshold=trans_cap_upp_threshold,
                      trans_pmax=trans_pmax,
                      ratio_lwr_threshold=ratio_lwr_threshold, ratio_upp_threshold=ratio_upp_threshold)
        failed_nodes[t].extend(newly_failed_nodes_flow)
        failed_links[t].extend(newly_failed_links_flow)

        # Identify failures based on diffusion process
        newly_failed_nodes_dif = fail_SI(G[t],
                                         infection_probability=infection_probability,
                                         recovery_probability=recovery_probability,
                                         geo_threshold=geo_threshold,
                                         geo_probability=geo_probability)
        failed_nodes[t].extend(newly_failed_nodes_dif)

        failed_nodes[t] = list(set(failed_nodes[t]))
        failed_links[t] = list(set(failed_links[t]))

    # Last iteration
    print("ITERATION", time_horizon)

    # Percolate failed nodes & links
    Gn, bydef_failed_links = percolate_nodes(G[time_horizon-1], failed_nodes=failed_nodes[time_horizon-1])
    failed_links[time_horizon-1].extend(bydef_failed_links)
    Gn, bydef_failed_links = percolate_links(Gn, failed_links=failed_links[time_horizon-1])
    failed_links[time_horizon-1].extend(bydef_failed_links)
    G.append(Gn)

    print("Iteration", str(time_horizon-1), "failed nodes:", str(failed_nodes[time_horizon-1]))
    print("Iteration", str(time_horizon-1), "failed links:", str(failed_links[time_horizon-1]))

    # Save the whole thing
    pickle.dump(G, open(r'data/combined_network/random_transport/'+id+".pkl", 'wb+'))
    with open(r'data/combined_network/random_transport/'+id+".json", 'w') as json_file:
        json.dump(shortest_paths, json_file)  # indent=2, cls=util.CustomEncoder


if __name__ == "__main__":

    # run_simulation(id, simplified=True, time_horizon=30,
    #                network="power", mode="degree", top=1, override=["Cobourg Street Electricity Substation"],
    #                pow_cap_lwr_threshold=0.9, pow_cap_upp_threshold=1.2,
    #                trans_cap_lwr_threshold=0.9, trans_cap_upp_threshold=1.2,
    #                ratio_lwr_threshold=4.0, ratio_upp_threshold=5.0,
    #                infection_probability=0.05, recovery_probability=0.0,
    #                geo_threshold=0.1, geo_probability=0.2)

    terminal = sys.stdout

    for test in range(0, 10):
        id = test

        id = "RT" + str(id) + "_" + datetime.now().strftime("%m%d%Y_%H%M")
        print(id)

        try:
            sys.stdout = open(r'data/combined_network/random_transport/' + id + '.log', 'w+')
            run_simulation(id, simplified=True, time_horizon=50,
                           network="transport", mode="random", top=1, override=None,
                           pow_cap_lwr_threshold=0.8, pow_cap_upp_threshold=1.0, pow_pmax=1.0,
                           trans_cap_lwr_threshold=0.8, trans_cap_upp_threshold=1.0, trans_pmax=0.2,
                           ratio_lwr_threshold=1e6, ratio_upp_threshold=1e7,
                           infection_probability=0.05, recovery_probability=0.0,
                           geo_threshold=0.1, geo_probability=0.05)
            sys.stdout.close()
            sys.stdout = terminal
        except Exception as e:
            sys.stdout.close()
            sys.stdout = terminal
            print(e)
            continue  # continue to next test

    # TODO Sensitivity analysis:
    #  Justification for pow_cap_lwr_threshold: Based on Nie's setting of threshold near to baseline utilisation
    #  which is average 57-58% for power network
    #  Justification for trans_cap_lwr_threshold: Based on 10% (Goldbeck)


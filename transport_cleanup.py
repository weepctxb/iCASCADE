import numpy as np
import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from scipy import sparse

from util import perpendicular_dist, haversine
from globalparams import TRANSPORT_LINE_PP_EPS, TRANSPORT_PROXIMITY_THRESHOLD, TRANSPORT_COLORS

if __name__ == "__main__":

    # FYI NOT IN USE

    # Stations and lines
    transport_multiplex_G = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', "rb"))
    # Power_supply, site, yard, depot, engine_shed
    transport_osm_G = pickle.load(open(r'data/transport_osm/out/transport_OSM_G.pkl', "rb"))

    transport_combined_G = nx.compose(transport_multiplex_G, transport_osm_G)

    # CLEANUP

    # TODO Cluster switches together (Add back switches first)

    # Connect power supply / switches / signals / depots etc. with train lines
    # TODO Assume that they go "directly between" the train stations for simplification of topology
    for node in transport_combined_G.nodes():

        # Connect power supply / switches / signals with train lines (only applies to London Overground or TfL rail)
        # TODO ignored switches first
        if transport_combined_G.nodes[node].get("railway", "-") in ["power_supply", "signal", "signal_box"]:
            dmin = np.Inf
            bestu = None
            bestv = None
            snap = False
            for u, v in transport_combined_G.edges():  # Only consider LO & TfL stations
                if transport_combined_G.nodes[u].get("railway", "-") == "station" and \
                        transport_combined_G.nodes[v].get("railway", "-") == "station" and \
                        (str(transport_combined_G.edges[u, v]["Line"]).startswith("lo-") or \
                         str(transport_combined_G.edges[u, v]["Line"]).startswith("tfl-")):
                    pd = perpendicular_dist(transport_combined_G.nodes[node]["pos"][1], transport_combined_G.nodes[node]["pos"][0],
                                           transport_combined_G.nodes[u]["pos"][1], transport_combined_G.nodes[u]["pos"][0],
                                           transport_combined_G.nodes[v]["pos"][1], transport_combined_G.nodes[v]["pos"][0])
                    pxu = haversine(transport_combined_G.nodes[node]["pos"][1], transport_combined_G.nodes[node]["pos"][0],
                                   transport_combined_G.nodes[u]["pos"][1], transport_combined_G.nodes[u]["pos"][0])
                    pxv = haversine(transport_combined_G.nodes[node]["pos"][1], transport_combined_G.nodes[node]["pos"][0],
                                   transport_combined_G.nodes[v]["pos"][1], transport_combined_G.nodes[v]["pos"][0])
                    if pd < TRANSPORT_LINE_PP_EPS and (pxu < TRANSPORT_PROXIMITY_THRESHOLD or pxv < TRANSPORT_PROXIMITY_THRESHOLD):
                        if pd < dmin:
                            dmin = pd
                            bestu = u
                            bestv = v
                            snap = True
            if snap:
                print("signal: ", bestu, node, bestv)
                transport_combined_G.add_edge(bestu, node,
                                              Line=transport_combined_G.edges[bestu, bestv]["Line"],
                                              StationA=transport_combined_G.edges[bestu, bestv]["StationA"],
                                              StationB=transport_combined_G.edges[bestu, bestv]["StationB"],
                                              Distance=haversine(
                                                  transport_combined_G.nodes[bestu]["pos"][1],
                                                  transport_combined_G.nodes[bestu]["pos"][0],
                                                  transport_combined_G.nodes[node]["pos"][1],
                                                  transport_combined_G.nodes[node]["pos"][0]))
                transport_combined_G.add_edge(node, bestv,
                                              Line=transport_combined_G.edges[bestu, bestv]["Line"],
                                              StationA=transport_combined_G.edges[bestu, bestv]["StationA"],
                                              StationB=transport_combined_G.edges[bestu, bestv]["StationB"],
                                              Distance=haversine(
                                                  transport_combined_G.nodes[bestv]["pos"][1],
                                                  transport_combined_G.nodes[bestv]["pos"][0],
                                                  transport_combined_G.nodes[node]["pos"][1],
                                                  transport_combined_G.nodes[node]["pos"][0]))
                transport_combined_G.remove_edge(bestu, bestv)

        # Connect depots / engine sheds / sites / yard with train lines (applies to all lines)
        elif transport_combined_G.nodes[node].get("railway", "-") in ["depot", "engine_shed", "site", "yard"]:
            for u, v in transport_combined_G.edges():  # Only consider train stations (all lines)
                if transport_combined_G.nodes[u].get("railway", "-") == "station" and \
                        transport_combined_G.nodes[v].get("railway", "-") == "station":
                    pd = perpendicular_dist(transport_combined_G.nodes[node]["pos"][1], transport_combined_G.nodes[node]["pos"][0],
                                           transport_combined_G.nodes[u]["pos"][1], transport_combined_G.nodes[u]["pos"][0],
                                           transport_combined_G.nodes[v]["pos"][1], transport_combined_G.nodes[v]["pos"][0])
                    pxu = haversine(transport_combined_G.nodes[node]["pos"][1], transport_combined_G.nodes[node]["pos"][0],
                                    transport_combined_G.nodes[u]["pos"][1], transport_combined_G.nodes[u]["pos"][0])
                    pxv = haversine(transport_combined_G.nodes[node]["pos"][1], transport_combined_G.nodes[node]["pos"][0],
                                    transport_combined_G.nodes[v]["pos"][1], transport_combined_G.nodes[v]["pos"][0])
                    if pd < TRANSPORT_LINE_PP_EPS and (pxu < TRANSPORT_PROXIMITY_THRESHOLD or pxv < TRANSPORT_PROXIMITY_THRESHOLD):
                        print("depot: ", u, node, v)
                        transport_combined_G.add_edge(u, node,
                                                      Line=transport_combined_G.edges[u, v]["Line"],
                                                      StationA=transport_combined_G.edges[u, v]["StationA"],
                                                      StationB=transport_combined_G.edges[u, v]["StationB"],
                                                      Distance=haversine(
                                                          transport_combined_G.nodes[u]["pos"][1],
                                                          transport_combined_G.nodes[u]["pos"][0],
                                                          transport_combined_G.nodes[node]["pos"][1],
                                                          transport_combined_G.nodes[node]["pos"][0]))
                        transport_combined_G.add_edge(node, v,
                                                      Line=transport_combined_G.edges[u, v]["Line"],
                                                      StationA=transport_combined_G.edges[u, v]["StationA"],
                                                      StationB=transport_combined_G.edges[u, v]["StationB"],
                                                      Distance=haversine(
                                                          transport_combined_G.nodes[v]["pos"][1],
                                                          transport_combined_G.nodes[v]["pos"][0],
                                                          transport_combined_G.nodes[node]["pos"][1],
                                                          transport_combined_G.nodes[node]["pos"][0]))

    try:
        pickle.dump(transport_combined_G, open(r'data/transport_combined/transport_combined_G.pkl', 'wb+'))
    except FileNotFoundError as e:
        print(e)

    try:
        combined_nodelist = pd.DataFrame([i[1] for i in transport_combined_G.nodes(data=True)],
                                         index=[i[0] for i in transport_combined_G.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_combined/transport_combined_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)  # FIXME xlsx file here is outdated because of: 'float' object has no attribute 'DataFrame'

    try:
        combined_adjmat = nx.adjacency_matrix(transport_combined_G, nodelist=None, weight="weight")
        # gives scipy sparse matrix
        sparse.save_npz(r'data/transport_combined/transport_combined_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    node_colors = [TRANSPORT_COLORS.get(
        transport_combined_G.nodes[node].get("line", "-")
        if transport_combined_G.nodes[node].get("railway", "-") == "station"
        else transport_combined_G.nodes[node].get("railway", "-"),
        "#FF0000")
        for node in transport_combined_G.nodes()]
    edge_colors = [TRANSPORT_COLORS.get(transport_combined_G.edges[u, v].get("Line", "-"), "#808080")
                   for u, v in transport_combined_G.edges()]

    nx.draw(transport_combined_G, pos=nx.get_node_attributes(transport_combined_G, 'pos'),
            node_size=5, node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/transport_combined/transport_combined.png")
    plt.savefig("data/transport_combined/transport_combined.svg")
    plt.show()

import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from scipy import sparse

from globalparams import TRANSPORT_COLORS


# T.O.1.1 Load nodes and edges data
def load_transport_osm_data():

    # T.O.1.1.1 Load OSM points
    try:
        point = pd.read_pickle("data/transport_osm/in/transport_OSM_point.pkl")
    except IOError:
        point = pd.read_excel("data/transport_osm/raw/transport_OSM_combined.xlsx", sheet_name="transport_OSM_point", header=0)
        point.to_pickle("data/transport_osm/in/transport_OSM_point.pkl")

    # T.O.1.1.2 Load OSM polygons
    try:
        polyg = pd.read_pickle("data/transport_osm/in/transport_OSM_polygon.pkl")
    except IOError:
        polyg = pd.read_excel("data/transport_osm/raw/transport_OSM_combined.xlsx", sheet_name="transport_OSM_polygon", header=0)
        polyg.to_pickle("data/transport_osm/in/transport_OSM_polygon.pkl")

    # T.O.1.1.3 Load OSM lines
    try:
        line = pd.read_pickle("data/transport_osm/in/transport_OSM_line.pkl")
    except IOError:
        line = pd.read_excel("data/transport_osm/raw/transport_OSM_combined.xlsx", sheet_name="transport_OSM_line", header=0)
        line.to_pickle("data/transport_osm/in/transport_OSM_line.pkl")

    # T.O.1.1.4 Load OSM line vertices
    try:
        linevertex = pd.read_pickle("data/transport_osm/in/transport_OSM_line_vertices.pkl")
    except IOError:
        linevertex = pd.read_excel("data/transport_osm/raw/transport_OSM_combined.xlsx", sheet_name="transport_OSM_line_vertices", header=0)
        linevertex.to_pickle("data/transport_osm/in/transport_OSM_line_vertices.pkl")
    
    return point, polyg, line, linevertex


# T.O.1.2 Create networkx graph
def create_transport_osm_graph(point, polyg, line, linevertex):

    G = nx.Graph()

    # T.O.1.2.1 Extract specific nodes
    for _, row in point.iterrows():
        if row["railway"] in ["power_supply", "site", "yard"]:  # "signal", "switch"
            G.add_node(row["full_id"], pos=(row["$x"], row["$y"]),
                       railway=row["railway"], network=row["network"], line=row["line"],
                       name=row["name"], operator=row["operator"], electrified=row["electrified"])

    # T.O.1.2.2 Only extract depots and engine sheds from polygons
    for _, row in polyg.iterrows():
        if row["railway"] in ["depot", "engine_shed"]:  # "signal_box"
            G.add_node(row["full_id"], pos=(row["x(centroid($geometry))"], row["y(centroid($geometry))"]),
                       railway=row["railway"], network=row["network"], line=row["line"],
                       name=row["name"], operator=row["operator"])

    return G


if __name__ == "__main__":

    # T.O.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/transport_osm/out/transport_OSM_G.pkl', "rb"))
    except IOError:
        point, polyg, line, linevertex = load_transport_osm_data()
        G = create_transport_osm_graph(point, polyg, line, linevertex)
        try:
            pickle.dump(G, open(r'data/transport_osm/out/transport_OSM_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.O.2 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_osm/out/transport_OSM_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # T.O.3 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/transport_osm/out/transport_OSM_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # T.O.4 Add colours and export map plot
    node_colors = [TRANSPORT_COLORS.get(G.nodes[node]["railway"], "#FF0000") for node in G.nodes()]
    # edge_powers = nx.get_edge_attributes(G, "Line")
    # edge_colors = [transport_colors.get(edge_powers[u, v], "#808080") for u, v in G.edges()]

    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=5, node_color=node_colors)
    # nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=5,
    #         node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/transport_osm/img/transport_osm.png")
    plt.savefig("data/transport_osm/img/transport_osm.svg")
    plt.show()

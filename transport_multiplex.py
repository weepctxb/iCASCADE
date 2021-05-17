import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse

from globalparams import TRANSPORT_COLORS


# T.M.1.1 Load nodes and edges data
def load_transport_multiplex_data():

    # T.M.1.1.1 Load nodes
    try:
        nodes = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_nodes.pkl")
    except IOError:
        nodes = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx", sheet_name="london_transport_nodes", header=0)
        nodes.to_pickle("data/transport_multiplex/in/transport_multiplex_nodes.pkl")

    # T.M.1.1.2 Load edges
    try:
        edges = pd.read_pickle("data/transport_multiplex/in/transport_multiplex_edges.pkl")
    except IOError:
        edges = pd.read_excel("data/transport_multiplex/raw/transport_multiplex.xlsx", sheet_name="london_transport_raw", header=0)
        edges.to_pickle("data/transport_multiplex/in/transport_multiplex_edges.pkl")

    return nodes, edges


# T.M.1.2 Create networkx graph
def create_transport_multiplex_graph(nodes, edges):

    # T.M.1.2.1 Create graph from edges dataframe
    # TODO add travel times, flows and capacities
    G = nx.from_pandas_edgelist(edges, source="StationA_ID", target="StationB_ID",
                                edge_attr=["Line", "StationA", "StationB", "Distance"])

    # T.M.1.2.2 Create and assign node attributes
    pos = dict()
    nodeLabel = dict()
    interchange = dict()
    line = dict()
    for index, row in nodes.iterrows():
        pos[row["nodeID"]] = (row["nodeLong"], row["nodeLat"])
        nodeLabel[row["nodeID"]] = row["nodeLabel"]
        interchange[row["nodeID"]] = row["interchange"]
        line[row["nodeID"]] = row["line"]

    nx.set_node_attributes(G, pos, 'pos')
    nx.set_node_attributes(G, nodeLabel, 'nodeLabel')
    nx.set_node_attributes(G, interchange, 'interchange')
    nx.set_node_attributes(G, line, 'line')

    nx.set_node_attributes(G, "station", 'railway')

    # T.M.1.2.3 Calculate and assign centralities
    dd = nx.degree_centrality(G)
    bb = nx.betweenness_centrality(G, normalized=True, weight=True)
    cc = nx.closeness_centrality(G, distance="Distance")
    ee = nx.eigenvector_centrality(G, max_iter=1000)

    nx.set_node_attributes(G, dd, 'degree')
    nx.set_node_attributes(G, bb, 'betweenness')
    nx.set_node_attributes(G, cc, 'closeness')
    nx.set_node_attributes(G, ee, 'eigenvector')

    nodes['degree'] = None
    nodes['betweenness'] = None
    nodes['closeness'] = None
    nodes['eigenvector'] = None
    for node in dd:
        nodes.at[node, 'degree'] = dd[node]
    for node in bb:
        nodes.at[node, 'betweenness'] = bb[node]
    for node in cc:
        nodes.at[node, 'closeness'] = cc[node]
    for node in ee:
        nodes.at[node, 'eigenvector'] = ee[node]

    # T.M.1.2.4 Calculate node sizes
    node_sizes = [int(bb[node] * 1000) for node in bb]

    return G, pos, node_sizes


def plot_degree_histogram(G):

    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.loglog(degrees[:], degree_freq[:], 'go-')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

    weighted_degrees = []
    for n in G.nodes():
        dist = sum(edges.loc[edges["StationA_ID"] == n]["Distance"]) + sum(
            edges.loc[edges["StationB_ID"] == n]["Distance"])
        weighted_degrees.append(round(dist * 1000))
    n, bins, patches = plt.hist(weighted_degrees, bins=100)
    plt.clf()
    plt.loglog(bins[:-1], n[:], ".")
    plt.show()


if __name__ == "__main__":

    # T.M.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', "rb"))
    except IOError:
        nodes, edges = load_transport_multiplex_data()
        G, pos, node_sizes = create_transport_multiplex_graph(nodes, edges)
        try:
            pickle.dump(G, open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # T.M.2 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/transport_multiplex/out/transport_multiplex_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # T.M.3 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/transport_multiplex/out/transport_multiplex_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # T.M.4 Add colours and export map plot
    node_colors = [TRANSPORT_COLORS.get(G.nodes[node]["line"], "#FF0000") for node in G.nodes()]
    edge_lines = nx.get_edge_attributes(G, "Line")
    edge_colors = [TRANSPORT_COLORS.get(edge_lines[u, v], "#808080") for u, v in G.edges()]

    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/transport_multiplex/img/transport_multiplex.png")
    plt.savefig("data/transport_multiplex/img/transport_multiplex.svg")
    plt.show()

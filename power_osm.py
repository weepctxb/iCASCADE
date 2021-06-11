import pandas as pd
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import sparse

import util
from globalparams import POWER_COLORS, POWER_PROXIMITY_THRESHOLD


# P.O.1.1 Load nodes and edges data
def load_power_osm_data():

    # P.O.1.1.1 Load OSM points
    try:
        point = pd.read_pickle("data/power_osm/in/power_OSM_point.pkl")
    except IOError:
        point = pd.read_excel("data/power_osm/raw/power_OSM_combined.xlsx", sheet_name="power_London_point", header=0)
        point['power'] = point['power'].astype(str)
        point['name'] = point['name'].astype(str)
        point['operator'] = point['operator'].astype(str)
        point.to_pickle("data/power_osm/in/power_OSM_point.pkl")

    # P.O.1.1.2 Load OSM polygons
    try:
        polyg = pd.read_pickle("data/power_osm/in/power_OSM_polygon.pkl")
    except IOError:
        polyg = pd.read_excel("data/power_osm/raw/power_OSM_combined.xlsx", sheet_name="power_London_polygon", header=0)
        polyg['power'] = polyg['power'].astype(str)
        polyg['name'] = polyg['name'].astype(str)
        polyg['substation'] = polyg['substation'].astype(str)
        polyg['operator'] = polyg['operator'].astype(str)
        polyg.to_pickle("data/power_osm/in/power_OSM_polygon.pkl")

    # P.O.1.1.3 Load OSM lines
    try:
        line = pd.read_pickle("data/power_osm/in/power_OSM_line.pkl")
    except IOError:
        line = pd.read_excel("data/power_osm/raw/power_OSM_combined.xlsx", sheet_name="power_London_line", header=0)
        line['power'] = line['power'].astype(str)
        line['name'] = line['name'].astype(str)
        line['substation'] = line['substation'].astype(str)
        line['operator'] = line['operator'].astype(str)
        line.to_pickle("data/power_osm/in/power_OSM_line.pkl")

    # P.O.1.1.4 Load OSM line vertices
    try:
        linevertex = pd.read_pickle("data/power_osm/in/power_OSM_line_vertices.pkl")
    except IOError:
        linevertex = pd.read_excel("data/power_osm/raw/power_OSM_combined.xlsx", sheet_name="power_London_line_vertices", header=0)
        linevertex['power'] = linevertex['power'].astype(str)
        linevertex['name'] = linevertex['name'].astype(str)
        linevertex['substation'] = linevertex['substation'].astype(str)
        linevertex['operator'] = linevertex['operator'].astype(str)
        linevertex.to_pickle("data/power_osm/in/power_OSM_line_vertices.pkl")

    return point, polyg, line, linevertex


# P.O.1.2 Create networkx graph
def create_power_osm_graph(point, polyg, line, linevertex):

    G = nx.Graph()

    # P.0.1.2.1 Points
    for _, row in point.iterrows():
        if row["operator"] == "National Grid":
            G.add_node(row["full_id"], pos=(row["$x"], row["$y"]),
                       power=row["power"], name=row["name"], operator=row["operator"])

    # P.0.1.2.2 Extract polygons (exclude relations), but compress them into points based on their centroids
    for _, row in polyg.iterrows():
        if row["power"] == "substation" and ("National Grid" in row["operator"] or row["substation"] == "transmission"):
            G.add_node(row["full_id"], pos=(row["x(centroid($geometry))"], row["y(centroid($geometry))"]),
                       power=row["power"], substation=row["substation"], name=row["name"],
                       operator=row["operator"], voltage=row["voltage"])

    # P.0.1.2.3 Some substations, plants and transformers wrongly coded as line
    # -> compress them into points based on their centroids
    for _, row in line.iterrows():
        if row["power"] == "substation" and (row["substation"] in ["transmission", "transition"]
                                             or "National Grid" in row["operator"]):
            substnnodes = linevertex.loc[linevertex["full_id"] == row["full_id"]]
            x = np.nanmean(substnnodes["$x"])
            y = np.nanmean(substnnodes["$y"])
            G.add_node(row["full_id"], pos=(x, y),
                       power=row["power"], substation=row["substation"], name=row["name"],
                       operator=row["operator"], voltage=row["voltage"])

    # P.0.1.2.4 Extract cables and lines
    for _, row in line.iterrows():
        if row["power"] in ["line"] and ("National Grid" in row["name"] or "National Grid" in row["operator"]):
            # Overhead cables are supported by poles or towers, and assumed to be straight between poles & towers
            # Snap ends to existing substations, generators, transformers, compensators, switches etc.
            linenodes = linevertex.loc[linevertex["full_id"] == row["full_id"]]

            linenodes_id = []
            # For each vertex in multi-line
            for _, linenode in linenodes.iterrows():
                dmin = np.Inf
                bestnode = None
                snap = False
                # Find the best existing node that is the closest to the vertex within the threshold
                for node in G.nodes():
                    if (linenode["$x"], linenode["$y"]) == G.nodes[node]["pos"]:
                        bestnode = node
                        snap = True
                        break
                    else:
                        d = util.haversine(linenode["$x"], linenode["$y"], G.nodes[node]["pos"][0],
                                           G.nodes[node]["pos"][1])
                        if d < POWER_PROXIMITY_THRESHOLD:
                            if d < dmin:
                                dmin = d
                                bestnode = node
                                snap = True
                if snap:
                    linenodes_id.append(bestnode)
                else:
                    vid = str(linenode["full_id"]) + "-" + str(linenode["vertex_index"])
                    G.add_node(vid, pos=(linenode["$x"], linenode["$y"]),
                               power=linenode["power"], substation=linenode["substation"], name=linenode["name"],
                               operator=linenode["operator"], voltage=linenode["voltage"])
                    linenodes_id.append(vid)
            for lnid1, lnid2 in zip(linenodes_id[:-1], linenodes_id[1:]):
                G.add_edge(lnid1, lnid2, power=row["power"], substation=row["substation"], name=row["name"],
                           operator=row["operator"], voltage=row["voltage"])

        elif row["power"] == "cable" and ("National Grid" in row["name"] or "National Grid" in row["operator"]):
            # Underground cables are either linear (deep cable tunnels) or follow along road
            # -> Need to create non-physical nodes for geographical accuracy of intermediate line vertices
            # But simplify the line geometry
            # Snap ends to existing substations, generators, transformers, compensators, switches etc.
            linenodes = linevertex.loc[linevertex["full_id"] == row["full_id"]]

            # Simplify multi-line geometry
            if len(linenodes) > 2:
                linenodes = util.douglas_peucker(linenodes)

            linenodes_id = []
            # For each vertex in multi-line
            for _, linenode in linenodes.iterrows():
                dmin = np.Inf
                bestnode = None
                snap = False
                # Find the best existing node that is the closest to the vertex within the threshold
                for node in G.nodes():
                    if (linenode["$x"], linenode["$y"]) == G.nodes[node]["pos"]:
                        bestnode = node
                        snap = True
                        break
                    else:
                        d = util.haversine(linenode["$x"], linenode["$y"], G.nodes[node]["pos"][0],
                                           G.nodes[node]["pos"][1])
                        if d < POWER_PROXIMITY_THRESHOLD:
                            if d < dmin:
                                dmin = d
                                bestnode = node
                                snap = True
                if snap:
                    linenodes_id.append(bestnode)
                else:
                    vid = str(linenode["full_id"]) + "-" + str(linenode["vertex_index"])
                    G.add_node(vid, pos=(linenode["$x"], linenode["$y"]),
                               power=linenode["power"], substation=linenode["substation"], name=linenode["name"],
                               operator=linenode["operator"], voltage=linenode["voltage"])
                    linenodes_id.append(vid)
            for lnid1, lnid2 in zip(linenodes_id[:-1], linenodes_id[1:]):
                G.add_edge(lnid1, lnid2, power=row["power"], substation=row["substation"], name=row["name"],
                           operator=row["operator"], voltage=row["voltage"])

    return G


if __name__ == "__main__":

    # P.O.1 Load networkx graph
    try:
        G = pickle.load(open(r'data/power_osm/out/power_OSM_G.pkl', "rb"))
    except IOError:
        point, polyg, line, linevertex = load_power_osm_data()
        G = create_power_osm_graph(point, polyg, line, linevertex)
        try:
            pickle.dump(G, open(r'data/power_osm/out/power_OSM_G.pkl', 'wb+'))
        except FileNotFoundError as e:
            print(e)

    # P.O.2 Export graph nodelist
    try:
        combined_nodelist = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
        combined_nodelist = combined_nodelist.rename_axis('full_id')
        combined_nodelist.to_excel(r'data/power_osm/out/power_OSM_nodelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # P.O.3 Export adjacency matrix
    try:
        combined_adjmat = nx.adjacency_matrix(G, nodelist=None, weight="weight")  # gives scipy sparse matrix
        sparse.save_npz(r'data/power_osm/out/power_OSM_adjmat.npz', combined_adjmat)
    except Exception as e:
        print(e)

    # P.O.4 Export graph edgelist
    try:
        edgelist = pd.DataFrame(columns=["lnid1", "lnid2", "power", "substation", "name", "operator", "voltage"])
        for u, v in G.edges():
            series_obj = pd.Series([u, v,
                                    G.edges[u, v]["power"],
                                    G.edges[u, v]["substation"],
                                    G.edges[u, v]["name"],
                                    G.edges[u, v]["operator"],
                                    G.edges[u, v]["voltage"]],
                                   index=edgelist.columns)
            edgelist = edgelist.append(series_obj, ignore_index=True)
        edgelist.to_excel(r'data/power_osm/out/power_osm_edgelist.xlsx', index=True)
    except Exception as e:
        print(e)

    # P.O.5 Add colours and export map plot
    node_colors = [POWER_COLORS.get(G.nodes[node]["power"], "#000000") for node in G.nodes()]
    edge_powers = nx.get_edge_attributes(G, "power")
    edge_colors = [POWER_COLORS.get(edge_powers[u, v], "#000000") for u, v in G.edges()]

    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=5,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig("data/power_osm/img/power_OSM.png")
    plt.savefig("data/power_osm/img/power_OSM.svg")
    plt.show()

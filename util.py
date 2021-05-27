from math import radians, cos, sin, asin, sqrt

import networkx as nx

from globalparams import POWER_LINE_PP_EPS
import json
import numpy as np


class CustomEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle numpy variables"""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)  # cast numpy int to vanilla int
        elif isinstance(o, int):
            return int(o)
        elif isinstance(o, np.float):
            return round(float(o), 6)  # cast numpy float to vanilla float
        else:
            return super().default(o)  # otherwise fall back to base implementation in JSONEncoder


def haversine(lon1, lat1, lon2, lat2):
    """Calculates distance between two sets of coordinates"""
    # https://gist.github.com/Tofull/49fbb9f3661e376d2fe08c2e9d64320e

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r  # Returns distance in km


def perpendicular_dist(x, y, x1, y1, x2, y2):
    """Calculates perpendicular distance between a point and a line represented by two points"""
    # https://stackoverflow.com/a/6853926/5881159

    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C ** 2 + D ** 2
    param = dot / len_sq if len_sq != 0 else -1

    xx = x1 if param < 0 else x2 if param > 1 else x1 + param * C
    yy = y1 if param < 0 else y2 if param > 1 else y1 + param * D
    dx = x - xx
    dy = y - yy

    return sqrt(dx ** 2 + dy ** 2)


def douglas_peucker(linenodes, eps=POWER_LINE_PP_EPS):
    """Simplifies multi-line geometry"""
    # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm#Pseudocode

    linenodes.reset_index(inplace=True, drop=True)

    # Find point with maximum perpendicular distance from line between first & last nodes
    dmax = -1.
    index = -1
    firstnode = linenodes.head(1)
    lastnode = linenodes.tail(1)
    for i, linenode in linenodes[1:-1].iterrows():
        d = perpendicular_dist(linenode["$x"], linenode["$y"],
                               linenodes.at[0, "$x"], linenodes.at[0, "$y"],
                               linenodes.at[len(linenodes)-1, "$x"], linenodes.at[len(linenodes)-1, "$y"])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > eps:
        linenodes_res_1 = douglas_peucker(linenodes[:index], eps=eps)
        linenodes_res_2 = douglas_peucker(linenodes[index:], eps=eps)
        linenodes_res = linenodes_res_1[:-1].append(linenodes_res_2)
    else:
        linenodes_res = firstnode.append(lastnode)

    return linenodes_res


def weakly_connected_component_subgraphs(G, copy=True):
    """Generate weakly connected components as subgraphs.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph.

    copy: bool (default=True)
        If True make a copy of the graph attributes

    Returns
    -------
    comp : generator
        A generator of graphs, one for each weakly connected component of G.

    Examples
    --------
    Generate a sorted list of weakly connected components, largest first.

    >>> G = nx.path_graph(4, create_using=nx.DiGraph())
    >>> G.add_path([10, 11, 12])
    >>> [len(c) for c in sorted(nx.weakly_connected_component_subgraphs(G),
    ...                         key=len, reverse=True)]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> Gc = max(nx.weakly_connected_component_subgraphs(G), key=len)

    See Also
    --------
    strongly_connected_components
    connected_components

    Notes
    -----
    For directed graphs only.
    Graph, node, and edge attributes are copied to the subgraphs by default.

    """
    for comp in nx.weakly_connected_components(G):
        if copy:
            yield G.subgraph(comp).copy()
        else:
            yield G.subgraph(comp)


# TODO Proof-of-concept, not tested yet
def cluster(G, nodes_to_cluster):
    G1 = G
    for node in nodes_to_cluster[1:]:
        G1 = nx.contracted_nodes(G1, nodes_to_cluster[0], node, self_loops=False, copy=True)
    return G1


# TODO Flow balancing for electricity network



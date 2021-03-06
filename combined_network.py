import pickle
import pandas as pd
import networkx as nx
from mayavi import mlab
import numpy as np

# FYI Under Environment Variables, add ETS_TOOLKIT = qt4

# Load individual networks
from globalparams import TRANSPORT_COLORS, POWER_COLORS, TRANSPORT_COLORS_MINI, POWER_COLORS_MINI
from util import hex_to_rgb

SIMPLIFIED = True

if SIMPLIFIED:
    G_transport = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow_skele.pkl', "rb"))
else:
    G_transport = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', "rb"))
G_power = pickle.load(open(r'data/power_ukpn/out/power_ukpn_G_flow.pkl', "rb"))

# Extend coordinate systems of individual networks into 3D
# Tag nodes/links as belonging to specific network
# Tag state of nodes/links (working = 1, failed = 0)
for n in G_transport.nodes():
    G_transport.nodes[n]["pos"] = (G_transport.nodes[n]["pos"][0], G_transport.nodes[n]["pos"][1], 0)
    # 0 if G_transport.nodes[n]["line"].startswith(("lo-", "tfl-rail-")) else 0.1
    G_transport.nodes[n]["network"] = "transport"
    G_transport.nodes[n]["state"] = 1
for u, v in G_transport.edges():
    G_transport.edges[u, v]["network"] = "transport"
    G_transport.edges[u, v]["state"] = 1
for n in G_power.nodes():
    G_power.nodes[n]["pos"] = (G_power.nodes[n]["pos"][0], G_power.nodes[n]["pos"][1], 0.2
                               # if G_power.nodes[n]["type"] in ["GSP_transmission", "substation_transmission"]
                               # else 0.9 if G_power.nodes[n]["type"] == "generator"
                               # else 0.7 if G_power.nodes[n]["type"] == "load"
                               # else 0.4
                               )
    G_power.nodes[n]["network"] = "power"
    G_power.nodes[n]["state"] = 1
for u, v in G_power.edges():
    G_power.edges[u, v]["network"] = "power"
    G_power.edges[u, v]["state"] = 1

# Combine networks
G_infra = nx.compose(G_transport, G_power)

# Load interdependencies
physical_intdep = pd.read_excel("data/interdependencies/physical.xlsx", sheet_name="physical", header=0)
for index, row in physical_intdep.iterrows():
    if SIMPLIFIED:
        G_infra.add_edge(row["UKPN_substation"], row["Train_simpl_ID"],
                         network="physical_interdependency", state=1)
    else:
        G_infra.add_edge(row["UKPN_substation"], row["Train_multiplex_ID"],
                         network="physical_interdependency", state=1)

# Save combined network
if SIMPLIFIED:
    pickle.dump(G_infra, open(r'data/combined_network/infra_G_skele.pkl', 'wb+'))
else:
    pickle.dump(G_infra, open(r'data/combined_network/infra_G.pkl', 'wb+'))

# Plot combined network
# network_plot_3D(G_infra, 120)

G_infra_renum = nx.convert_node_labels_to_integers(G_infra)
xyz = np.array([G_infra_renum.nodes[n]["pos"] for n in sorted(G_infra_renum)])
# xyz = np.array([G_infra.nodes[n]["pos"] for n in G_infra.nodes()])

mlab.figure(1)
mlab.clf()
node_scalars = np.array(list(G_infra_renum.nodes()))
node_colors = np.array([hex_to_rgb(TRANSPORT_COLORS_MINI.get(G_infra_renum.nodes[n]["line"], "#0009AB"))
               if G_infra_renum.nodes[n]["network"] == "transport"
               else hex_to_rgb(POWER_COLORS_MINI.get(G_infra_renum.nodes[n]["type"], "#000000"))
               for n in G_infra_renum.nodes()])
edge_scalars = np.array(list(G_infra_renum.edges()))
edge_lines = nx.get_edge_attributes(G_infra_renum, "Line")
edge_colors = np.array([hex_to_rgb(TRANSPORT_COLORS_MINI.get(edge_lines[u, v], "#0033CC"))
               if G_infra_renum.edges[u, v]["network"] == "transport"
               else hex_to_rgb(POWER_COLORS_MINI.get("cable", "#000000"))
               for u, v in G_infra_renum.edges()])
pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    node_scalars,
                    scale_factor=0.01,
                    scale_mode='none',
                    # colormap='Blues',
                    resolution=20)
pts.glyph.color_mode = 'color_by_scalar'
pts.module_manager.scalar_lut_manager.lut.table = node_colors
pts.mlab_source.dataset.lines = np.array(list(G_infra_renum.edges()))
tube = mlab.pipeline.tube(pts, tube_radius=0.001)
mlab.pipeline.surface(tube, color=(51./255., 102./255., 153./255.))  # (0.0, 0.0, 0.0)
mlab.show()

import pickle
import pandas as pd
import networkx as nx
from mayavi import mlab
import numpy as np

# Load individual networks
from util import network_plot_3D

G_transport = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G_flow.pkl', "rb"))
G_power = pickle.load(open(r'data/power_ukpn/out/power_ukpn_G_flow.pkl', "rb"))

# Extend coordinate systems of individual networks into 3D
for n in G_transport.nodes():
    G_transport.nodes[n]["pos"] = (G_transport.nodes[n]["pos"][0], G_transport.nodes[n]["pos"][1], 0)
for n in G_power.nodes():
    G_power.nodes[n]["pos"] = (G_power.nodes[n]["pos"][0], G_power.nodes[n]["pos"][1], 0.5)

# Combine networks
G_infra = nx.compose(G_transport, G_power)

# Load interdependencies
physical_intdep = pd.read_excel("data/interdependencies/physical.xlsx", sheet_name="physical", header=0)
for index, row in physical_intdep.iterrows():
    G_infra.add_edge(row["UKPN_substation"], row["Train_multiplex_ID"], type="physical_interdependency")

# Plot combined network
network_plot_3D(G_infra, 120)

# # G_infra_renum = nx.convert_node_labels_to_integers(G_infra)
# # xyz = np.array([G_infra_renum.nodes[n]["pos"] for n in sorted(G_infra_renum)])
# xyz = np.array([G_infra.nodes[n]["pos"] for n in G_infra.nodes()])
#
# mlab.figure(1, bgcolor=(0, 0, 0))
# mlab.clf()
# scalars = np.array(list(G_infra.nodes())) + 5
# pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
#                     scalars,
#                     scale_factor=0.1,
#                     scale_mode='none',
#                     colormap='Blues',
#                     resolution=20)
# pts.mlab_source.dataset.lines = np.array(list(G_infra.edges()))
# mlab.show()

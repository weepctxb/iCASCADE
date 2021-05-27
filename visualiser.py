import pickle
import folium
import numpy as np
import cvxpy as cp

from globalparams import POWER_COLORS, TRANSPORT_COLORS, LONDON_COORDS


def visualiser_power_osm():

    power_osm_G = pickle.load(open(r'data/power_osm/out/power_OSM_G.pkl', "rb"))

    london_power_osm_map = folium.Map(location=LONDON_COORDS, zoom_start=10, prefer_canvas=True)

    for node in power_osm_G.nodes():
        marker_popup_desc = "ID: " + str(node) + " | "
        for attr in power_osm_G.nodes[node].keys():
            if power_osm_G.nodes[node][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(power_osm_G.nodes[node][attr]) + " | "
        folium.CircleMarker(location=(power_osm_G.nodes[node]["pos"][1],
                                      power_osm_G.nodes[node]["pos"][0]),
                            popup=marker_popup_desc,
                            # popup="ID: " + str(node) +
                            #       " | Power: " + str(power_osm_G.nodes[node].get("power", "-")) +
                            #       " | Name: " + str(power_osm_G.nodes[node].get("name", "-")) +
                            #       " | Operator: " + str(power_osm_G.nodes[node].get("operator", "-")) +
                            #       " | Substation: " + str(power_osm_G.nodes[node].get("substation", "-")) +
                            #       " | Voltage: " + str(power_osm_G.nodes[node].get("voltage", "-")),
                            radius=5,
                            color=POWER_COLORS.get(power_osm_G.nodes[node]["power"], "#000000"),
                            fill=True, fillOpacity=1).add_to(london_power_osm_map)

    for u, v in power_osm_G.edges():
        marker_popup_desc = "ID: " + str(u) + "-" + str(v) + " | "
        for attr in power_osm_G.edges[u, v].keys():
            if power_osm_G.edges[u, v][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(power_osm_G.edges[u, v][attr]) + " | "
        folium.PolyLine(locations=[(power_osm_G.nodes[u]["pos"][1], power_osm_G.nodes[u]["pos"][0]),
                                   (power_osm_G.nodes[v]["pos"][1], power_osm_G.nodes[v]["pos"][0])],
                        popup=marker_popup_desc,
                        # popup="ID: " + str(u) + "-" + str(v) +
                        #       " | Power: " + str(power_osm_G.edges[u, v].get("power", "-")) +
                        #       " | Name: " + str(power_osm_G.edges[u, v].get("name", "-")) +
                        #       " | Operator: " + str(power_osm_G.edges[u, v].get("operator", "-")) +
                        #       " | Substation: " + str(power_osm_G.edges[u, v].get("substation", "-")) +
                        #       " | Voltage: " + str(power_osm_G.edges[u, v].get("voltage", "-")),
                        weight=3,
                        color=POWER_COLORS.get(power_osm_G.edges[u, v].get("power", "-"), "#000000"),
                        fill=True, fillOpacity=1).add_to(london_power_osm_map)

    html_path_power_osm = f'data/visualiser/power_osm.html'
    london_power_osm_map.save(html_path_power_osm)


def visualiser_power_ukpn():

    power_ukpn_G = pickle.load(open(r'data/power_ukpn/out/power_UKPN_G.pkl', "rb"))

    london_power_ukpn_map = folium.Map(location=LONDON_COORDS, zoom_start=10, prefer_canvas=True)

    for node in power_ukpn_G.nodes():
        marker_popup_desc = "ID: " + str(node) + " | "
        for attr in power_ukpn_G.nodes[node].keys():
            if isinstance(power_ukpn_G.nodes[node][attr], cp.Variable):
                if power_ukpn_G.nodes[node][attr].value not in ["nan", "", "-", np.nan, None]:
                    marker_popup_desc += str(attr) + ": " + str(power_ukpn_G.nodes[node][attr].value) + " | "
            else:
                if power_ukpn_G.nodes[node][attr] not in ["nan", "", "-", np.nan, None]:
                    marker_popup_desc += str(attr) + ": " + str(power_ukpn_G.nodes[node][attr]) + " | "
        folium.CircleMarker(location=(power_ukpn_G.nodes[node]["pos"][1],
                                      power_ukpn_G.nodes[node]["pos"][0]),
                            popup=marker_popup_desc,
                            # popup="ID: " + str(node) +
                            #       " | Power: " + str(power_ukpn_G.nodes[node].get("power", "-")) +
                            #       " | Name: " + str(power_ukpn_G.nodes[node].get("name", "-")) +
                            #       " | Operator: " + str(power_ukpn_G.nodes[node].get("operator", "-")) +
                            #       " | Substation: " + str(power_ukpn_G.nodes[node].get("substation", "-")) +
                            #       " | Voltage: " + str(power_ukpn_G.nodes[node].get("voltage", "-")),
                            radius=5,
                            color=POWER_COLORS.get(power_ukpn_G.nodes[node]["type"], "#000000"),
                            fill=True, fillOpacity=1).add_to(london_power_ukpn_map)

    for u, v in power_ukpn_G.edges():
        marker_popup_desc = "ID: " + str(u) + "-" + str(v) + " | "
        for attr in power_ukpn_G.edges[u, v].keys():
            if isinstance(power_ukpn_G.edges[u, v][attr], cp.Variable):
                if power_ukpn_G.edges[u, v][attr].value not in ["nan", "", "-", np.nan, None]:
                    marker_popup_desc += str(attr) + ": " + str(power_ukpn_G.edges[u, v][attr].value) + " | "
            else:
                if power_ukpn_G.edges[u, v][attr] not in ["nan", "", "-", np.nan, None]:
                    marker_popup_desc += str(attr) + ": " + str(power_ukpn_G.edges[u, v][attr]) + " | "
        folium.PolyLine(locations=[(power_ukpn_G.nodes[u]["pos"][1], power_ukpn_G.nodes[u]["pos"][0]),
                                   (power_ukpn_G.nodes[v]["pos"][1], power_ukpn_G.nodes[v]["pos"][0])],
                        popup=marker_popup_desc,
                        # popup="ID: " + str(u) + "-" + str(v) +
                        #       " | Power: " + str(power_ukpn_G.edges[u, v].get("power", "-")) +
                        #       " | Name: " + str(power_ukpn_G.edges[u, v].get("name", "-")) +
                        #       " | Operator: " + str(power_ukpn_G.edges[u, v].get("operator", "-")) +
                        #       " | Substation: " + str(power_ukpn_G.edges[u, v].get("substation", "-")) +
                        #       " | Voltage: " + str(power_ukpn_G.edges[u, v].get("voltage", "-")),
                        weight=3,
                        color=POWER_COLORS.get(power_ukpn_G.edges[u, v].get("type", "-"), "#000000"),
                        fill=True, fillOpacity=1).add_to(london_power_ukpn_map)

    html_path_power_ukpn = f'data/visualiser/power_ukpn.html'
    london_power_ukpn_map.save(html_path_power_ukpn)


def visualiser_transport_multiplex():

    transport_multiplex_G = pickle.load(open(r'data/transport_multiplex/out/transport_multiplex_G.pkl', "rb"))

    london_transport_multiplex_map = folium.Map(location=LONDON_COORDS, zoom_start=10, prefer_canvas=True)

    for node in transport_multiplex_G.nodes():
        marker_popup_desc = "ID: " + str(node) + " | "
        for attr in transport_multiplex_G.nodes[node].keys():
            if transport_multiplex_G.nodes[node][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(transport_multiplex_G.nodes[node][attr]) + " | "
        folium.CircleMarker(location=(transport_multiplex_G.nodes[node]["pos"][1],
                                      transport_multiplex_G.nodes[node]["pos"][0]),
                            popup=marker_popup_desc,
                            # popup="ID: " + str(node) +
                            #       " | Name: " + str(transport_multiplex_G.nodes[node].get("nodeLabel", "-")) +
                            #       " | Interchange: " + str(transport_multiplex_G.nodes[node].get("interchange", "-")) +
                            #       " | Line: " + str(transport_multiplex_G.nodes[node].get("line", "-")),
                            radius=5,
                            color=TRANSPORT_COLORS.get(transport_multiplex_G.nodes[node]["line"], "#000000"),
                            fill=True, fillOpacity=1).add_to(london_transport_multiplex_map)

    for u, v in transport_multiplex_G.edges():
        marker_popup_desc = "ID: " + str(u) + "-" + str(v) + " | "
        for attr in transport_multiplex_G.edges[u, v].keys():
            if transport_multiplex_G.edges[u, v][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(transport_multiplex_G.edges[u, v][attr]) + " | "
        folium.PolyLine(locations=[(transport_multiplex_G.nodes[u]["pos"][1],
                                    transport_multiplex_G.nodes[u]["pos"][0]),
                                   (transport_multiplex_G.nodes[v]["pos"][1],
                                    transport_multiplex_G.nodes[v]["pos"][0])],
                        popup=marker_popup_desc,
                        # popup="ID: " + str(u) + "-" + str(v) +
                        #       " | Line: " + str(transport_multiplex_G.edges[u, v].get("Line", "-")) +
                        #       " | StationA: " + str(transport_multiplex_G.edges[u, v].get("StationA", "-")) +
                        #       " | StationB: " + str(transport_multiplex_G.edges[u, v].get("StationB", "-")) +
                        #       " | Distance: " + str(transport_multiplex_G.edges[u, v].get("Distance", "-")),
                        weight=3,
                        color=TRANSPORT_COLORS.get(transport_multiplex_G.edges[u, v].get("Line", "-"), "#000000"),
                        fill=True, fillOpacity=1).add_to(london_transport_multiplex_map)

    html_path_transport_multiplex = f'data/visualiser/transport_multiplex.html'
    london_transport_multiplex_map.save(html_path_transport_multiplex)


def visualiser_transport_osm():

    transport_osm_G = pickle.load(open(r'data/transport_osm/out/transport_OSM_G.pkl', "rb"))

    london_transport_osm_map = folium.Map(location=LONDON_COORDS, zoom_start=10, prefer_canvas=True)

    for node in transport_osm_G.nodes():
        marker_popup_desc = "ID: " + str(node) + " | "
        for attr in transport_osm_G.nodes[node].keys():
            if transport_osm_G.nodes[node][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(transport_osm_G.nodes[node][attr]) + " | "
        folium.CircleMarker(location=(transport_osm_G.nodes[node]["pos"][1],
                                      transport_osm_G.nodes[node]["pos"][0]),
                            popup=marker_popup_desc,
                            # popup="ID: " + str(node) +
                            #       " | Railway: " + str(transport_osm_G.nodes[node].get("railway", "-")) +
                            #       " | Network: " + str(transport_osm_G.nodes[node].get("network", "-")) +
                            #       " | Line: " + str(transport_osm_G.nodes[node].get("line", "-")) +
                            #       " | Name: " + str(transport_osm_G.nodes[node].get("name", "-")) +
                            #       " | Operator: " + str(transport_osm_G.nodes[node].get("operator", "-")) +
                            #       " | Electrified: " + str(transport_osm_G.nodes[node].get("electrified", "-")),
                            radius=5,
                            color=TRANSPORT_COLORS.get(transport_osm_G.nodes[node]["railway"], "#000000"),
                            fill=True, fillOpacity=1).add_to(london_transport_osm_map)

    html_path_transport_osm = f'data/visualiser/transport_osm.html'

    london_transport_osm_map.save(html_path_transport_osm)


def visualiser_transport_combined():

    transport_combined_G = pickle.load(open(r'data/transport_combined/transport_combined_G.pkl', "rb"))

    london_transport_combined_map = folium.Map(location=LONDON_COORDS, zoom_start=10, prefer_canvas=True)

    for node in transport_combined_G.nodes():
        marker_popup_desc = "ID: " + str(node) + " | "
        for attr in transport_combined_G.nodes[node].keys():
            if transport_combined_G.nodes[node][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(transport_combined_G.nodes[node][attr]) + " | "
        folium.CircleMarker(location=(transport_combined_G.nodes[node]["pos"][1],
                                      transport_combined_G.nodes[node]["pos"][0]),
                            popup=marker_popup_desc,
                            # popup="ID: " + str(node) +
                            #       " | Railway: " + str(transport_combined_G.nodes[node].get("railway", "-")) +
                            #       " | Network: " + str(transport_combined_G.nodes[node].get("network", "-")) +
                            #       " | Line: " + str(transport_combined_G.nodes[node].get("line", "-")) +
                            #       " | Name: " + str(transport_combined_G.nodes[node].get("nodeLabel",
                            #                                                              transport_combined_G.nodes[node].get("name", "-"))) +
                            #       " | Interchange: " + str(transport_combined_G.nodes[node].get("interchange", "-")) +
                            #       " | Operator: " + str(transport_combined_G.nodes[node].get("operator", "-")) +
                            #       " | Electrified: " + str(transport_combined_G.nodes[node].get("electrified", "-")),
                            radius=5,
                            color=TRANSPORT_COLORS.get(
                                transport_combined_G.nodes[node].get("line", "-")
                                if transport_combined_G.nodes[node].get("railway", "-") == "station"
                                else transport_combined_G.nodes[node].get("railway", "-"),
                                "#FF0000"),
                            fill=True, fillOpacity=1).add_to(london_transport_combined_map)

    for u, v in transport_combined_G.edges():
        marker_popup_desc = "ID: " + str(u) + "-" + str(v) + " | "
        for attr in transport_combined_G.edges[u, v].keys():
            if transport_combined_G.edges[u, v][attr] not in ["nan", "", "-", np.nan, None]:
                marker_popup_desc += str(attr) + ": " + str(transport_combined_G.edges[u, v][attr]) + " | "
        folium.PolyLine(locations=[(transport_combined_G.nodes[u]["pos"][1], transport_combined_G.nodes[u]["pos"][0]),
                                   (transport_combined_G.nodes[v]["pos"][1], transport_combined_G.nodes[v]["pos"][0])],
                        popup=marker_popup_desc,
                        # popup="ID: " + str(u) + "-" + str(v) +
                        #       " | Line: " + str(transport_combined_G.edges[u, v].get("Line", "-")) +
                        #       " | StationA: " + str(transport_combined_G.edges[u, v].get("StationA", "-")) +
                        #       " | StationB: " + str(transport_combined_G.edges[u, v].get("StationB", "-")) +
                        #       " | Distance: " + str(transport_combined_G.edges[u, v].get("Distance", "-")),
                        weight=3,
                        color=TRANSPORT_COLORS.get(transport_combined_G.edges[u, v].get("Line", "-"), "#000000"),
                        fill=True, fillOpacity=1).add_to(london_transport_combined_map)

    html_path_transport_combined = f'data/visualiser/transport_combined.html'

    london_transport_combined_map.save(html_path_transport_combined)


if __name__ == "__main__":
    # visualiser_power_osm()
    visualiser_power_ukpn()
    # visualiser_transport_multiplex()
    # visualiser_transport_osm()

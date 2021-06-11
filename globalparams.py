# https://www.w3schools.com/colors/colors_wheels.asp

POWER_COLORS = {
    "cable": "#000000",  # 336600
    "line": "#000000",  # #000099
    "minor_line": "#6666ff",
    "substation": "#FE2712",
    "substation_transmission": "#990000",
    "substation_traction": "#FF33CC",
    "sub_station": "#FE2712",
    "transformer": "#FB9902",
    "terminal": "#FC600A",
    "generator": "#66B032",
    "plant": "#B2D732",
    "switchgear": "#8601AF",
    "switch": "#8601AF",
    "compensator": "#808080",
    "pole": "#996633",
    "tower": "#604020",
    "load": "#CC66FF"
}

TRANSPORT_COLORS = {
    "bakerloo": "#894E24",
    "central": "#DC241F",
    "circle": "#FFCE00",
    "district": "#007229",
    "hammersmith&city": "#D799AF",
    "jubilee": "#868F98",
    "metropolitan": "#751056",
    "northern": "#000000",
    "piccadilly": "#0019A8",
    "victoria": "#00A0E2",
    "waterloo&city": "#76D0BD",
    "lo-watform-euston": "#E86A10",
    "lo-north-london": "#E86A10",
    "lo-east-london": "#E86A10",
    "lo-gospeloak-barking": "#E86A10",
    "dlr": "#00AFAD",
    "lo-westanglia": "#E86A10",
    "lo-romford-upminster": "#E86A10",
    "tfl-rail-shenfield": "#0019a8",
    "tfl-rail-reading-heathrow": "#0019A8",
    "trams": "#66CC00",
    "power_supply": "#FF33CC",
    "signal": "#00FFCC",
    "site": "#FF9900",
    "switch": "#D1D1E0",
    "yard": "#00FFCC",
    "depot": "#FF9900",
    "engine_shed": "#FF9900",
    "signal_box": "#00FFCC"
}

LONDON_COORDS = [51.5085300, -0.1257400]

POWER_LINE_PP_EPS = 0.005
POWER_PROXIMITY_THRESHOLD = 0.15
TRANSPORT_LINE_PP_EPS = 0  # FIXME disabled first, used to be 0.005
TRANSPORT_PROXIMITY_THRESHOLD = 0.6

LOAD_CAP = 1.2
TRAIN_DUR_THRESHOLD_MIN = 1e5  # 60
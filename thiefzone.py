"""Load the zonal petrophysical table, check its roll-ups, run kNN and compute flow metrics.

Rows sit at three levels (formation, unit, zone) and each parent row is a
weighted average of its children.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "petrophysical_data.xlsx"

COLUMNS = ["zone", "top", "base", "ng", "phif", "sw", "k_ari", "k_har", "k_geo"]
PERM_COLUMNS = ["k_ari", "k_har", "k_geo"]

HIERARCHY = {
    "Zechstein": ("formation", None, "Zechstein"),
    "Devonian": ("formation", None, "Devonian"),
    "Upper Devonian": ("unit", "Devonian", "Devonian"),
    "Lower Devonian": ("unit", "Devonian", "Devonian"),
    "Devonian 3.3 (12)": ("zone", "Upper Devonian", "Devonian"),
    "Devonian 3.2 (11)": ("zone", "Upper Devonian", "Devonian"),
    "Devonian 3.1 (10)": ("zone", "Upper Devonian", "Devonian"),
    "Rotliegend": ("formation", None, "Rotliegend"),
    "Crystalline basement": ("formation", None, "Basement"),
    "Top basement": ("formation", None, "Basement"),
}

PARENT_ROWS = ["Devonian", "Upper Devonian"]

LEAF_ZONES = [
    "Devonian 3.3 (12)",
    "Devonian 3.2 (11)",
    "Devonian 3.1 (10)",
    "Lower Devonian",
    "Rotliegend",
    "Crystalline basement",
    "Top basement",
]

SUSPECT_NAME = "Suspect zone"

SUSPECT = {
    "top": 3455.0,
    "base": 3464.0,
    "ng": 0.923,
    "phif": 0.196,
    "sw": 0.939,
    "k_ari": 1280.0,
    "k_har": 48.0,
    "k_geo": 358.0,
}

FEATURE_VIEWS = {
    "stratigraphic": ["top", "base"],
    "petrophysical": ["ng", "phif", "sw", "log_k_ari", "log_k_har", "log_k_geo"],
    "all": ["top", "base", "ng", "phif", "sw", "log_k_ari", "log_k_har", "log_k_geo"],
}

PORT_CLASSES = [
    (10.0, "megaport"),
    (2.0, "macroport"),
    (0.5, "mesoport"),
    (0.1, "microport"),
    (0.0, "nanoport"),
]


def load_zone_table(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the worksheet as one tidy row per zone, with its place in the hierarchy."""
    raw = pd.read_excel(path, sheet_name="petrophysical_data", header=None)
    table = raw.iloc[3:, 1:10].copy()
    table.columns = COLUMNS
    table = table.dropna(how="all").reset_index(drop=True)
    table["zone"] = table["zone"].str.strip()
    for column in COLUMNS[1:]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["h"] = table["base"] - table["top"]
    table["h_net"] = table["h"] * table["ng"]
    table["level"] = table["zone"].map(lambda z: HIERARCHY[z][0])
    table["parent"] = table["zone"].map(lambda z: HIERARCHY[z][1])
    table["formation"] = table["zone"].map(lambda z: HIERARCHY[z][2])
    return table


def add_log_perm(frame: pd.DataFrame) -> pd.DataFrame:
    """Add base-10 log permeability columns used as kNN features."""
    frame = frame.copy()
    for column in PERM_COLUMNS:
        frame[f"log_{column}"] = np.log10(frame[column])
    return frame


def suspect_frame(**overrides: float) -> pd.DataFrame:
    """The suspect zone as a one-row frame."""
    values = {**SUSPECT, **overrides}
    frame = pd.DataFrame([{"zone": SUSPECT_NAME, **values}])
    frame["h"] = frame["base"] - frame["top"]
    frame["h_net"] = frame["h"] * frame["ng"]
    frame["formation"] = np.nan
    return add_log_perm(frame)


def roll_up(rows: pd.DataFrame) -> dict[str, float]:
    """Average child rows the way the parent rows in the table were built.

    Net-to-gross is net over gross thickness. Porosity and the three permeability
    averages are weighted by net thickness; water saturation by pore volume.
    """
    h = rows["h"].to_numpy(float)
    h_net = (rows["h"] * rows["ng"]).to_numpy(float)
    phif = rows["phif"].to_numpy(float)
    pore_volume = h_net * phif
    return {
        "h": h.sum(),
        "ng": h_net.sum() / h.sum(),
        "phif": (h_net * phif).sum() / h_net.sum(),
        "sw": (pore_volume * rows["sw"].to_numpy(float)).sum() / pore_volume.sum(),
        "k_ari": (h_net * rows["k_ari"].to_numpy(float)).sum() / h_net.sum(),
        "k_geo": float(np.exp((h_net * np.log(rows["k_geo"].to_numpy(float))).sum() / h_net.sum())),
        "k_har": h_net.sum() / (h_net / rows["k_har"].to_numpy(float)).sum(),
    }


def reconstruct_gap(table: pd.DataFrame) -> dict[str, float]:
    """Back-calculate the unnamed interval between the Upper and Lower Devonian.

    Each average is converted to a total, the two units are subtracted from the
    Devonian, and the remainder is divided by the interval's net thickness. The
    geometric and harmonic means are rounded too coarsely in the table to invert.
    """
    row = table.set_index("zone")
    parent, upper, lower = row.loc["Devonian"], row.loc["Upper Devonian"], row.loc["Lower Devonian"]
    top, base = upper["base"], lower["top"]
    h_net = parent["h_net"] - upper["h_net"] - lower["h_net"]

    def net_weighted(column: str) -> float:
        total = parent["h_net"] * parent[column] - upper["h_net"] * upper[column] - lower["h_net"] * lower[column]
        return total / h_net

    phif = net_weighted("phif")
    pore_volume = {name: r["h_net"] * r["phif"] for name, r in (("parent", parent), ("upper", upper), ("lower", lower))}
    sw = (
        pore_volume["parent"] * parent["sw"] - pore_volume["upper"] * upper["sw"] - pore_volume["lower"] * lower["sw"]
    ) / (h_net * phif)
    return {
        "top": top,
        "base": base,
        "h": base - top,
        "h_net": h_net,
        "ng": h_net / (base - top),
        "phif": phif,
        "sw": sw,
        "k_ari": net_weighted("k_ari"),
    }


def leaf_catalogue(table: pd.DataFrame, include_zechstein: bool = False) -> pd.DataFrame:
    """Non-overlapping zones in stratigraphic order, ready to use as kNN training rows."""
    names = (["Zechstein"] if include_zechstein else []) + LEAF_ZONES
    catalogue = table.set_index("zone").loc[names].reset_index()
    return add_log_perm(catalogue)


def devonian_column(table: pd.DataFrame) -> pd.DataFrame:
    """The Devonian zones with the suspect zone in the gap (3454.7 to 3463.8 m)."""
    gap = reconstruct_gap(table)
    leaves = table.set_index("zone").loc[LEAF_ZONES[:4]].reset_index()
    suspect = suspect_frame(top=gap["top"], base=gap["base"])
    column = pd.concat([leaves.iloc[:3], suspect, leaves.iloc[3:]], ignore_index=True)
    column["h"] = column["base"] - column["top"]
    column["h_net"] = column["h"] * column["ng"]
    return column[["zone", "top", "base", "h", "h_net", "ng", "phif", "sw", "k_ari", "k_har", "k_geo"]]


def flow_profile(column: pd.DataFrame) -> pd.DataFrame:
    """Flow capacity (kh), storage capacity (phi h), their shares and relative front speed (k / phi) per layer."""
    profile = column.copy()
    profile["kh"] = profile["k_ari"] * profile["h_net"]
    profile["phih"] = profile["phif"] * profile["h_net"]
    profile["flow_share"] = profile["kh"] / profile["kh"].sum()
    profile["storage_share"] = profile["phih"] / profile["phih"].sum()
    profile["net_share"] = profile["h_net"] / profile["h_net"].sum()
    profile["flow_to_storage"] = profile["flow_share"] / profile["storage_share"]
    profile["speed"] = profile["k_ari"] / profile["phif"]
    return profile


def smlp_curve(profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative storage and flow capacity in stratigraphic order (Gunter et al., 1997)."""
    storage = np.r_[0.0, profile["phih"].cumsum() / profile["phih"].sum()]
    flow = np.r_[0.0, profile["kh"].cumsum() / profile["kh"].sum()]
    return storage, flow


def winland_r35(k_md, phi_fraction):
    """Pore-throat radius in microns at 35% mercury saturation (Kolodzie, 1980; Pittman, 1992)."""
    log_r35 = 0.732 + 0.588 * np.log10(k_md) - 0.864 * np.log10(np.asarray(phi_fraction) * 100.0)
    return 10.0 ** log_r35


def port_class(r35: float) -> str:
    """Port-size class of a pore-throat radius (Martin et al., 1997)."""
    for lower_bound, name in PORT_CLASSES:
        if r35 > lower_bound:
            return name
    return "nanoport"


def winland_k(r35_um: float, phi_fraction):
    """Permeability on a Winland iso-r35 line, for drawing port-class boundaries."""
    log_k = (np.log10(r35_um) - 0.732 + 0.864 * np.log10(np.asarray(phi_fraction) * 100.0)) / 0.588
    return 10.0 ** log_k


def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"unknown scaler: {kind}")


def locate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    k: int = 3,
    label: str = "formation",
    scaler: str = "standard",
) -> tuple[str, pd.DataFrame]:
    """Return the kNN vote for the test row and every training row ranked by distance.

    The scaler is fitted on the training rows only.
    """
    fitted = make_scaler(scaler).fit(train[features])
    x_train = fitted.transform(train[features])
    x_test = fitted.transform(test[features])
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, train[label])
    prediction = model.predict(x_test)[0]
    distance, index = NearestNeighbors(n_neighbors=len(train)).fit(x_train).kneighbors(x_test)
    neighbours = pd.DataFrame(
        {
            "rank": np.arange(1, len(train) + 1),
            "zone": train["zone"].to_numpy()[index[0]],
            "formation": train["formation"].to_numpy()[index[0]],
            "distance": distance[0].round(2),
        }
    )
    return prediction, neighbours


def nearest_by_property(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """For each property alone, the closest training zone and whether the test value is outside the training range."""
    rows = []
    for feature in features:
        values = train[feature]
        target = float(test[feature].iloc[0])
        gap = (values - target).abs() / values.std(ddof=0)
        nearest = gap.idxmin()
        rows.append(
            {
                "property": feature,
                "suspect": target,
                "nearest zone": train.loc[nearest, "zone"],
                "nearest value": values[nearest],
                "gap (sd)": round(float(gap[nearest]), 2),
                "outside training range": not (values.min() <= target <= values.max()),
            }
        )
    return pd.DataFrame(rows)

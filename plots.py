"""Figures for the notebook (light), presentations (dark) and the article (report, report-mono)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

import thiefzone as tz

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "devonian": "#2a78d6",
        "suspect": "#eb6834",
        "other": "#898781",
        "devonian_soft": "#6da7ec",
        "other_soft": "#b9b7b0",
        "devonian_tint": "#e4eefb",
        "suspect_tint": "#fbe1d5",
        "other_tint": "#efeee9",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "devonian": "#3987e5",
        "suspect": "#d95926",
        "other": "#898781",
        "devonian_soft": "#256abf",
        "other_soft": "#5d5c57",
        "devonian_tint": "#1c2c40",
        "suspect_tint": "#46261a",
        "other_tint": "#262624",
    },
    "report": {
        "surface": "#ffffff",
        "ink": "#111111",
        "secondary": "#444444",
        "muted": "#8a8a8a",
        "grid": "#e8e8e8",
        "axis": "#bdbdbd",
        "devonian": "#3d3d3d",
        "suspect": "#eb6834",
        "other": "#a6a6a6",
        "devonian_soft": "#8f8f8f",
        "other_soft": "#cfcfcf",
        "devonian_tint": "#ededed",
        "suspect_tint": "#fbe1d5",
        "other_tint": "#f7f7f7",
        "fonts": ["Space Grotesk", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "titles": False,
        "figscale": 0.82,
        "text_scale": 1.25,
    },
    "report-mono": {
        "surface": "#ffffff",
        "ink": "#111111",
        "secondary": "#444444",
        "muted": "#8a8a8a",
        "grid": "#e8e8e8",
        "axis": "#bdbdbd",
        "devonian": "#7a7a7a",
        "suspect": "#111111",
        "other": "#bdbdbd",
        "devonian_soft": "#a8a8a8",
        "other_soft": "#d9d9d9",
        "devonian_tint": "#f0f0f0",
        "suspect_tint": "#cfcfcf",
        "other_tint": "#f9f9f9",
        "tick_on_suspect": "#ffffff",
        "fonts": ["Space Grotesk", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "titles": False,
        "figscale": 0.82,
        "text_scale": 1.25,
    },
}

FONT_FAMILY = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]

SHORT_NAMES = {
    "Zechstein": "Zechstein",
    "Devonian 3.3 (12)": "Devonian 3.3",
    "Devonian 3.2 (11)": "Devonian 3.2",
    "Devonian 3.1 (10)": "Devonian 3.1",
    tz.SUSPECT_NAME: "Suspect zone",
    "Lower Devonian": "Lower Devonian",
    "Rotliegend": "Rotliegend",
    "Crystalline basement": "Crystalline basement",
    "Top basement": "Top basement",
}

PROPERTY_LABELS = {
    "top": "Top depth",
    "base": "Base depth",
    "ng": "Net-to-gross",
    "phif": "Porosity (PHIF)",
    "log_k_ari": "Permeability, arithmetic",
    "log_k_geo": "Permeability, geometric",
    "log_k_har": "Permeability, harmonic",
    "sw": "Water saturation",
}


@contextmanager
def themed(theme: str):
    colours = THEMES[theme]
    params = {
        "font.family": "sans-serif",
        "font.sans-serif": colours.get("fonts", FONT_FAMILY),
        "font.size": fs(10, colours),
        "figure.facecolor": colours["surface"],
        "axes.facecolor": colours["surface"],
        "savefig.facecolor": colours["surface"],
        "axes.edgecolor": colours["axis"],
        "axes.labelcolor": colours["secondary"],
        "axes.titlecolor": colours["ink"],
        "xtick.color": colours["muted"],
        "ytick.color": colours["muted"],
        "xtick.labelcolor": colours["secondary"],
        "ytick.labelcolor": colours["secondary"],
        "text.color": colours["ink"],
        "grid.color": colours["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.labelcolor": colours["secondary"],
    }
    with plt.rc_context(params):
        yield colours


def fs(size: float, colours: dict) -> float:
    return size * colours.get("text_scale", 1.0)


def size(dimensions: tuple[float, float], colours: dict) -> tuple[float, float]:
    scale = colours.get("figscale", 1.0)
    return (dimensions[0] * scale, dimensions[1] * scale)


def top_margin(default: float, colours: dict, untitled: float = 0.96) -> float:
    return default if colours.get("titles", True) else untitled


def is_compact(colours: dict) -> bool:
    return not colours.get("titles", True)


def zone_colour(zone: str, colours: dict, tint: bool = False, soft: bool = False) -> str:
    suffix = "_tint" if tint else "_soft" if soft else ""
    if zone == tz.SUSPECT_NAME:
        return colours["suspect" + ("_tint" if tint else "")]
    if "Devonian" in zone:
        return colours["devonian" + suffix]
    return colours["other" + suffix]


def add_title(fig, title: str, subtitle: str, colours: dict, x: float = 0.04) -> None:
    if not colours.get("titles", True):
        return
    fig.text(x, 0.955, title, fontsize=fs(15, colours), fontweight="semibold", color=colours["ink"], va="top")
    fig.text(x, 0.905, subtitle, fontsize=fs(10.5, colours), color=colours["secondary"], va="top")


def plain_log_ticks(axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))


def save(fig, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if path.suffix == ".svg" else None
    with plt.rc_context({"svg.hashsalt": "thief-zone-knn"}):
        fig.savefig(path, dpi=200, metadata=metadata)


def well_section(table, theme: str = "light", path=None):
    """Zone column, arithmetic permeability and share of Devonian flow capacity against depth."""
    named = table.set_index("zone")
    column = tz.devonian_column(table)
    profile = tz.flow_profile(column).set_index("zone")
    zones = [("Zechstein", named.loc["Zechstein", "top"], named.loc["Zechstein", "base"], np.nan)]
    zones += [(r["zone"], r["top"], r["base"], r["k_ari"]) for _, r in column.iterrows()]
    zones += [("Rotliegend", named.loc["Rotliegend", "top"], named.loc["Rotliegend", "base"], named.loc["Rotliegend", "k_ari"])]
    depth_min, depth_max = 3426.0, 3548.0

    with themed(theme) as c:
        fig = plt.figure(figsize=size((11, 6.6), c))
        compact = is_compact(c)
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.25, 2.1, 1.7],
            left=0.105 if compact else 0.07,
            right=0.97,
            top=top_margin(0.80, c, untitled=0.91),
            bottom=0.13 if compact else 0.15,
            wspace=0.08,
        )
        ax_zone, ax_k, ax_flow = (fig.add_subplot(grid[0, i]) for i in range(3))
        for ax in (ax_zone, ax_k, ax_flow):
            ax.set_ylim(depth_max, depth_min)
        ax_k.sharey(ax_zone)
        ax_flow.sharey(ax_zone)

        for name, top, base, k in zones:
            ax_zone.add_patch(Rectangle((0, top), 1, base - top, facecolor=zone_colour(name, c, tint=True), edgecolor=c["surface"], linewidth=2))
            label_colour = c["ink"] if name == tz.SUSPECT_NAME else c["secondary"]
            weight = "semibold" if name == tz.SUSPECT_NAME else "normal"
            ax_zone.text(0.08, (top + base) / 2, SHORT_NAMES[name], va="center", ha="left", fontsize=fs(9.5, c), color=label_colour, fontweight=weight)
            if not np.isnan(k):
                ax_k.add_patch(Rectangle((0.1, top + 0.35), k - 0.1, base - top - 0.7, facecolor=zone_colour(name, c, soft=True), edgecolor="none"))
            else:
                ax_k.text(0.13, (top + base) / 2, "no log data (seal)", va="center", fontsize=fs(8.5, c), color=c["muted"])
        ax_zone.set_xlim(0, 1)
        ax_zone.set_xticks([])
        ax_zone.spines["bottom"].set_visible(False)
        ax_zone.set_ylabel("Depth (m MD RKB)")
        ax_zone.set_title("Zones", loc="left", fontsize=fs(10, c), color=c["secondary"], pad=8)

        ax_k.set_xscale("log")
        ax_k.set_xlim(0.1, 5000)
        plain_log_ticks(ax_k.xaxis)
        ax_k.grid(axis="x")
        ax_k.set_axisbelow(True)
        ax_k.tick_params(axis="y", left=False, labelleft=False)
        ax_k.spines["left"].set_visible(False)
        ax_k.set_xlabel("Permeability, arithmetic average (mD)")
        ax_k.set_title("KLOGH", loc="left", fontsize=fs(10, c), color=c["secondary"], pad=8)
        suspect_mid = (column.loc[3, "top"] + column.loc[3, "base"]) / 2
        if compact:
            ax_k.text(1280 / 1.15, suspect_mid, "1,280 mD", va="center", ha="right", fontsize=fs(9.5, c), color=c["surface"], fontweight="semibold")
        else:
            ax_k.text(1280 * 1.15, suspect_mid, "1,280 mD", va="center", fontsize=fs(9.5, c), color=c["ink"], fontweight="semibold")

        for name, top, base, _ in zones:
            if name in profile.index:
                share = profile.loc[name, "flow_share"] * 100
                net = profile.loc[name, "net_share"] * 100
                ax_flow.add_patch(Rectangle((0, top + 0.35), share, base - top - 0.7, facecolor=zone_colour(name, c, soft=True), edgecolor="none"))
                tick = c.get("tick_on_suspect", c["ink"]) if name == tz.SUSPECT_NAME else c["ink"]
                ax_flow.plot([net, net], [top + 0.6, base - 0.6], color=tick, linewidth=2, solid_capstyle="butt")
        suspect = profile.loc[tz.SUSPECT_NAME]
        ax_flow.annotate(
            f"{suspect['flow_share']:.0%} of flow capacity\nfrom {suspect['net_share']:.0%} of net thickness",
            xy=(62, column.loc[3, "base"] - 0.4),
            xytext=(8, 3522),
            fontsize=fs(9.5, c),
            color=c["ink"],
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": c["muted"], "linewidth": 0.8},
        )
        ax_flow.set_xlim(0, 100)
        ax_flow.grid(axis="x")
        ax_flow.set_axisbelow(True)
        ax_flow.tick_params(axis="y", left=False, labelleft=False)
        ax_flow.spines["left"].set_visible(False)
        ax_flow.set_xlabel("Share of the Devonian (%)")
        ax_flow.set_title("Flow capacity, kh", loc="left", fontsize=fs(10, c), color=c["secondary"], pad=8)
        handles = [
            Patch(facecolor=c["devonian_soft"], label="Flow capacity share (bar)"),
            Line2D([0], [0], color=c["ink"], linewidth=2, label="Net thickness share (tick)"),
        ]
        ax_flow.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=fs(8.5, c), handlelength=1.2)

        add_title(
            fig,
            "One 9 m interval carries 90% of the Devonian's flow capacity",
            "Lismore field, single well. Zone averages of log-derived horizontal permeability (KLOGH).",
            c,
        )
        if not compact:
            fig.text(0.07, 0.022, "Basement zones (3543.7 to 3682 m, below 1 mD) are below the window shown.", fontsize=fs(8.5, c), color=c["muted"])
        save(fig, path)
    return fig


def rock_space(catalogue, suspect, neighbour_zones, theme: str = "light", path=None):
    """Winland porosity-permeability crossplot with the suspect zone's kNN neighbours."""
    label_offsets = {
        "Devonian 3.3 (12)": (-13, -3, "right"),
        "Devonian 3.2 (11)": (13, -3, "left"),
        "Devonian 3.1 (10)": (13, -3, "left"),
        "Lower Devonian": (9, -3, "left"),
        "Rotliegend": (9, -3, "left"),
        "Crystalline basement": (9, -3, "left"),
        "Top basement": (9, -3, "left"),
    }
    phi = np.linspace(0.10, 0.22, 200)
    with themed(theme) as c:
        fig, ax = plt.subplots(figsize=size((9.5, 6.4), c))
        compact = is_compact(c)
        fig.subplots_adjust(left=0.12 if compact else 0.10, right=0.86, top=top_margin(0.82, c, untitled=0.84), bottom=0.13 if compact else 0.11)
        for r35 in (0.5, 2.0, 10.0):
            ax.plot(phi * 100, tz.winland_k(r35, phi), color=c["axis"], linewidth=1)
            ax.text(22.15, tz.winland_k(r35, 0.22), f"r35 = {r35:g} µm", fontsize=fs(8.5, c), color=c["muted"], va="center")
        for name, (x, y) in {"megaport": (10.8, 900), "macroport": (10.8, 40), "mesoport": (10.8, 3.6), "microport": (10.8, 0.14)}.items():
            ax.text(x, y, name, fontsize=fs(9, c), color=c["muted"], style="italic")

        x_s, y_s = suspect["phif"].iloc[0] * 100, suspect["k_ari"].iloc[0]
        for zone in neighbour_zones:
            row = catalogue.set_index("zone").loc[zone]
            ax.scatter(row["phif"] * 100, row["k_ari"], s=260, facecolor="none", edgecolor=c["suspect"], linewidth=1.5, zorder=2)
        for _, row in catalogue.iterrows():
            ax.scatter(row["phif"] * 100, row["k_ari"], s=70, color=zone_colour(row["zone"], c), edgecolor=c["surface"], linewidth=2, zorder=3)
            dx, dy, ha = label_offsets[row["zone"]]
            ax.annotate(SHORT_NAMES[row["zone"]], (row["phif"] * 100, row["k_ari"]), xytext=(dx, dy), textcoords="offset points", fontsize=fs(9, c), color=c["secondary"], ha=ha)
        ax.scatter(x_s, y_s, s=150, marker="D", color=c["suspect"], edgecolor=c["surface"], linewidth=2, zorder=4)
        ax.annotate("Suspect zone", (x_s, y_s), xytext=(-12, 0), textcoords="offset points", fontsize=fs(9.5, c), fontweight="semibold", ha="right", va="center")

        ax.set_yscale("log")
        ax.set_xlim(10.5, 22)
        ax.set_ylim(0.1, 5000)
        plain_log_ticks(ax.yaxis)
        ax.set_xlabel("Porosity, PHIF (%)")
        ax.set_ylabel("Permeability, KLOGH arithmetic (mD)")
        ax.grid(True, which="major")
        ax.set_axisbelow(True)
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=8, markerfacecolor=c["devonian"], markeredgecolor="none", label="Devonian zone"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=8, markerfacecolor=c["other"], markeredgecolor="none", label="Other formation"),
            Line2D([0], [0], marker="D", linestyle="none", markersize=8, markerfacecolor=c["suspect"], markeredgecolor="none", label="Suspect zone (test point)"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=12, markerfacecolor="none", markeredgecolor=c["suspect"], markeredgewidth=1.5, label="Its three nearest neighbours"),
        ]
        if compact:
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=fs(8.5, c), borderaxespad=0)
        else:
            ax.legend(handles=handles, loc="lower right", fontsize=fs(8.5, c))
        add_title(
            fig,
            "Its three nearest neighbours are all Devonian zones",
            "Neighbours found using all six rock and fluid properties. Grey lines are Winland r35 pore-throat radii.",
            c,
            x=0.10,
        )
        save(fig, path)
    return fig


def property_fingerprint(catalogue, suspect, nearest, theme: str = "light", path=None):
    """The suspect zone against the training zones on each property, in standard deviations."""
    order = ["top", "base", "ng", "phif", "log_k_ari", "log_k_geo", "log_k_har", "sw"]
    nearest = nearest.set_index("property")
    with themed(theme) as c:
        fig, ax = plt.subplots(figsize=size((10, 5.8), c))
        compact = is_compact(c)
        fig.subplots_adjust(left=0.27 if compact else 0.22, right=0.66 if compact else 0.74, top=top_margin(0.80, c, untitled=0.90), bottom=0.12)
        for i, feature in enumerate(order):
            values = catalogue[feature]
            mean, sd = values.mean(), values.std(ddof=0)
            z = (values - mean) / sd
            ax.plot([z.min(), z.max()], [i, i], color=c["grid"], linewidth=6, solid_capstyle="round", zorder=1)
            for zone, value in zip(catalogue["zone"], z):
                ax.scatter(value, i, s=46, color=zone_colour(zone, c), edgecolor=c["surface"], linewidth=1.5, zorder=3)
            z_s = (suspect[feature].iloc[0] - mean) / sd
            ax.scatter(z_s, i, s=110, marker="D", color=c["suspect"], edgecolor=c["surface"], linewidth=2, zorder=4)
            row = nearest.loc[feature]
            name = SHORT_NAMES.get(row["nearest zone"], row["nearest zone"])
            if row["outside training range"]:
                direction = "higher" if z_s > z.max() else "lower"
                ax.text(3.35, i - 0.17, name, va="center", fontsize=fs(9, c), color=c["secondary"])
                ax.text(3.35, i + 0.21, f"suspect zone {direction} than all seven", va="center", fontsize=fs(7.5, c), color=c["muted"])
            else:
                ax.text(3.35, i, name, va="center", fontsize=fs(9, c), color=c["secondary"])
        ax.set_yticks(range(len(order)), [PROPERTY_LABELS[f] for f in order])
        ax.invert_yaxis()
        ax.set_xlim(-2.3, 3.2)
        ax.axvline(0, color=c["axis"], linewidth=0.8, zorder=0)
        ax.set_xlabel("Standard deviations from the mean of the seven training zones")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.text(3.35, -0.85, "Closest training zone", fontsize=fs(9, c), color=c["muted"])
        for y in (1.5, 6.5):
            ax.axhline(y, color=c["grid"], linewidth=0.8)
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=c["devonian"], markeredgecolor="none", label="Devonian zone"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=c["other"], markeredgecolor="none", label="Other formation"),
            Line2D([0], [0], marker="D", linestyle="none", markersize=7, markerfacecolor=c["suspect"], markeredgecolor="none", label="Suspect zone"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15 if compact else -0.12), ncol=3, fontsize=fs(8.5, c))
        fig.subplots_adjust(bottom=0.23 if compact else 0.2)
        add_title(
            fig,
            "Depth places it, rock quality ties it to the Upper Devonian, saturation to the Lower",
            "Permeability is on a log scale. Each row compares the suspect zone with the seven training zones on one property.",
            c,
        )
        save(fig, path)
    return fig


def lorenz_plot(profile, theme: str = "light", path=None):
    """Stratigraphic modified Lorenz plot of the Devonian column."""
    storage, flow = tz.smlp_curve(profile)
    with themed(theme) as c:
        fig, ax = plt.subplots(figsize=size((8.2, 6.4), c))
        fig.subplots_adjust(left=0.12, right=0.95, top=top_margin(0.80, c), bottom=0.11)
        ax.plot([0, 100], [0, 100], color=c["axis"], linewidth=1)
        ax.text(76, 71.5, "uniform column", rotation=38, fontsize=fs(8.5, c), color=c["muted"])
        for i, zone in enumerate(profile["zone"]):
            ax.plot(storage[i : i + 2] * 100, flow[i : i + 2] * 100, color=zone_colour(zone, c), linewidth=2.5, solid_capstyle="round")
        ax.scatter(storage * 100, flow * 100, s=36, color=c["ink"], edgecolor=c["surface"], linewidth=1.5, zorder=3)
        offsets = {
            tz.SUSPECT_NAME: (-12, 0, "right"),
            "Lower Devonian": (0, -16, "center"),
        }
        ax.text(21, 3.2, "Devonian 3.3, 3.2 and 3.1", fontsize=fs(9, c), color=c["secondary"], va="center")
        for i, zone in enumerate(profile["zone"]):
            if zone not in offsets:
                continue
            mid = ((storage[i] + storage[i + 1]) / 2 * 100, (flow[i] + flow[i + 1]) / 2 * 100)
            weight = "semibold" if zone == tz.SUSPECT_NAME else "normal"
            colour = c["ink"] if zone == tz.SUSPECT_NAME else c["secondary"]
            dx, dy, ha = offsets[zone]
            ax.annotate(SHORT_NAMES[zone], mid, xytext=(dx, dy), textcoords="offset points", fontsize=fs(9, c), color=colour, fontweight=weight, ha=ha, va="center")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel(r"Cumulative storage capacity, $\varphi h$ (%)")
        ax.set_ylabel(r"Cumulative flow capacity, $kh$ (%)")
        ax.grid(True)
        ax.set_axisbelow(True)
        suspect = profile.set_index("zone").loc[tz.SUSPECT_NAME]
        add_title(
            fig,
            "The suspect zone is the Devonian's speed zone",
            f"Zones in stratigraphic order. It holds {suspect['storage_share']:.0%} of the storage and {suspect['flow_share']:.0%} of the flow capacity.",
            c,
            x=0.12,
        )
        save(fig, path)
    return fig

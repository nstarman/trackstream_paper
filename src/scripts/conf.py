"""Configuration."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import astropy.units as u
import numpy as np
import paths
from astropy.coordinates import (
    ICRS,
    Angle,
    Galactocentric,
    SkyCoord,
)
from astropy.table import QTable, vstack
from astropy.visualization import quantity_support
from matplotlib.collections import EllipseCollection, PatchCollection, PathCollection
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerBase, HandlerPatch, HandlerPathCollection
from matplotlib.patches import Circle
from palettable.scientific.diverging import Vik_15
from trackstream import Stream
from trackstream.track.fit.utils import _v2c
from trackstream.track.utils import covariance_ellipse

if TYPE_CHECKING:
    from astropy.units import Quantity
    from matplotlib.axes import Axes
    from trackstream.stream import StreamArm


##############################################################################
# PARAMETERS

# Galpy stuff
RO = 8 * u.kpc
VO = 220 * u.km / u.s

# Physical types
LENGTH = u.get_physical_type("length")
SPEED = u.get_physical_type("speed")

# Plotting Configuration
quantity_support()  # automatic units labelling

cnorm = Normalize(vmin=-1, vmax=1)
cmap = Vik_15.mpl_colormap
color1 = cmap(0.25)
color2 = cmap(0.9)

# SOM stuff
SOM_KW = {
    "c": (0, 0, 0, 0.5),
    "ls": "--",
    "marker": "o",
    "markersize": 10,
    "markerfacecolor": "none",
    "markeredgecolor": (0, 0, 0, 0.75),
}

ARM_KW = {
    "s": 3,
    "cmap": cmap,
    "norm": cnorm,
    "marker": "*",
}

ARM1_KW: dict[str, Any] = {"color": color1, "marker": "*", "s": 4}
ARM2_KW: dict[str, Any] = {"color": color2, "marker": "*", "s": 4}


def scatter_handler(handle: HandlerBase, original: HandlerBase) -> None:
    """Update the scatter plot handler."""
    handle.update_from(original)
    handle.set_alpha(1)
    handle.set_sizes([40])


class HandlerOrigin(HandlerPatch):  # type: ignore[misc]
    """Handler for `.PatchCollection` instances."""

    def create_artists(  # noqa: PLR0913
        self: Any,
        legend: Any,  # noqa: ANN401, ARG002
        orig_handle: Any,  # noqa: ANN401, ARG002
        xdescent: Any,  # noqa: ANN401
        ydescent: Any,  # noqa: ANN401
        width: Any,  # noqa: ANN401
        height: Any,  # noqa: ANN401
        fontsize: Any,  # noqa: ANN401, ARG002
        trans: Any,  # noqa: ANN401, ARG002
    ) -> list[Circle]:
        """Create a list of artists for the legend."""
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        c1 = Circle(center, radius=6, facecolor="none", edgecolor="red", linewidth=3, zorder=-10)
        c2 = Circle(center, radius=1.5, facecolor="red", edgecolor="red", zorder=10)
        return [c1, c2]


handler_map = {
    PathCollection: HandlerPathCollection(update_func=scatter_handler),
    PatchCollection: HandlerOrigin(),
}

LABEL_KW = {"fontsize": 14}  # can remove when mpl3.7.1


##############################################################################
# CODE
##############################################################################


def get_from_vasiliev2019_table(name: str) -> SkyCoord:
    """Get coordinates from the Vasiliev 2019 table.

    Parameters
    ----------
    name : str
        Name of the globular cluster.

    Returns
    -------
    `astropy.coordinates.SkyCoord`
    """
    table = QTable.read(paths.data / "Vasiliev2019.ecsv")
    table.add_index("Name")
    gc = table.loc[name]  # Get coordinates from the table

    return SkyCoord(
        ra=gc["RAJ2000"],
        dec=gc["DEJ2000"],
        distance=gc["Dist"],
        pm_ra_cosdec=gc["pmRA"],
        pm_dec=gc["pmDE"],
        radial_velocity=gc["HRV"],
    )


#####################################################################


def get_ngc5466_stream(which: Literal["unperturbed", "perturbed"] = "unperturbed") -> Stream:
    """Get a Stream for NGC5466.

    Returns
    -------
    Stream
    """
    # -------------------------------------------
    # Read data file

    if which == "unperturbed":
        data = QTable.read(paths.data / "NGC5466_mw_model.ecsv")
    else:
        data = QTable.read(paths.data / "NGC5466_mw_and_dwarfgalaxy_model.ecsv")
        # Munge column names
        renames = {"vx": "v_x", "vy": "v_y", "vz": "v_z"}
        data.rename_columns(tuple(renames.keys()), tuple(renames.values()))

    # Turn data into SkyCoord
    sc = SkyCoord(Galactocentric(**{k: data[k] for k in ("x", "y", "z", "v_x", "v_y", "v_z")}))
    data["coords"] = sc.icrs

    # -------------------------------------------

    origin = SkyCoord(
        Galactocentric(
            x=4.67 * u.kpc,
            y=3.0 * u.kpc,
            z=15.3 * u.kpc,
            v_x=56 * u.km / u.s,
            v_y=-92 * u.km / u.s,
            v_z=51 * u.km / u.s,
        ),
    )

    # Assigning arms
    arm1_idx = sc.x <= origin.x
    arm2_idx = sc.x > origin.x
    data["arm"] = "none"
    data["arm"][arm1_idx] = "arm1"
    data["arm"][arm2_idx] = "arm2"

    data_err = QTable()
    data_err["arm"] = data["arm"]
    data_err["x_err"] = 0 * data["x"]  # (for the shape)
    data_err["y_err"] = 0 * u.kpc
    data_err["z_err"] = 0 * u.kpc
    data_err["lon_err"] = 0 * u.deg
    data_err["lat_err"] = 0 * u.deg

    return Stream.from_data(data, data_err=data_err, origin=origin, frame=ICRS(), name="NGC5466 in MW")


def get_pal5_stream(subsample: int = 100) -> Stream:
    """Get pal5 data from Ibata 2017.

    Parameters
    ----------
    subsample : int, optional
        Subsample the data by this factor, by default 100.

    Returns
    -------
    Stream
    """
    # Origin
    origin = SkyCoord(
        ra=Angle("15h 16m 05.3s"),
        dec=Angle("-00:06:41 degree"),
        distance=23 * u.kpc,
    )

    # Read data from Ibata et al. 2017
    data1 = QTable.read(paths.data / "IbataEtAl2017.ecsv")
    del data1["radial_velocity"]

    # Read data from Starkman et al. 2019
    data2 = QTable.read(paths.data / "StarkmanEtAl19.ecsv")
    data2 = data2[data2["ra"] < data1["ra"].min()]  # Only keep the new stuff
    data2 = data2[::subsample]  # Subsample.

    # Combine data
    data = vstack([data1, data2], join_type="inner")

    # Remove progenitor
    sel = np.abs(data["ra"] - 229 * u.deg) < 0.5 * u.deg
    data = data[~sel]

    data["arm"] = "arm1"
    data["arm"][data["ra"] < origin.ra] = "arm2"
    data = data.group_by("arm")

    data_err = QTable()
    data_err["lon_err"] = 0 * data["ra"]  # (for the shape)
    data_err["lat_err"] = 0 * u.deg
    data_err["arm"] = data["arm"]

    stream = Stream.from_data(data, origin=origin, data_err=data_err, frame=None, name="Palomar 5")
    stream.flags.set(minPmemb=50 * u.percent)

    # Don't mask anything
    stream.mask_outliers(threshold=100, verbose=True)

    return stream


###############################################################################


def plot_kalman(  # noqa: PLR0913
    ax: Axes,
    arm: StreamArm,
    kind: Literal["positions", "kinematics"],
    step: int = 5,
    label: str = "",
    zorder: int = 0,
) -> None:
    """Plot Kalman filter.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.
    arm : StreamArm
        The arm to plot.
    kind : {"positions", "kinematics}
        Whether to plot positions or kinematics.
    step : int, optional
        Plot every `step`th point, by default 5.
    label : str, optional
        Label for the plot, by default "".
    zorder: int, optional
        The zorder for the plot, by default 0.
    """
    track = arm.track
    path = track.path
    crd = _v2c(track.kalman, np.array(path._meta["smooth"].x[:, ::2]))  # noqa: SLF001

    if kind == "positions":
        components = tuple(crd.get_representation_component_names().keys())
        x, y = getattr(crd, components[0]), getattr(crd, components[1])
        covs = path._meta["smooth"].P[:, 0:4:2, 0:4:2]  # noqa: SLF001

        if not track.has_distances:
            covs = covs * u.rad.to(u.deg) ** 2
    else:
        components = tuple(crd.get_representation_component_names("s").keys())
        x, y = getattr(crd, components[0]), getattr(crd, components[1])
        covs = path._meta["smooth"].P[:, 4::2, 4::2]  # noqa: SLF001

    ax.plot(x, y, c="k", label=label)
    subsel = slice(None, None, step)
    mean = np.array((x, y)).reshape((2, -1)).T[subsel]
    angle, wh = covariance_ellipse(covs[subsel], nstd=1)
    ec = EllipseCollection(
        2 * np.atleast_1d(wh[..., 0]),
        2 * np.atleast_1d(wh[..., 1]),
        angle.to_value(u.deg),
        units="x",
        offsets=np.atleast_2d(mean),
        transOffset=ax.transData,
        facecolor="gray",
        edgecolor="none",
        alpha=0.5,
        lw=2,
        ls="solid",
        zorder=zorder,
    )
    ax.add_collection(ec)


def plot_origin(ax: Axes, x: Quantity, y: Quantity) -> None:
    """Plot origin."""
    ax.scatter(x, y, s=10, color="red", zorder=10)
    ax.scatter(x, y, s=400, facecolor="None", edgecolor="red", linewidth=3, zorder=-10)

    # circle for the origin
    c1 = Circle((x.value, y.value), radius=0, facecolor="none", linewidth=3, zorder=-10)
    c2 = Circle((x.value, y.value), radius=0, facecolor="none", zorder=10)
    ax.add_collection(PatchCollection([c1, c2], label="origin", match_original=True))


def fraction_format(x: float, _: float) -> str:
    """Format fraction."""
    f = Fraction(x)
    sign = "-" if np.sign(f.numerator) == -1 else ""

    if f.denominator == 1:
        return f"{sign}{np.abs(f.numerator)}"

    return sign + r"$\frac{" + f"{np.abs(f.numerator)}" + "}{" + f"{f.denominator}" + "}$"

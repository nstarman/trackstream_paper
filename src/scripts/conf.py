"""Configuration."""


from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import astropy.coordinates as coords
import astropy.units as u
import numpy as np
import paths
from astropy.table import QTable, vstack
from astropy.units import Quantity
from astropy.visualization import quantity_support
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from palettable.scientific.diverging import Vik_15
from trackstream import Stream
from trackstream.track.fit.utils import _v2c
from trackstream.track.utils import covariance_ellipse

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from matplotlib.axes import Axes
    from trackstream.stream import StreamArm


##############################################################################
# PARAMETERS

# Galpy stuff
RO = u.Quantity(8, u.kpc)
VO = u.Quantity(220, u.km / u.s)

# Physical types
LENGTH = u.get_physical_type("length")
SPEED = u.get_physical_type("speed")

# Plotting Configuration
quantity_support()  # automatic units labelling

cnorm = Normalize(vmin=-1, vmax=1)
cmap = Vik_15.mpl_colormap
color1 = cmap(0.25)
color2 = cmap(0.9)


# Globular Cluster Info : Marks and Kroupa '10
GC_47TUC_Mass = u.Quantity(7e5, u.solMass)


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

    return coords.SkyCoord(
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
    sc = coords.SkyCoord(coords.Galactocentric(**{k: data[k] for k in ("x", "y", "z", "v_x", "v_y", "v_z")}))
    data["coords"] = sc.icrs

    # -------------------------------------------

    origin = coords.SkyCoord(
        coords.Galactocentric(
            x=u.Quantity(4.67, u.kpc),
            y=u.Quantity(3.0, u.kpc),
            z=u.Quantity(15.3, u.kpc),
            v_x=u.Quantity(56, u.km / u.s),
            v_y=u.Quantity(-92, u.km / u.s),
            v_z=u.Quantity(51, u.km / u.s),
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
    data_err["y_err"] = u.Quantity(0, u.kpc)
    data_err["z_err"] = u.Quantity(0, u.kpc)
    data_err["lon_err"] = u.Quantity(0, u.deg)
    data_err["lat_err"] = u.Quantity(0, u.deg)

    return Stream.from_data(data, data_err=data_err, origin=origin, frame=coords.ICRS(), name="NGC5466 in MW")


def read_ngc5466_mw_gc_model() -> SkyCoord:
    """Read NGC5466 Milky Way Model simulation from data directory.

    Returns
    -------
    SkyCoord
    """
    data = QTable.read(paths.data / "NGC5466_mw_and_dwarfgalaxy_model.ecsv")
    return coords.SkyCoord.guess_from_table(data, representation_type="cartesian", frame="galactocentric")


def get_ngc5466_mw_gc_stream() -> Stream:
    """Get a Stream for NGC5466, fitting a track.

    Returns
    -------
    Stream
        With a fit track.
    """
    # Read data
    sc = read_ngc5466_mw_gc_model()

    origin = coords.SkyCoord(
        coords.Galactocentric(x=Quantity(4.67, u.kpc), y=Quantity(3.0, u.kpc), z=Quantity(15.3, u.kpc)),
    )
    arm1 = sc.x <= origin.x
    arm2 = sc.x > origin.x

    # make data
    data = QTable()
    data["coord"] = sc.icrs  # observer coordinates
    data["ra_err"] = 0
    data["dec_err"] = 0
    data["tail"] = "none"
    data["tail"][arm1] = "arm1"
    data["tail"][arm2] = "arm2"

    return Stream.from_data(data, origin=origin, name="NGC5466 in MW + dwarf galaxy")


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
    origin = coords.SkyCoord(
        ra=coords.Angle("15h 16m 05.3s"),
        dec=coords.Angle("-00:06:41 degree"),
        distance=u.Quantity(23, u.kpc),
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
    sel = np.abs(data["ra"] - u.Quantity(229, u.deg)) < u.Quantity(0.5, u.deg)
    data = data[~sel]

    data["arm"] = "arm1"
    data["arm"][data["ra"] < origin.ra] = "arm2"
    data = data.group_by("arm")

    data_err = QTable()
    data_err["lon_err"] = 0 * data["ra"]  # (for the shape)
    data_err["lat_err"] = u.Quantity(0, u.deg)
    data_err["arm"] = data["arm"]

    stream = Stream.from_data(data, origin=origin, data_err=data_err, frame=None, name="Palomar 5")
    stream.flags.set(minPmemb=u.Quantity(50, u.percent))

    # Don't mask anything
    stream.mask_outliers(threshold=100, verbose=True)

    return stream


###############################################################################


def plot_kalman(
    ax: Axes,
    arm: StreamArm,
    kind: Literal["positions", "kinematics"],
    step: int = 5,
    label: str = "",
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
    """
    track = arm.track
    path = track.path
    crd = _v2c(track.kalman, np.array(path._meta["smooth"].x[:, ::2]))  # noqa: SLF001

    if kind == "positions":
        components = tuple(crd.get_representation_component_names().keys())
        x, y = getattr(crd, components[0]), getattr(crd, components[1])
        covs = path._meta["smooth"].P[:, 0:4:2, 0:4:2]  # noqa: SLF001
    else:
        components = tuple(crd.get_representation_component_names("s").keys())
        x, y = getattr(crd, components[0]), getattr(crd, components[1])
        covs = path._meta["smooth"].P[:, 4::2, 4::2]  # noqa: SLF001

    ax.plot(x, y, zorder=100, c="k", label=label)
    subsel = slice(None, None, step)
    mean = np.array((x, y)).reshape((2, -1)).T[subsel]
    angle, wh = covariance_ellipse(covs[subsel], nstd=1)
    width = 2 * np.atleast_1d(wh[..., 0])
    height = 2 * np.atleast_1d(wh[..., 1])
    ec = EllipseCollection(
        width,
        height,
        angle.to_value(u.deg),
        units="x",
        offsets=np.atleast_2d(mean),
        transOffset=ax.transData,
        facecolor="gray",
        edgecolor="none",
        alpha=0.5,
        lw=2,
        ls="solid",
        zorder=0,
    )
    ax.add_collection(ec)

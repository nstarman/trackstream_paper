"""Plot frame-fit method."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib
from typing import cast

# THIRD-PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.units import Quantity
from trackstream.frame import fit_stream

# FIRST-PARTY
import paths
from conf import cmap, get_NGC5466_stream

##############################################################################
# CODE


def wrap_stream_lon_order(
    lon: Quantity, cut_at: Quantity = Quantity(100, u.deg), wrap_by: Quantity = Quantity(-360, u.deg)
) -> tuple[Quantity, np.ndarray]:
    """Wrap longitude data.

    Parameters
    ----------
    lon : Quantity
        The longitude.
    cut_at : Quantity, optional
        Angle at which to cut the longitude.
    wrap_by : Quantity, optional
        Wrap by.

    Returns
    -------
    lon : Quantity
    order : ndarray[int]
    """
    lt = np.where(lon < cut_at)[0]
    gt = np.where(lon > cut_at)[0]

    lon = cast("Quantity", np.concatenate((lon[gt] + wrap_by, lon[lt])))
    order = np.concatenate((gt, lt))

    return lon, order


##############################################################################
# SCRIPT
##############################################################################

stream = get_NGC5466_stream()
stream = fit_stream(stream, force=True, rot0=u.Quantity(110, u.deg))

# ===================================================================
# Plot

fig, axs = plt.subplots(3, 1, figsize=(6.7, 5.8), constrained_layout=True)

# ----
# Plot 1

stream.plot.in_frame(
    frame="icrs",
    c="tab:blue",
    label={"arm1": "", "arm2": "NGC5466 mock stream"},
    origin=True,
    ax=axs[0],
)
axs[0].legend(loc="lower left", fontsize=13)

# ----
# Plot 2 : Rotated Stream

ax = axs[1]
ax.axhline(0, c="gray", ls="--", zorder=0)

icrs = stream.coords.icrs  # coordinates
rsc = icrs.transform_to(stream.frame)
lon, sorter = wrap_stream_lon_order(rsc.lon, u.Quantity(100, u.deg), u.Quantity(-360, u.deg))

ax.scatter(lon, rsc.lat[sorter], s=3, marker="*", cmap=cmap, c="tab:blue", label=r"NGC-5466 mock stream")
stream.plot.origin(stream.origin, stream.frame, kind="positions", ax=axs[1])

ax.set_xlabel(r"$\phi_1$ (stream) [deg]", fontsize=13)
ax.set_ylabel(r"$\phi_2$ y (stream) [deg]", fontsize=13)
ax.set_aspect("equal")
ax.legend(loc="lower left", fontsize=13)

# ----
# Plot 3

stream.cache["frame_fit_result"].plot.residual(stream, ax=axs[2], format_ax=True, color="gray")
axs[2].legend(loc="lower left", fontsize=13)

# ----

fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

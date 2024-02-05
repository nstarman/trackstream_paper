"""Plot frame-fit method."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, cast

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.coordinates import CartesianRepresentation, UnitSphericalRepresentation
from astropy.units import Quantity
from conf import ARM1_KW, LABEL_KW, get_ngc5466_stream, handler_map, plot_origin
from matplotlib.ticker import FormatStrFormatter
from trackstream.frame import fit_stream
from trackstream.frame.fit import residual

if TYPE_CHECKING:
    from astropy.units import NDArray
    from numpy import floating

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")

##############################################################################
# CODE


def wrap_stream_lon_order(
    lon: Quantity,
    cut_at: Quantity = Quantity(100, u.deg),
    wrap_by: Quantity = Quantity(-360, u.deg),
) -> tuple[Quantity, NDArray[floating[Any]]]:
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

stream = get_ngc5466_stream()
stream = fit_stream(stream, force=True, rot0=110 * u.deg)

# ===================================================================
# Plot

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.7, 5.9))

frame = stream.frame
arm1c = stream["arm1"].coords
arm1c_icrs = arm1c.icrs
arm1c_icrs.ra.wrap_angle = 180 * u.deg

arm2c = stream["arm2"].coords
arm2c_icrs = arm2c.icrs
arm2c_icrs.ra.wrap_angle = 180 * u.deg

origin = stream["arm1"].origin.transform_to(frame)
origin_icrs = origin.transform_to("icrs")
origin_icrs.ra.wrap_angle = 180 * u.deg

# ----
# Plot 1

ax1.scatter(arm1c_icrs.ra, arm1c_icrs.dec, **ARM1_KW, label="NGC5466 mock stream")
plot_origin(ax1, origin_icrs.ra, origin_icrs.dec)
ax1.scatter(arm2c_icrs.ra, arm2c.icrs.dec, **ARM1_KW)

ax1.set_xlabel(r"RA (ICRS) [$\degree$]", **LABEL_KW)
ax1.set_ylabel(r"Dec (ICRS) [$\degree$]", **LABEL_KW)
lgnd = ax1.legend(loc="lower center", handler_map=handler_map)

# ----
# Plot 2 : Rotated Stream

ax2.axhline(0, c="gray", ls="--", zorder=0)

lon, sorter = wrap_stream_lon_order(arm1c.lon, 100 * u.deg, -360 * u.deg)
ax2.scatter(lon, arm1c.lat[sorter], **ARM1_KW, label=r"NGC-5466 mock stream")
plot_origin(ax2, origin.lon, origin.lat)
ax2.scatter(arm2c.lon, arm2c.lat, **ARM1_KW)

ax2.set_ylim(ax2.get_ylim()[0], 40)
ax2.set_xlabel(r"$\phi_1$ (stream) [$\degree$]", **LABEL_KW)
ax2.set_ylabel(r"$\phi_2$ (stream) [$\degree$]", **LABEL_KW)
ax2.set_aspect("equal")
lgnd = ax2.legend(loc="lower left", handler_map=handler_map)

# ----
# Plot 3

# Residual plot
fr = stream.cache["frame_fit_result"]
rotation_angles = np.linspace(-180, 180, num=1_000, dtype=float)
r = fr.origin.data
xyz = stream.data_coords.represent_as(UnitSphericalRepresentation).represent_as(CartesianRepresentation).xyz.value
res = np.array(
    [residual((float(angle), float(r.lon.deg), float(r.lat.deg)), xyz, scalar=True) for angle in rotation_angles],
)
ax3.plot(rotation_angles * u.deg, res, color="gray")

# Plot the best-fit rotation
ax3.axvline(
    fr.rotation.value,
    c="k",
    ls="--",
    label=rf"best-fit = {fr.rotation.value:.2f} $^\degree$",
)
# and the next period
next_period = 180 if (fr.rotation.value - 180) < rotation_angles.min() else -180
ax3.axvline(fr.rotation.value + next_period, c="k", ls="--", alpha=0.5)

ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax3.set_xlabel(r"Rotation angle $\theta$", **LABEL_KW)
ax3.set_ylabel(r"Scaled Residual", **LABEL_KW)
ax3.legend(loc="lower left", handler_map=handler_map)

# ----

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

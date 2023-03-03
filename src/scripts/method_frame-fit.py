"""Plot frame-fit method."""


from __future__ import annotations

import pathlib
from typing import cast

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.coordinates import CartesianRepresentation, UnitSphericalRepresentation
from astropy.units import Quantity
from conf import cmap, get_ngc5466_stream
from trackstream.frame import fit_stream
from trackstream.frame.fit import residual

##############################################################################
# CODE


def wrap_stream_lon_order(
    lon: Quantity,
    cut_at: Quantity = Quantity(100, u.deg),
    wrap_by: Quantity = Quantity(-360, u.deg),
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

stream = get_ngc5466_stream()
stream = fit_stream(stream, force=True, rot0=u.Quantity(110, u.deg))

# ===================================================================
# Plot

fig, axs = plt.subplots(3, 1, figsize=(6.7, 5.8), constrained_layout=True)

frame = stream.frame
arm1c = stream["arm1"].coords
arm2c = stream["arm2"].coords
origin = stream["arm1"].origin.transform_to(frame)

# ----
# Plot 1

# arm1
axs[0].scatter(
    arm1c.icrs.ra.wrap_at(180 * u.deg),
    arm1c.icrs.dec,
    s=3,
    marker="*",
    c="tab:blue",
    label="NGC5466 mock stream",
)
# origin
axs[0].scatter(origin.icrs.ra.wrap_at(180 * u.deg), origin.icrs.dec, s=10, color="red", label="origin")
axs[0].scatter(origin.icrs.ra.wrap_at(180 * u.deg), origin.icrs.dec, s=800, facecolor="None", edgecolor="red")
# arm2
axs[0].scatter(
    arm2c.icrs.ra.wrap_at(180 * u.deg),
    arm2c.icrs.dec,
    s=3,
    marker="*",
    c="tab:blue",
    label="NGC5466 mock stream",
)

axs[0].set_xlabel(r"RA (ICRS) [deg]", fontsize=13)
axs[0].set_ylabel(r"Dec (ICRS) [deg]", fontsize=13)
axs[0].legend(loc="lower right", fontsize=13)

# ----
# Plot 2 : Rotated Stream

ax = axs[1]
ax.axhline(0, c="gray", ls="--", zorder=0)

icrs = stream.coords.icrs  # coordinates
rsc = icrs.transform_to(stream.frame)
lon, sorter = wrap_stream_lon_order(rsc.lon, u.Quantity(100, u.deg), u.Quantity(-360, u.deg))

ax.scatter(lon, rsc.lat[sorter], s=3, marker="*", cmap=cmap, c="tab:blue", label=r"NGC-5466 mock stream")
# origin
origin = stream.origin.transform_to(stream.frame)
ax.scatter(origin.lon, origin.lat, s=10, color="red", label="origin")
ax.scatter(origin.lon, origin.lat, s=800, facecolor="None", edgecolor="red")

ax.set_xlabel(r"$\phi_1$ (stream) [deg]", fontsize=13)
ax.set_ylabel(r"$\phi_2$ y (stream) [deg]", fontsize=13)
ax.set_aspect("equal")
ax.legend(loc="lower left", fontsize=13)

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
axs[2].scatter(rotation_angles, res, color="gray")

# Plot the best-fit rotation
axs[2].axvline(
    fr.rotation.value,
    c="k",
    ls="--",
    label=rf"best-fit rotation = {fr.rotation.value:.2f} $^\degree$",
)
# and the next period
next_period = 180 if (fr.rotation.value - 180) < rotation_angles.min() else -180
axs[2].axvline(fr.rotation.value + next_period, c="k", ls="--", alpha=0.5)

axs[2].set_xlabel(r"Rotation angle $\theta$", fontsize=13)
axs[2].set_ylabel(r"Residual / # data pts", fontsize=10)
axs[2].legend(loc="lower left", fontsize=13)

# ----

fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

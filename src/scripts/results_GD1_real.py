"""Run TrackStream on GD-1 data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib

# THIRD-PARTY
import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from gala import coordinates as gc
from matplotlib.collections import EllipseCollection
from palettable.scientific.diverging import Roma_3

# FIRST-PARTY
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack, Times
from trackstream.track.fit.utils import _v2c
from trackstream.track.utils import covariance_ellipse
from trackstream.track.width import UnitSphericalWidth, Widths

# LOCAL
import paths
from conf import LENGTH

##############################################################################
# SCRIPTS
##############################################################################

# origin
origin = coords.SkyCoord(
    ra=u.Quantity(-135.01, u.deg),
    dec=u.Quantity(58, u.deg),
    pm_ra_cosdec=u.Quantity(-7.78, u.Unit("mas / yr")),
    pm_dec=u.Quantity(-7.85, u.Unit("mas / yr")),
)

# read data
full_data = QTable.read(paths.data / "PWB18_thinsel.ecsv")
full_data = full_data[full_data["pm_mask"] & full_data["gi_cmd_mask"] & full_data["stream_track_mask"]]
data = full_data[full_data["thin_stream_sel"]]

dif = coords.UnitSphericalDifferential(d_lon=data["pm_phi1_cosphi2_no_reflex"], d_lat=data["pm_phi2_no_reflex"])
rep = coords.UnitSphericalRepresentation(lon=data["phi1"], lat=data["phi2"], differentials=dif)
data["coords"] = coords.SkyCoord(gc.GD1Koposov10(rep))
data["arm"] = "arm1"
data = data.group_by("arm")

data_err = QTable()
data_err["ra_err"] = 0 * data["ra"]  # (for the shape)
data_err["dec_err"] = u.Quantity(0, u.deg)
data_err["arm"] = data["arm"]

# stream
stream = Stream.from_data(data, data_err=data_err, origin=origin, frame=gc.GD1Koposov10(), name="GD-1")


stream_width0 = Widths.from_format({"length": UnitSphericalWidth(lat=u.Quantity(1, u.deg), lon=u.Quantity(1, u.deg))})

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 8, "sigma": 0.1, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)

_ = stream.fit_track(
    fitters=fitters,
    force=True,
    tune=True,
    composite=True,
    som_kw={"num_iteration": int(5e4), "progress": True},
    kalman_kw={"dtmax": Times({LENGTH: u.Quantity(0.1, u.deg)})},
)


# ===================================================================
# Plot

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axs.shape = (-1, 1)
full_name = stream["arm1"].full_name or ""

# Plot stream in system frame
axs[0, 0].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")

arm1c = stream["arm1"].coords
frame = arm1c.frame
axs[0, 0].scatter(arm1c.phi1, arm1c.phi2, s=1, c="tab:blue", label=full_name, marker="*")

# origin
origin = stream["arm1"].origin.transform_to(frame)
axs[0, 0].scatter(origin.phi1, origin.phi2, s=10, color="red", label="origin")
axs[0, 0].scatter(origin.phi1, origin.phi2, s=800, facecolor="None", edgecolor="red")

# -------------------------------------------------------------
# SOM

axs[1, 0].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")

# origin
axs[1, 0].scatter(origin.phi1, origin.phi2, s=10, color="red", label="origin")
axs[1, 0].scatter(origin.phi1, origin.phi2, s=800, facecolor="None", edgecolor="red")
# data
axs[1, 0].scatter(
    arm1c.phi1, arm1c.phi2, s=1, c=np.arange(len(arm1c)), label=full_name, marker="*", cmap=Roma_3.mpl_colormap
)
# som
som1 = stream["arm1"].track.som
ps1 = som1.prototypes.transform_to(frame)
axs[1, 0].plot(ps1.phi1, ps1.phi2, c="k")
axs[1, 0].scatter(ps1.phi1, ps1.phi2, marker="P", edgecolors="black", facecolor="none")


# -------------------------------------------------------------
# Kalman filter plot

axs[2, 0].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")

# origin
axs[2, 0].scatter(origin.phi1, origin.phi2, s=10, color="red", label="origin")
axs[2, 0].scatter(origin.phi1, origin.phi2, s=800, facecolor="None", edgecolor="red")
# data
axs[2, 0].scatter(
    arm1c.phi1, arm1c.phi2, s=1, c=np.arange(len(arm1c)), label=full_name, marker="*", cmap=Roma_3.mpl_colormap
)
# kalman
track = stream["arm1"].track
path = track.path
crd = _v2c(track.kalman, np.array(path._meta["smooth"].x[:, ::2]))
Ps = path._meta["smooth"].P[:, 0:4:2, 0:4:2]

x, y = crd.phi1, crd.phi2
axs[2, 0].plot(x.value, y.value, label=f"track {track.name}", zorder=100, c="k")
subsel = slice(None, None, 5)
mean = np.array((x, y)).reshape((2, -1)).T[subsel]
angle, wh = covariance_ellipse(Ps[subsel], nstd=1)
width = 2 * np.atleast_1d(wh[..., 0])
height = 2 * np.atleast_1d(wh[..., 1])
ec = EllipseCollection(
    width,
    height,
    angle.to_value(u.deg),
    units="x",
    offsets=np.atleast_2d(mean),
    transOffset=axs[2, 0].transData,
    facecolor="gray",
    edgecolor="none",
    alpha=0.5,
    lw=2,
    ls="solid",
    zorder=0,
)
axs[2, 0].add_collection(ec)


# -------------------------------------------------------------

for ax in fig.axes:
    ax.set_xlabel(r"$\phi_1$ (GD1) [deg]", fontsize=15)
    ax.set_ylabel(r"$\phi_2$ (GD1) [deg]", fontsize=15)
    ax.legend(fontsize=12)

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

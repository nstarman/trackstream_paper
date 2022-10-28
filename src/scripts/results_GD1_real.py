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
from astropy.table import QTable
from gala import coordinates as gc
from palettable.scientific.diverging import Roma_3
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack, Times
from trackstream.track.width import UnitSphericalWidth, Widths

# FIRST-PARTY
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

fig, axes = plt.subplots(3, 1, figsize=(8, 8))
axes.shape = (-1, 1)
full_name = stream["arm1"].full_name or ""

# Plot stream in system frame
axes[0, 0].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")

in_frame_kw = {"c": "tab:blue", "label": full_name, "format_ax": True, "origin": True}
stream["arm1"].plot.in_frame(frame="stream", kind="positions", ax=axes[0, 0], **in_frame_kw)

# -------------------------------------------------------------
# SOM

axes[1, 0].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")
stream["arm1"].track.plot(
    ax=axes[1, 0],
    frame="stream",
    kind="positions",
    format_ax=True,
    origin=True,
    som=True,
    som_kw=None,
    kalman=False,
    cmap=Roma_3.mpl_colormap,
)

# -------------------------------------------------------------
# Kalman filter plot

axes[2, 0].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")
kalman_kw = {"std": 1, "alpha": 0.5, "c": "k"}
stream["arm1"].track.plot(
    ax=axes[2, 0],
    frame="stream",
    kind="positions",
    origin=True,
    format_ax=True,
    som=False,
    kalman=True,
    kalman_kw=kalman_kw,
    cmap=Roma_3.mpl_colormap,
)

for ax in fig.axes:
    ax.set_xlabel(r"$\phi_1$ (GD1) [deg]", fontsize=15)
    ax.set_ylabel(r"$\phi_2$ (GD1) [deg]", fontsize=15)
    ax.legend(fontsize=12)

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

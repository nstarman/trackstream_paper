"""Show the Kalman filter method."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib
import warnings

# THIRD-PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# FIRST-PARTY
from trackstream.frame import fit_stream as fit_frame_to_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import AngularWidth, UnitSphericalWidth, Widths

# LOCAL
import paths
from conf import LENGTH, cmap, cnorm, get_NGC5466_stream, plot_kalman

##############################################################################
# SCRIPT
##############################################################################

# Get stream
stream = get_NGC5466_stream()
# Fit frame
stream = fit_frame_to_stream(stream, force=True, rot0=u.Quantity(110, u.deg))

# Mask the wrapped data for a prettier plot
stream["arm1"].data["order"].mask[stream["arm1"].coords.lon > u.Quantity(100, u.deg)] = True

# Stream widths
stream_width0 = Widths({LENGTH: UnitSphericalWidth(lon=u.Quantity(2, u.deg), lat=u.Quantity(2, u.deg))})
fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 15, "sigma": 2, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths({LENGTH: AngularWidth(u.Quantity(4, u.deg))})
dtmax = Times({LENGTH: u.Quantity(2, u.deg)})

# Fit stream
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    track = stream.fit_track(
        fitters=fitters,
        som_kw={"num_iteration": int(1e3), "progress": False},
        kalman_kw={"dtmax": dtmax, "width_min": width_min},
        composite=True,
        force=True,
    )

# ===================================================================
# Plot

fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.grid(visible=True)

frame = stream.frame
arm1c = stream["arm1"].coords
arm2c = stream["arm2"].coords
origin = stream["arm1"].origin.transform_to(frame)

# arm 1
ax.scatter(
    arm1c.lon.wrap_at("180d"),
    arm1c.lat,
    label="arm 1",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c.lon)),
    s=3,
)
# origin
ax.scatter(origin.lon.wrap_at("180d"), origin.lat, s=10, color="red", label="origin")
ax.scatter(origin.lon.wrap_at("180d"), origin.lat, s=800, facecolor="None", edgecolor="red")
# arm 2
ax.scatter(
    arm2c.lon.wrap_at("180d"),
    arm2c.lat,
    label="arm 2",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c.lon)),
    s=3,
)
# kalman
plot_kalman(ax, stream["arm1"], "positions", step=1, label="track arm 1")
plot_kalman(ax, stream["arm2"], "positions", step=1, label="track arm 2")

ax.legend()
fig.tight_layout()

ax.set_xlim(None, 50)
ax.set_xlabel(r"Longitude (Stream) [deg]")
ax.set_ylabel(r"Latitude (Stream) [deg]")

# ----

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

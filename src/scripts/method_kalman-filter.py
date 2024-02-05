"""Show the Kalman filter method."""

from __future__ import annotations

import pathlib
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from conf import ARM_KW, LABEL_KW, LENGTH, color1, color2, get_ngc5466_stream, handler_map, plot_kalman, plot_origin
from trackstream.frame import fit_stream as fit_frame_to_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import AngularWidth, UnitSphericalWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
LABEL_KW = LABEL_KW | {"fontsize": 17}

##############################################################################
# SCRIPT
##############################################################################

# Get stream
stream = get_ngc5466_stream()
# Fit frame
stream = fit_frame_to_stream(stream, force=True, rot0=110 * u.deg)

# Mask the wrapped data for a prettier plot
stream["arm1"].data["order"].mask[stream["arm1"].coords.lon > 100 * u.deg] = True

# Stream widths
stream_width0 = Widths({LENGTH: UnitSphericalWidth(lon=2 * u.deg, lat=2 * u.deg)})
fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 15, "sigma": 2, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths({LENGTH: AngularWidth(4 * u.deg)})
dtmax = Times({LENGTH: 2 * u.deg})

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

fig = plt.figure(figsize=(8, 3.5))
ax = fig.add_subplot(111)
ax.grid(visible=True, zorder=-10_000, alpha=0.5)
ax.set_axisbelow(b=True)

frame = stream.frame
arm1c = stream["arm1"].coords
arm1c.lon.wrap_angle = 180 * u.deg

arm2c = stream["arm2"].coords
origin = stream["arm1"].origin.transform_to(frame)

# arm 1
ax.scatter(arm1c.lon, arm1c.lat, label="arm 1", c=np.linspace(0, -1, len(arm1c.lon)), **ARM_KW)
# origin
plot_origin(ax, origin.lon, origin.lat)
# arm 2
ax.scatter(arm2c.lon.wrap_at("180d"), arm2c.lat, label="arm 2", c=np.linspace(0, 1, len(arm2c.lon)), **ARM_KW)
# kalman
plot_kalman(ax, stream["arm1"], "positions", step=1, label="arm tracks")
plot_kalman(ax, stream["arm2"], "positions", step=1)

ax.set_xlim(None, 50)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.set_xlabel(r"Longitude (Stream) [$\degree$]", **LABEL_KW)
ax.set_ylabel(r"Latitude (Stream) [$\degree$]", **LABEL_KW)

lgnd = ax.legend(handler_map=handler_map, fontsize=13)
lgnd.legend_handles[0].set_color(color1)
lgnd.legend_handles[2].set_color(color2)


# ----

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

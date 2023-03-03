"""Run TrackStream on a streamspray simulation."""

#############################################################################
# IMPORTS

from __future__ import annotations

import pathlib

import asdf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.coordinates import Galactocentric
from conf import LENGTH, SPEED, cmap, cnorm, color1, color2, plot_kalman
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack, Times
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

##############################################################################
# SCRIPT
##############################################################################

with asdf.open(paths.data / "streamspraydf.asdf") as af:
    origin = af["origin"]

    data = af["data"]
    data = data.group_by("arm")
    data.add_index("arm")

frame = Galactocentric()

# Make stream
stream = Stream.from_data(data, origin=origin, frame=frame, name="47 Tuc")
stream.mask_outliers(threshold=1.5)

# ------------
# Fit stream

stream_width0 = Widths(
    {
        LENGTH: Cartesian3DWidth(x=u.Quantity(100, u.pc), y=u.Quantity(100, u.pc), z=u.Quantity(100, u.pc)),
        SPEED: Cartesian3DiffWidth(
            d_x=u.Quantity(10, u.km / u.s),
            d_y=u.Quantity(10, u.km / u.s),
            d_z=u.Quantity(10, u.km / u.s),
        ),
    },
)

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=True,
    som_kw={"nlattice": 20, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": u.Quantity(15, u.kpc)}},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths(
    {LENGTH: Cartesian1DWidth(u.Quantity(300, u.pc)), SPEED: Cartesian1DiffWidth(u.Quantity(4, u.km / u.s))},
)
dtmax = Times({LENGTH: u.Quantity(2, u.kpc), SPEED: u.Quantity(1, u.km / u.s)})
track = stream.fit_track(
    fitters=fitters,
    force=True,
    tune=True,
    som_kw={"num_iteration": int(2e4), "progress": True},
    kalman_kw={"dtmax": dtmax, "width_min": width_min},
    composite=True,
)


# ===================================================================
# Plot

# Plot and save
#     },
#         },
#         },
#     },

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
frame = stream.frame
arm1c = stream["arm1"].coords
arm2c = stream["arm2"].coords
origin = stream["arm1"].origin.transform_to(frame)

# -------------------------------------------------------------
# Data

# data
axs[0, 0].scatter(arm1c.x, arm1c.y, s=1, color=color1, label="arm 1", marker="*")
axs[0, 0].scatter(arm2c.x, arm2c.y, s=1, color=color2, label="arm 2", marker="*")
# origin
axs[0, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[0, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")

# data
axs[0, 1].scatter(arm1c.v_x, arm1c.v_y, s=1, color=color1, label="arm 1", marker="*")
axs[0, 1].scatter(arm2c.v_x, arm2c.v_y, s=1, color=color2, label="arm 2", marker="*")
# origin
axs[0, 1].scatter(origin.v_x, origin.v_y, s=10, color="red", label="origin")
axs[0, 1].scatter(origin.v_x, origin.v_y, s=800, facecolor="None", edgecolor="red")


# -------------------------------------------------------------
# SOM

ps1 = stream["arm1"].track.som.prototypes.transform_to(frame)
ps2 = stream["arm2"].track.som.prototypes.transform_to(frame)

# arm1
axs[1, 0].scatter(
    arm1c.x,
    arm1c.y,
    s=1,
    label="arm 1",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
)
# origin
axs[1, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[1, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[1, 0].scatter(
    arm2c.x,
    arm2c.y,
    s=1,
    label="arm 2",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
)
# som
axs[1, 0].plot(ps1.x, ps1.y, c="k")
axs[1, 0].scatter(ps1.x, ps1.y, marker="P", edgecolors="black", facecolor="none")
axs[1, 0].plot(ps2.x, ps2.y, c="k")
axs[1, 0].scatter(ps2.x, ps2.y, marker="P", edgecolors="black", facecolor="none")

# arm1
axs[1, 1].scatter(
    arm1c.v_x,
    arm1c.v_y,
    s=1,
    label="arm 1",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
)
# origin
axs[1, 1].scatter(origin.v_x, origin.v_y, s=10, color="red", label="origin")
axs[1, 1].scatter(origin.v_x, origin.v_y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[1, 1].scatter(
    arm2c.v_x,
    arm2c.v_y,
    s=1,
    label="arm 2",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
)
# som
axs[1, 1].plot(ps1.v_x, ps1.v_y, c="k")
axs[1, 1].scatter(ps1.v_x, ps1.v_y, marker="P", edgecolors="black", facecolor="none")
axs[1, 1].plot(ps2.v_x, ps2.v_y, c="k")
axs[1, 1].scatter(ps2.v_x, ps2.v_y, marker="P", edgecolors="black", facecolor="none")

# -------------------------------------------------------------
# Kalman

# arm1
axs[2, 0].scatter(
    arm1c.x,
    arm1c.y,
    s=1,
    label="arm 1",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
)
# origin
axs[2, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[2, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[2, 0].scatter(
    arm2c.x,
    arm2c.y,
    s=1,
    label="arm 2",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
)
# kalman
plot_kalman(axs[2, 0], stream["arm1"], kind="positions", step=1)
plot_kalman(axs[2, 0], stream["arm2"], kind="positions", step=1)

# arm1
axs[2, 1].scatter(
    arm1c.v_x,
    arm1c.v_y,
    s=1,
    label="arm 1",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
)
# origin
axs[2, 1].scatter(origin.v_x, origin.v_y, s=10, color="red", label="origin")
axs[2, 1].scatter(origin.v_x, origin.v_y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[2, 1].scatter(
    arm2c.v_x,
    arm2c.v_y,
    s=1,
    label="arm 2",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
)
# kalman
plot_kalman(axs[2, 1], stream["arm1"], kind="kinematics", step=1)
plot_kalman(axs[2, 1], stream["arm2"], kind="kinematics", step=1)

# -------------------------------------------------------------

for ax in fig.axes:
    ax.legend(loc="upper left", fontsize=13)
fig.savefig(
    str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")),
    bbox_inches="tight",
)

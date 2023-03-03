"""Run TrackStream on a streamdf mock stream."""


from __future__ import annotations

import pathlib

import asdf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from conf import LENGTH, SPEED, cmap, cnorm, color1, color2, get_from_vasiliev2019_table, plot_kalman
from trackstream import Stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

##############################################################################
# SCRIPT
##############################################################################

prog_sc = get_from_vasiliev2019_table("NGC 104")

# GET DATA
with asdf.open(paths.data / "streamdf.asdf") as af:
    true_sct = af["trailing"]
    true_scl = af["leading"]

    data = af["samples"]
    data = data.group_by("arm")
    data.add_index("arm")

# Working in Galactocentric frame. Galpy has different definition,
# so let's use that one.
prog_sc = prog_sc.transform_to(true_sct.frame)
frame = prog_sc.replicate_without_data()

stream = Stream.from_data(data, origin=prog_sc, name="Comp DF", frame=frame)
stream.mask_outliers(threshold=1.2)

# Define and run the fitters
stream_width0 = Widths.from_format(
    {
        "length": Cartesian3DWidth(x=u.Quantity(100, u.pc), y=u.Quantity(100, u.pc), z=u.Quantity(100, u.pc)),
        "speed": Cartesian3DiffWidth(
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
    som_kw={"nlattice": 15, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": u.Quantity(20, u.kpc)}},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths.from_format(
    {"length": Cartesian1DWidth(u.Quantity(300, u.pc)), "speed": Cartesian1DiffWidth(u.Quantity(4, u.km / u.s))},
)
track = stream.fit_track(
    fitters=fitters,
    force=True,
    tune=True,
    som_kw={"num_iteration": int(1e4), "progress": True},
    kalman_kw={
        "dtmax": Times({LENGTH: u.Quantity(50, unit=u.pc), SPEED: u.Quantity(5, u.km / u.s)}),
        "width_min": width_min,
    },
)

# ===================================================================
# Plot

fig, axs = plt.subplots(3, 2, figsize=(16, 12))

frame = stream.frame
arm1c = stream["arm1"].coords
arm2c = stream["arm2"].coords
origin = stream["arm1"].origin.transform_to(frame)

# -------------------------------------------------------------
# Data

# data
axs[0, 0].scatter(arm1c.x, arm1c.y, s=1, color=color1, label="arm 1", marker="*")
# origin
axs[0, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[0, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[0, 0].scatter(arm2c.x, arm2c.y, s=1, color=color2, label="arm 2", marker="*")

# arm1
axs[0, 1].scatter(arm1c.v_x, arm1c.v_y, s=1, color=color1, label="arm 1", marker="*")
# origin
axs[0, 1].scatter(origin.v_x, origin.v_y, s=10, color="red", label="origin")
axs[0, 1].scatter(origin.v_x, origin.v_y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[0, 1].scatter(arm2c.v_x, arm2c.v_y, s=1, color=color2, label="arm 2", marker="*")


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

# data
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
plot_kalman(axs[2, 0], stream["arm1"], kind="positions")
plot_kalman(axs[2, 0], stream["arm2"], kind="positions")

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
plot_kalman(axs[2, 1], stream["arm1"], kind="kinematics")
plot_kalman(axs[2, 1], stream["arm2"], kind="kinematics")

# -------------------------------------------------------------

# overlay true track
axs[2, 0].plot(true_sct.x, true_sct.y, label="DF track", c="k", ls="--", zorder=0)
axs[2, 0].plot(true_scl.x, true_scl.y, c="k", ls="--", zorder=0)
axs[2, 1].plot(true_sct.v_x, true_sct.v_y, label="DF track", c="k", ls="--", zorder=0)
axs[2, 1].plot(true_scl.v_x, true_scl.v_y, c="k", ls="--", zorder=0)

# -------------------------------------------------------------------
# Add kinematics cutaway plots

xi, yi = 0.4, 0.27
dx, dy = 0.425, 0.55
x1, x2, y1, y2 = -4, 0, -7.25, -5

axins = axs[0, 0].inset_axes([xi, yi, dx, dy])
axins.scatter(arm1c.x, arm1c.y, color=color1, s=10, label="arm 1", marker="*")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[0, 0].indicate_inset_zoom(axins, edgecolor="black")

axins = axs[1, 0].inset_axes([xi, yi, dx, dy])
axins.scatter(arm1c.x, arm1c.y, s=1, label="arm 1", marker="*", cmap=cmap, norm=cnorm, c=np.linspace(0, -1, len(arm1c)))
axins.plot(ps1.x, ps1.y, c="k")
axins.scatter(ps1.x, ps1.y, marker="P", edgecolors="black", facecolor="none")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[1, 0].indicate_inset_zoom(axins, edgecolor="black")

xi = 0.3
axins = axs[2, 0].inset_axes([xi, yi, dx, dy])
plot_kalman(axins, stream["arm1"], kind="positions")
axins.scatter(arm1c.x, arm1c.y, s=1, label="arm 1", marker="*", cmap=cmap, norm=cnorm, c=np.linspace(0, -1, len(arm1c)))
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[2, 0].indicate_inset_zoom(axins, edgecolor="black")

# -------------------------------------------------------------------
# Add kinematic cutaway plots

xi, yi = 0.2, 0.3
dx, dy = 0.5, 0.4
x1, x2, y1, y2 = 25, 120, 150, 190

axins = axs[0, 1].inset_axes([xi, yi, dx, dy])
axins.scatter(arm2c.v_x, arm2c.v_y, color=color2, s=10, label="arm 2", marker="*")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[0, 1].indicate_inset_zoom(axins, edgecolor="black")

axins = axs[1, 1].inset_axes([xi, yi, dx, dy])
axins.scatter(
    arm2c.v_x,
    arm2c.v_y,
    s=1,
    label="arm 1",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm2c)),
)
axins.plot(ps2.v_x, ps2.v_y, c="k")
axins.scatter(ps2.v_x, ps2.v_y, marker="P", edgecolors="black", facecolor="none")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[1, 1].indicate_inset_zoom(axins, edgecolor="black")

axins = axs[2, 1].inset_axes([xi, yi, dx, dy])
plot_kalman(axins, stream["arm2"], kind="kinematics")
axins.scatter(
    arm2c.v_x,
    arm2c.v_y,
    s=1,
    label="arm 1",
    marker="*",
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm2c)),
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[2, 1].indicate_inset_zoom(axins, edgecolor="black")

# -------------------------------------------------------------------

# remake the legends
axs[0, 0].legend(loc="lower right", fontsize=12)
axs[0, 1].legend(loc="lower right", fontsize=12)
axs[1, 0].legend(loc="lower right", fontsize=12)
axs[1, 1].legend(loc="lower right", fontsize=12)
axs[2, 0].legend(loc="lower right", fontsize=12)
axs[2, 1].legend(loc="lower right", fontsize=12)


fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

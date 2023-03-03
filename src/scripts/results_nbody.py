"""Run TrackStream on an N-Body."""


from __future__ import annotations

import pathlib

import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import QTable
from conf import LENGTH, SPEED, cmap, cnorm, color1, color2, plot_kalman
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

##############################################################################
# PARAMETERS

# Origin
origin = SkyCoord(
    Galactocentric(
        x=u.Quantity(-8.25, u.kpc),
        y=u.Quantity(0, u.kpc),
        z=u.Quantity(16, u.kpc),
        v_x=u.Quantity(45, u.km / u.s),
        v_y=u.Quantity(-110, u.km / u.s),
        v_z=u.Quantity(-15, u.km / u.s),
    ),
)

##############################################################################
# SCRIPT
##############################################################################

# Read data table
data = QTable.read(paths.data / "nbody_pal5.ecsv")
# remove progenitor and unwanted
data = data[data["arm"] != "none"]
data = data[data["arm"] != "prog"]
# label tails
data["arm"][data["arm"] == "lead"] = "arm1"
data["arm"][data["arm"] == "trail"] = "arm2"
# group by arm
data = data.group_by("arm")
data.add_index("arm")
# make coord, for Stream
data["coords"] = SkyCoord(Galactocentric(**{k: data[k] for k in ("x", "y", "z", "v_x", "v_y", "v_z")}))

data_err = QTable()
data_err["x_err"] = 0 * data["x"]  # (for the shape)
data_err["y_err"] = u.Quantity(0, u.kpc)
data_err["z_err"] = u.Quantity(0, u.kpc)
data_err["arm"] = data["arm"]

# make stream, with pre-defined frame
stream = Stream.from_data(
    data[::10],
    data_err=data_err[::10],
    origin=origin,
    frame=coords.Galactocentric(),
    name="Pal5 N-body",
)
stream.mask_outliers(threshold=1.1)

stream_width0 = Widths(
    {
        LENGTH: Cartesian3DWidth(x=u.Quantity(200, u.pc), y=u.Quantity(200, u.pc), z=u.Quantity(200, u.pc)),
        SPEED: Cartesian3DiffWidth(
            d_x=u.Quantity(5, u.km / u.s),
            d_y=u.Quantity(5, u.km / u.s),
            d_z=u.Quantity(5, u.km / u.s),
        ),
    },
)
fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=True,
    som_kw={"nlattice": 20, "sigma": 2, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)

# Fit track
width_min = Widths.from_format(
    {"length": Cartesian1DWidth(u.Quantity(300, u.pc)), "speed": Cartesian1DiffWidth(u.Quantity(4, u.km / u.s))},
)
dtmax = u.Quantity((40, 4), dtype=[("length", float), ("speed", float)], unit=(u.pc, u.km / u.s))
_ = stream.fit_track(
    fitters=fitters,
    som_kw={"num_iteration": int(1e4)},
    kalman_kw={"dtmax": dtmax, "width_min": width_min},
    force=True,
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
axs[0, 0].scatter(arm2c.x, arm2c.y, s=1, color=color2, label="arm 2", marker="*")
# origin
axs[0, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[0, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")

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
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
    label="arm 1",
    marker="*",
)
# origin
axs[1, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[1, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[1, 0].scatter(
    arm2c.x,
    arm2c.y,
    s=1,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
    label="arm 2",
    marker="*",
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
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
    label="arm 1",
    marker="*",
)
# origin
axs[1, 1].scatter(origin.v_x, origin.v_y, s=10, color="red", label="origin")
axs[1, 1].scatter(origin.v_x, origin.v_y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[1, 1].scatter(
    arm2c.v_x,
    arm2c.v_y,
    s=1,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
    label="arm 2",
    marker="*",
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
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
    label="arm 1",
    marker="*",
)
# origin
axs[2, 0].scatter(origin.x, origin.y, s=10, color="red", label="origin")
axs[2, 0].scatter(origin.x, origin.y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[2, 0].scatter(
    arm2c.x,
    arm2c.y,
    s=1,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
    label="arm 2",
    marker="*",
)
# kalman
for arm in ("arm1", "arm2"):
    plot_kalman(axs[2, 0], stream[arm], kind="positions")

# arm1
axs[2, 1].scatter(
    arm1c.v_x,
    arm1c.v_y,
    s=1,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(arm1c)),
    label="arm 1",
    marker="*",
)
# origin
axs[2, 1].scatter(origin.v_x, origin.v_y, s=10, color="red", label="origin")
axs[2, 1].scatter(origin.v_x, origin.v_y, s=800, facecolor="None", edgecolor="red")
# arm2
axs[2, 1].scatter(
    arm2c.v_x,
    arm2c.v_y,
    s=1,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, 1, len(arm2c)),
    label="arm 2",
    marker="*",
)
# kalman
plot_kalman(axs[2, 1], stream["arm1"], kind="kinematics")
plot_kalman(axs[2, 1], stream["arm2"], kind="kinematics")


# -------------------------------------------------------------------


for ax in axs[:, 0].flat:
    ax.set_xlabel(f"x (Galactocentric) [{ax.get_xlabel()}]", fontsize=16)
    ax.set_ylabel(f"y (Galactocentric) [{ax.get_ylabel()}]", fontsize=16)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_rasterization_zorder(10000)

for ax in axs[:, 1].flat:
    ax.set_xlabel(f"$v_x$ (Galactocentric) [{ax.get_xlabel()}]", fontsize=16)
    ax.set_ylabel(f"$v_y$ (Galactocentric) [{ax.get_ylabel()}]", fontsize=16)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_rasterization_zorder(10000)

fig.tight_layout()

fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

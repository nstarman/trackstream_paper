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
from conf import (
    ARM1_KW,
    ARM2_KW,
    ARM_KW,
    LABEL_KW,
    LENGTH,
    SOM_KW,
    SPEED,
    color1,
    color2,
    handler_map,
    plot_kalman,
    plot_origin,
)
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
LABEL_KW = LABEL_KW | {"fontsize": 19}

##############################################################################
# PARAMETERS

# Origin
origin = SkyCoord(
    Galactocentric(
        x=-8.25 * u.kpc,
        y=0 * u.kpc,
        z=16 * u.kpc,
        v_x=45 * u.km / u.s,
        v_y=-110 * u.km / u.s,
        v_z=-15 * u.km / u.s,
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
data_err["y_err"] = 0 * u.kpc
data_err["z_err"] = 0 * u.kpc
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
        LENGTH: Cartesian3DWidth(x=200 * u.pc, y=200 * u.pc, z=200 * u.pc),
        SPEED: Cartesian3DiffWidth(d_x=5 * u.km / u.s, d_y=5 * u.km / u.s, d_z=5 * u.km / u.s),
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
    {"length": Cartesian1DWidth(300 * u.pc), "speed": Cartesian1DiffWidth(4 * u.km / u.s)},
)
dtmax = u.Quantity((40, 4), dtype=[("length", float), ("speed", float)], unit=(u.pc, u.km / u.s))
_ = stream.fit_track(
    fitters=fitters,
    som_kw={"num_iteration": int(1e4)},
    kalman_kw={"dtmax": dtmax, "width_min": width_min},
    force=True,
)


##############################################################################
# Plot

fig, axs = plt.subplots(3, 2, figsize=(16, 12))

frame = stream.frame
arm1c = stream["arm1"].coords
arm2c = stream["arm2"].coords
origin = stream["arm1"].origin.transform_to(frame)

# ============================================================================
# Data

axs[0, 0].scatter(arm1c.x, arm1c.y, label="arm 1", **ARM1_KW)
plot_origin(axs[0, 0], origin.x, origin.y)
axs[0, 0].scatter(arm2c.x, arm2c.y, label="arm 2", **ARM2_KW)

axs[0, 1].scatter(arm1c.v_x, arm1c.v_y, label="arm 1", **ARM1_KW)
plot_origin(axs[0, 1], origin.v_x, origin.v_y)
axs[0, 1].scatter(arm2c.v_x, arm2c.v_y, label="arm 2", **ARM2_KW)


# ============================================================================
# SOM

ps1 = stream["arm1"].track.som.prototypes.transform_to(frame)
ps2 = stream["arm2"].track.som.prototypes.transform_to(frame)

# arms
axs[1, 0].scatter(arm1c.x, arm1c.y, c=np.linspace(0, -1, len(arm1c)), label="arm 1", **ARM_KW)
plot_origin(axs[1, 0], origin.x, origin.y)
axs[1, 0].scatter(arm2c.x, arm2c.y, c=np.linspace(0, 1, len(arm2c)), label="arm 2", **ARM_KW)
# som
axs[1, 0].plot(ps1.x, ps1.y, **SOM_KW, markeredgewidth=2, label="SOM")
axs[1, 0].plot(ps2.x, ps2.y, **SOM_KW, markeredgewidth=2)


# arms
axs[1, 1].scatter(arm1c.v_x, arm1c.v_y, c=np.linspace(0, -1, len(arm1c)), label="arm 1", **ARM_KW)
plot_origin(axs[1, 1], origin.v_x, origin.v_y)
axs[1, 1].scatter(arm2c.v_x, arm2c.v_y, c=np.linspace(0, 1, len(arm2c)), label="arm 2", **ARM_KW)
# som
axs[1, 1].plot(ps1.v_x, ps1.v_y, **SOM_KW, markeredgewidth=2, label="SOM")
axs[1, 1].plot(ps2.v_x, ps2.v_y, **SOM_KW, markeredgewidth=2)


# ============================================================================
# Kalman

# arm1
axs[2, 0].scatter(arm1c.x, arm1c.y, c=np.linspace(0, -1, len(arm1c)), label="arm 1", **ARM_KW)
# origin
plot_origin(axs[2, 0], origin.x, origin.y)
# arm2
axs[2, 0].scatter(arm2c.x, arm2c.y, c=np.linspace(0, 1, len(arm2c)), label="arm 2", **ARM_KW)
# kalman
plot_kalman(axs[2, 0], stream["arm1"], kind="positions", label="track")
plot_kalman(axs[2, 0], stream["arm2"], kind="positions")


# arm1
axs[2, 1].scatter(arm1c.v_x, arm1c.v_y, c=np.linspace(0, -1, len(arm1c)), label="arm 1", **ARM_KW)
# origin
plot_origin(axs[2, 1], origin.v_x, origin.v_y)
# arm2
axs[2, 1].scatter(arm2c.v_x, arm2c.v_y, c=np.linspace(0, 1, len(arm2c)), label="arm 2", **ARM_KW)
# kalman
plot_kalman(axs[2, 1], stream["arm1"], kind="kinematics", label="track")
plot_kalman(axs[2, 1], stream["arm2"], kind="kinematics")

# ============================================================================

for i, ax in enumerate(axs[:, 0]):
    if i != 2:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    ax.set_ylabel(rf"$y$ (GXYC) [{ax.get_ylabel()}]", **LABEL_KW)

axs[-1, 0].set_xlabel(rf"$x$ (GXYC) [{axs[2, 0].get_xlabel()}]", **LABEL_KW)

for i, ax in enumerate(axs[:, 1]):
    if i < 2:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    ax.set_ylabel(rf"$v_y$ (GXYC) [{ax.get_ylabel()}]", **LABEL_KW)

axs[-1, 1].set_xlabel(rf"$v_x$ (GXYC) [{axs[2, 1].get_xlabel()}]", **LABEL_KW)


for ax in axs.flat:
    lgnd = ax.legend(loc="lower left", handler_map=handler_map)

    for handle in lgnd.legend_handles:
        if handle.get_label() == "arm 1":
            handle.set_color(color1)
            handle.set_facecolor(color1)
        elif handle.get_label() == "arm 2":
            handle.set_color(color2)
            handle.set_facecolor(color2)

    ax.set_rasterization_zorder(10000)

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

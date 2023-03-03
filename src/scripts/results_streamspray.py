"""Run TrackStream on a streamspray simulation."""

from __future__ import annotations

import pathlib

import asdf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.coordinates import Galactocentric
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
from trackstream.track.fit import FitterStreamArmTrack, Times
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
LABEL_KW = LABEL_KW | {"fontsize": 19}

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
        LENGTH: Cartesian3DWidth(x=100 * u.pc, y=100 * u.pc, z=100 * u.pc),
        SPEED: Cartesian3DiffWidth(d_x=10 * u.km / u.s, d_y=10 * u.km / u.s, d_z=10 * u.km / u.s),
    },
)

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=True,
    som_kw={"nlattice": 20, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": 15 * u.kpc}},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths(
    {LENGTH: Cartesian1DWidth(300 * u.pc), SPEED: Cartesian1DiffWidth(4 * u.km / u.s)},
)
dtmax = Times({LENGTH: 2 * u.kpc, SPEED: 1 * u.km / u.s})
track = stream.fit_track(
    fitters=fitters,
    force=True,
    tune=True,
    som_kw={"num_iteration": int(2e4), "progress": True},
    kalman_kw={"dtmax": dtmax, "width_min": width_min},
    composite=True,
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

# arm1
axs[1, 0].scatter(arm1c.x, arm1c.y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
# origin
plot_origin(axs[1, 0], origin.x, origin.y)
# arm2
axs[1, 0].scatter(arm2c.x, arm2c.y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# som
axs[1, 0].plot(ps1.x, ps1.y, **SOM_KW, label="SOM")
axs[1, 0].plot(ps2.x, ps2.y, **SOM_KW)

# arm1
axs[1, 1].scatter(arm1c.v_x, arm1c.v_y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
# origin
plot_origin(axs[1, 1], origin.v_x, origin.v_y)
# arm2
axs[1, 1].scatter(arm2c.v_x, arm2c.v_y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# som
axs[1, 1].plot(ps1.v_x, ps1.v_y, **SOM_KW, label="SOM")
axs[1, 1].plot(ps2.v_x, ps2.v_y, **SOM_KW)


# ============================================================================
# Kalman

# arm1
axs[2, 0].scatter(arm1c.x, arm1c.y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
# origin
plot_origin(axs[2, 0], origin.x, origin.y)
# arm2
axs[2, 0].scatter(arm2c.x, arm2c.y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# kalman
plot_kalman(axs[2, 0], stream["arm1"], kind="positions", step=1, label="track")
plot_kalman(axs[2, 0], stream["arm2"], kind="positions", step=1)


# arm1
axs[2, 1].scatter(arm1c.v_x, arm1c.v_y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
# origin
plot_origin(axs[2, 1], origin.v_x, origin.v_y)
# arm2
axs[2, 1].scatter(arm2c.v_x, arm2c.v_y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# kalman
plot_kalman(axs[2, 1], stream["arm1"], kind="kinematics", step=1, label="track")
plot_kalman(axs[2, 1], stream["arm2"], kind="kinematics", step=1)


# ============================================================================

for i, ax in enumerate(axs[:, 0]):
    if i < 2:
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
    lgnd = ax.legend(loc="upper left", handler_map=handler_map)

    for handle in lgnd.legend_handles:
        if handle.get_label() == "arm 1":
            handle.set_color(color1)
            handle.set_facecolor(color1)
        elif handle.get_label() == "arm 2":
            handle.set_color(color2)
            handle.set_facecolor(color2)

# -------------------------------------------------------------

fig.tight_layout()
fig.savefig(
    str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")),
    bbox_inches="tight",
)

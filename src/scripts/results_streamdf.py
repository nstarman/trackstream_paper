"""Run TrackStream on a streamdf mock stream."""

from __future__ import annotations

import pathlib

import asdf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
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
    get_from_vasiliev2019_table,
    handler_map,
    plot_kalman,
    plot_origin,
)
from trackstream import Stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
LABEL_KW = LABEL_KW | {"fontsize": 19}

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
        "length": Cartesian3DWidth(x=100 * u.pc, y=100 * u.pc, z=100 * u.pc),
        "speed": Cartesian3DiffWidth(d_x=10 * u.km / u.s, d_y=10 * u.km / u.s, d_z=10 * u.km / u.s),
    },
)

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=True,
    som_kw={"nlattice": 15, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": 20 * u.kpc}},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths.from_format(
    {"length": Cartesian1DWidth(300 * u.pc), "speed": Cartesian1DiffWidth(4 * u.km / u.s)},
)
track = stream.fit_track(
    fitters=fitters,
    force=True,
    tune=True,
    som_kw={"num_iteration": int(1e4), "progress": True},
    kalman_kw={
        "dtmax": Times({LENGTH: 50 * u.pc, SPEED: 5 * u.km / u.s}),
        "width_min": width_min,
    },
)

##############################################################################
# Plot


def setup_axin(ax: plt.Axes, axins: plt.Axes, xlims: tuple[float, float], ylims: tuple[float, float]) -> None:
    """Inset axes setup."""
    axins.set_xlim(*xlims)
    axins.set_ylim(*ylims)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xlabel("")
    axins.set_ylabel("")
    # inside the axes, plot the zoomed region
    ax.indicate_inset_zoom(axins, edgecolor="black")


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
axs[1, 0].scatter(arm1c.x, arm1c.y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
plot_origin(axs[1, 0], origin.x, origin.y)
axs[1, 0].scatter(arm2c.x, arm2c.y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# som
axs[1, 0].plot(ps1.x, ps1.y, **SOM_KW, label="SOM")
axs[1, 0].plot(ps2.x, ps2.y, **SOM_KW)


# arms
axs[1, 1].scatter(arm1c.v_x, arm1c.v_y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
plot_origin(axs[1, 1], origin.v_x, origin.v_y)
axs[1, 1].scatter(arm2c.v_x, arm2c.v_y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# som
axs[1, 1].plot(ps1.v_x, ps1.v_y, **SOM_KW, label="SOM")
axs[1, 1].plot(ps2.v_x, ps2.v_y, **SOM_KW)


# ============================================================================
# Kalman

# arms
axs[2, 0].scatter(arm1c.x, arm1c.y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
plot_origin(axs[2, 0], origin.x, origin.y)
axs[2, 0].scatter(arm2c.x, arm2c.y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# kalman
plot_kalman(axs[2, 0], stream["arm1"], kind="positions", label="track")
plot_kalman(axs[2, 0], stream["arm2"], kind="positions")


# arms
axs[2, 1].scatter(arm1c.v_x, arm1c.v_y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
plot_origin(axs[2, 1], origin.v_x, origin.v_y)
axs[2, 1].scatter(arm2c.v_x, arm2c.v_y, label="arm 2", c=np.linspace(0, 1, len(arm2c)), **ARM_KW)
# kalman
plot_kalman(axs[2, 1], stream["arm1"], kind="kinematics", label="track")
plot_kalman(axs[2, 1], stream["arm2"], kind="kinematics")


# -------------------------------------------------------------

# overlay true track
axs[2, 0].plot(true_sct.x, true_sct.y, label="DF track", c="k", ls="--", zorder=0)
axs[2, 0].plot(true_scl.x, true_scl.y, c="k", ls="--", zorder=0)
axs[2, 1].plot(true_sct.v_x, true_sct.v_y, label="DF track", c="k", ls="--", zorder=0)
axs[2, 1].plot(true_scl.v_x, true_scl.v_y, c="k", ls="--", zorder=0)


# ============================================================================
# Add position cutaway plots

xlims, ylims = (-4, 0), (-7.25, -5)

axins = axs[0, 0].inset_axes([0.4, 0.27, 0.425, 0.55])
axins.scatter(arm1c.x, arm1c.y, color=color1, s=10, label="arm 1", marker="*")
setup_axin(axs[0, 0], axins, xlims, ylims)

axins = axs[1, 0].inset_axes([0.4, 0.27, 0.425, 0.55])
axins.scatter(arm1c.x, arm1c.y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW)
axins.plot(ps1.x, ps1.y, **SOM_KW)
setup_axin(axs[1, 0], axins, xlims, ylims)

axins = axs[2, 0].inset_axes([0.4, 0.27, 0.425, 0.55])
plot_kalman(axins, stream["arm1"], kind="positions")
axins.scatter(arm1c.x, arm1c.y, label="arm 1", c=np.linspace(0, -1, len(arm1c)), **ARM_KW | {"s": 10})
setup_axin(axs[2, 0], axins, xlims, ylims)

# -------------------------------------------------------------------
# Add kinematic cutaway plots

xlims, ylims = (25, 120), (150, 190)

axins = axs[0, 1].inset_axes([0.2, 0.3, 0.5, 0.4])
axins.scatter(arm2c.v_x, arm2c.v_y, color=color2, s=10, label="arm 2", marker="*")
setup_axin(axs[0, 1], axins, xlims, ylims)

axins = axs[1, 1].inset_axes([0.2, 0.3, 0.5, 0.4])
axins.scatter(arm2c.v_x, arm2c.v_y, label="arm 1", c=np.linspace(0, 1, len(arm2c)), **ARM_KW | {"s": 10})
axins.plot(ps2.v_x, ps2.v_y, **SOM_KW)
setup_axin(axs[1, 1], axins, xlims, ylims)

axins = axs[2, 1].inset_axes([0.2, 0.3, 0.5, 0.4])
plot_kalman(axins, stream["arm2"], kind="kinematics")
axins.scatter(arm2c.v_x, arm2c.v_y, label="arm 1", c=np.linspace(0, 1, len(arm2c)), **ARM_KW | {"s": 10})
setup_axin(axs[2, 1], axins, xlims, ylims)

# -------------------------------------------------------------------

for i, ax in enumerate(axs[:, 0]):
    lgnd = ax.legend(loc="upper right", handler_map=handler_map)
    ax.set_ylabel(rf"$y$ (GXYC) [{ax.get_ylabel()}]", **LABEL_KW)
    if i < 2:
        ax.set_xlabel("")
        ax.set_xticklabels([])

axs[-1, 0].set_xlabel(rf"$x$ (GXYC) [{axs[2, 0].get_xlabel()}]", **LABEL_KW)

for i, ax in enumerate(axs[:, 1]):
    lgnd = ax.legend(loc="lower right", handler_map=handler_map)
    ax.set_ylabel(rf"$v_y$ (GXYC) [{ax.get_ylabel()}]", **LABEL_KW)
    if i < 2:
        ax.set_xlabel("")
        ax.set_xticklabels([])

axs[-1, 1].set_xlabel(rf"$v_x$ (GXYC) [{axs[2, 1].get_xlabel()}]", **LABEL_KW)

# remake the legends
for ax in axs.flat:
    lgnd = ax.get_legend()

    for handle in lgnd.legend_handles:
        if handle.get_label() == "arm 1":
            handle.set_color(color1)
            handle.set_facecolor(color1)
        elif handle.get_label() == "arm 2":
            handle.set_color(color2)
            handle.set_facecolor(color2)

# -------------------------------------------------------------

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

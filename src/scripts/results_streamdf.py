"""Run TrackStream on a streamdf mock stream."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib

# THIRD-PARTY
import asdf
import astropy.units as u
import numpy as np
from trackstream import Stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

# FIRST-PARTY
import paths
from conf import LENGTH, SPEED, cmap, cnorm, color1, color2, get_from_Vasiliev2019_table

##############################################################################
# SCRIPT
##############################################################################

prog_sc = get_from_Vasiliev2019_table("NGC 104")

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
            d_x=u.Quantity(10, u.km / u.s), d_y=u.Quantity(10, u.km / u.s), d_z=u.Quantity(10, u.km / u.s)
        ),
    }
)

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=True,
    som_kw={"nlattice": 15, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": u.Quantity(20, u.kpc)}},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths.from_format(
    {"length": Cartesian1DWidth(u.Quantity(300, u.pc)), "speed": Cartesian1DiffWidth(u.Quantity(4, u.km / u.s))}
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

fig, axs = stream.track.plot.full_multipanel(
    in_frame_kw={"arm1": {"color": color1, "marker": "*", "s": 10}, "arm2": {"color": color2, "marker": "*", "s": 10}},
    som_kw={
        "arm1": {"in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, -1, len(stream["arm1"].coords))}},
        "arm2": {"in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, 1, len(stream["arm2"].coords))}},
    },
    kalman_kw={
        "arm1": {
            "connect": True,
            "subselect": 1,
            "in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, -1, len(stream["arm1"].coords))},
        },
        "arm2": {
            "connect": True,
            "subselect": 1,
            "in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, 1, len(stream["arm2"].coords))},
        },
    },
)

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
stream["arm1"].plot.in_frame(frame="stream", kind="positions", ax=axins, color=color1, s=10)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[0, 0].indicate_inset_zoom(axins, edgecolor="black")

axins = axs[1, 0].inset_axes([xi, yi, dx, dy])
stream["arm1"].track.plot(
    ax=axins,
    frame="stream",
    kind="positions",
    format_ax=False,
    origin=False,
    som=True,
    kalman=False,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(stream["arm1"].coords)),
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[1, 0].indicate_inset_zoom(axins, edgecolor="black")

xi = 0.3
axins = axs[2, 0].inset_axes([xi, yi, dx, dy])
stream["arm1"].track.plot(
    ax=axins,
    frame="stream",
    kind="positions",
    format_ax=False,
    origin=False,
    som=False,
    kalman=True,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(stream["arm1"].coords)),
)
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
stream["arm2"].plot.in_frame(frame="stream", kind="kinematics", ax=axins, color=color1, s=10)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[0, 1].indicate_inset_zoom(axins, edgecolor="black")

axins = axs[1, 1].inset_axes([xi, yi, dx, dy])
stream["arm2"].track.plot(
    ax=axins,
    frame="stream",
    kind="kinematics",
    format_ax=False,
    origin=False,
    som=True,
    kalman=False,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(stream["arm2"].coords)),
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel("")
axins.set_ylabel("")
axs[1, 1].indicate_inset_zoom(axins, edgecolor="black")

axins = axs[2, 1].inset_axes([xi, yi, dx, dy])
stream["arm2"].track.plot(
    ax=axins,
    frame="stream",
    kind="kinematics",
    format_ax=False,
    origin=False,
    som=False,
    kalman=True,
    cmap=cmap,
    norm=cnorm,
    c=np.linspace(0, -1, len(stream["arm2"].coords)),
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

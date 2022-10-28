"""Run TrackStream on a streamspray simulation."""

#############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib

# THIRD-PARTY
import asdf
import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack, Times
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

# FIRST-PARTY
import paths
from conf import LENGTH, SPEED, cmap, cnorm, color1, color2

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
            d_x=u.Quantity(10, u.km / u.s), d_y=u.Quantity(10, u.km / u.s), d_z=u.Quantity(10, u.km / u.s)
        ),
    }
)

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=True,
    som_kw={"nlattice": 20, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": u.Quantity(15, u.kpc)}},
    kalman_kw={"width0": stream_width0},
)

width_min = Widths(
    {LENGTH: Cartesian1DWidth(u.Quantity(300, u.pc)), SPEED: Cartesian1DiffWidth(u.Quantity(4, u.km / u.s))}
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
fig, _ = track.plot.full_multipanel(
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
for ax in fig.axes:
    ax.legend(loc="upper left", fontsize=13)
fig.savefig(
    str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")),
    bbox_inches="tight",
)

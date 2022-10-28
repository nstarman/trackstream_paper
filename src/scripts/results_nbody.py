"""Run TrackStream on an N-Body."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib

# THIRD-PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import QTable
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack
from trackstream.track.width import Cartesian1DiffWidth, Cartesian1DWidth, Cartesian3DiffWidth, Cartesian3DWidth, Widths

# FIRST-PARTY
import paths
from conf import LENGTH, SPEED, cmap, cnorm, color1, color2

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
    )
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
            d_x=u.Quantity(5, u.km / u.s), d_y=u.Quantity(5, u.km / u.s), d_z=u.Quantity(5, u.km / u.s)
        ),
    }
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
    {"length": Cartesian1DWidth(u.Quantity(300, u.pc)), "speed": Cartesian1DiffWidth(u.Quantity(4, u.km / u.s))}
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

fig, axs = stream.track.plot.full_multipanel(
    in_frame_kw={"arm1": {"color": color1}, "arm2": {"color": color2}},
    som_kw={
        "arm1": {"in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, -1, len(stream["arm1"].coords))}},
        "arm2": {"in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, 1, len(stream["arm2"].coords))}},
    },
    kalman_kw={
        "arm1": {
            "connect": True,
            "in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, -1, len(stream["arm1"].coords))},
        },
        "arm2": {
            "connect": True,
            "in_frame_kw": {"cmap": cmap, "norm": cnorm, "c": np.linspace(0, 1, len(stream["arm2"].coords))},
        },
    },
)

for ax in axs.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_rasterization_zorder(10000)

fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

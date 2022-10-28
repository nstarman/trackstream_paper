"""Plot results of running TrackStream on Palomar 5."""

##############################################################################
# IMPORTS

# STDLIB
import pathlib

# THIRD-PARTY
import astropy.units as u
import matplotlib.pyplot as plt

# FIRST PARTY
from trackstream.frame import fit_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import UnitSphericalWidth, Widths

# FIRST-PARTY
import paths
from conf import LENGTH, color1, color2, get_pal5_stream

##############################################################################
# SCRIPT
##############################################################################

# Get stream
stream = get_pal5_stream()
# Fit frame
stream = fit_stream(stream, force=True, rot0=u.Quantity(150, u.deg))

stream_width0 = Widths.from_format({"length": UnitSphericalWidth(lon=u.Quantity(2, u.deg), lat=u.Quantity(5, u.deg))})

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 10, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": u.Quantity(20, u.deg)}},
    kalman_kw={"width0": stream_width0},
)

_ = stream.fit_track(
    fitters=fitters,
    tune=True,
    som_kw={"num_iteration": int(1e4), "progress": True},
    kalman_kw={"dtmax": Times({LENGTH: u.Quantity(1.3, u.deg)})},
)


# ===================================================================
# Plot

fig, axs = plt.subplots(4, 1, figsize=(8, 12))
axs.shape = (-1, 1)

stream["arm1"].plot.in_frame(frame="icrs", kind="positions", ax=axs[0, 0], format_ax=False, origin=True, color=color1)
stream["arm2"].plot.in_frame(frame="icrs", kind="positions", ax=axs[0, 0], format_ax=True, color=color2)

# ===================================

stream["arm1"].plot.in_frame(frame="stream", kind="positions", ax=axs[1, 0], format_ax=False, origin=True, color=color1)
stream["arm2"].plot.in_frame(frame="stream", kind="positions", ax=axs[1, 0], format_ax=True, color=color2)

# ===================================
# SOM plot

stream["arm1"].track.plot(
    ax=axs[2, 0],
    frame="stream",
    kind="positions",
    format_ax=False,
    origin=False,
    som=True,
    som_kw=None,
    kalman=False,
    color=color1,
)
stream["arm2"].track.plot(
    ax=axs[2, 0],
    frame="stream",
    kind="positions",
    format_ax=True,
    origin=True,
    som=True,
    som_kw=None,
    kalman=False,
    color=color2,
)

# ===================================
# Kalman Filter

kalman_kw = {"std": 1, "alpha": 0.5}
stream["arm1"].track.plot(
    ax=axs[3, 0],
    frame="stream",
    kind="positions",
    origin=False,
    format_ax=False,
    som=False,
    kalman=True,
    kalman_kw=kalman_kw,
    color=color1,
)
stream["arm2"].track.plot(
    ax=axs[3, 0],
    frame="stream",
    kind="positions",
    origin=True,
    format_ax=True,
    som=False,
    kalman=True,
    kalman_kw=kalman_kw,
    color=color2,
)

for ax in fig.axes:
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.legend(fontsize=15)

axs[-1, 0].legend(fontsize=15, ncol=2)

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

"""Plot example of the time proxy."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import pathlib
import warnings

# THIRD-PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d

# FIRST-PARTY
from trackstream.frame import fit_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.fit.timesteps import make_timesteps
from trackstream.track.width import UnitSphericalWidth, Widths

# LOCAL
import paths
from conf import LENGTH, cmap, cnorm, get_NGC5466_stream

##############################################################################
# SCRIPT
##############################################################################

# Get stream
stream = get_NGC5466_stream()
# Fit frame
stream = fit_stream(stream, force=True, rot0=u.Quantity(110, u.deg))

# Fit track. This is overkill for just showing the SOM, but fitting a full
# track also re-orders the stream and makes plotting much easier.
stream_width0 = Widths({LENGTH: UnitSphericalWidth(lon=u.Quantity(2, u.deg), lat=u.Quantity(2, u.deg))})

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 25, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": u.Quantity(30, u.deg)}},
    kalman_kw={"width0": stream_width0},
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    _ = stream.fit_track(
        fitters=fitters,
        som_kw={"num_iteration": int(1e3), "progress": False},
        kalman_kw={"dtmax": Times({LENGTH: u.Quantity(3, u.deg)})},
        composite=True,
        force=True,
    )


# ===================================================================
# Plot

thin = 20

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()

# ---------------------------------
# Arm 1

dts1 = make_timesteps(stream["arm1"].coords, stream["arm1"].track.kalman, dt0=Times({LENGTH: u.Quantity(0, u.deg)}),)[
    "length"
].to_value(u.deg)
ds1 = stream["arm1"].coords[1:].separation(stream["arm1"].coords[:-1]).to_value(u.deg)
arm1_visit_order = np.arange(len(ds1))

ax.scatter(-arm1_visit_order, ds1, alpha=0.1, c=np.linspace(0, -1, len(ds1)), norm=cnorm, cmap=cmap, label="arm 1")

ax.plot(
    -arm1_visit_order,
    interp1d(-arm1_visit_order[::thin], dts1[::thin], kind="quadratic", fill_value="extrapolate")(-arm1_visit_order),
    c="k",
    alpha=0.5,
)
ax.scatter(
    -arm1_visit_order[::thin],
    dts1[::thin],
    edgecolor="black",
    facecolors="none",
    s=40,
    label="arm1 'time' steps (subsampled)",
    zorder=100,
    marker="<",
)

# ---------------------------------
# Arm 2

dts2 = make_timesteps(stream["arm2"].coords, stream["arm2"].track.kalman, dt0=Times({LENGTH: u.Quantity(0, u.deg)}),)[
    "length"
].to_value(u.deg)
ds2 = stream["arm2"].coords[1:].separation(stream["arm2"].coords[:-1]).to_value(u.deg)
arm2_visit_order = np.arange(len(ds2))

ax.scatter(arm2_visit_order, ds2, alpha=0.1, c=np.linspace(0, 1, len(ds2)), norm=cnorm, cmap=cmap, label="arm 2")

ax.plot(
    arm2_visit_order,
    interp1d(
        arm2_visit_order[::thin],
        dts2[::thin],
        kind="quadratic",
        fill_value="extrapolate",
    )(arm2_visit_order),
    c="k",
    alpha=0.5,
)
ax.scatter(
    arm2_visit_order[::thin],
    dts2[::thin],
    edgecolor="black",
    facecolors="none",
    s=40,
    label="arm2 'time' steps (subsampled)",
    zorder=100,
    marker=">",
)

ax.set_xlabel(r"($\leftarrow$arm 1)  SOM index  (arm 2$\rightarrow$)")
ax.set_ylabel("Point-to-point SOM-projected distance [deg]")
ax.legend(fontsize=12)

# ----

fig.colorbar(cm.ScalarMappable(norm=Normalize(-len(ds1), len(ds2)), cmap=cmap), ax=ax)

# ----

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

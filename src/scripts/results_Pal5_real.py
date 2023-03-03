"""Plot results of running TrackStream on Palomar 5."""


import pathlib

import astropy.units as u
import matplotlib.pyplot as plt
import paths
from conf import ARM1_KW, ARM2_KW, LABEL_KW, LENGTH, SOM_KW, get_pal5_stream, handler_map, plot_kalman, plot_origin
from trackstream.frame import fit_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import AngularWidth, UnitSphericalWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
LABEL_KW = LABEL_KW | {"fontsize": 19}

##############################################################################
# SCRIPT
##############################################################################

# Get stream
stream = get_pal5_stream()
# Fit frame
stream = fit_stream(stream, force=True, rot0=150 * u.deg)

stream_width0 = Widths.from_format({"length": UnitSphericalWidth(lon=2 * u.deg, lat=5 * u.deg)})

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 10, "sigma": 2, "learning_rate": 0.5, "prototype_kw": {"maxsep": 20 * u.deg}},
    kalman_kw={"width0": stream_width0},
)

_ = stream.fit_track(
    fitters=fitters,
    tune=True,
    som_kw={"num_iteration": int(1e4), "progress": True},
    kalman_kw={"dtmin": Times({LENGTH: 0.4 * u.deg}), "width_min": Widths({LENGTH: AngularWidth(0.5 * u.deg)})},
)


# ===================================================================
# Plot

fig, axs = plt.subplots(4, 1, figsize=(8, 12))

frame = stream["arm1"].frame
arm1c = stream["arm1"].coords.transform_to(frame)
arm2c = stream["arm2"].coords.transform_to(frame)
origin = stream["arm1"].origin.transform_to(frame)

# arms
axs[0].scatter(arm1c.icrs.ra, arm1c.icrs.dec, label="arm 1", **ARM1_KW)
plot_origin(axs[0], origin.icrs.ra, origin.icrs.dec)
axs[0].scatter(arm2c.icrs.ra, arm2c.icrs.dec, label="arm 2", **ARM2_KW)

axs[0].set_xlabel("RA (ICRS) [deg]", **LABEL_KW)
axs[0].set_ylabel("Dec (ICRS) [deg]", **LABEL_KW)
axs[0].legend(loc="lower right", handler_map=handler_map)


# ===================================

# arms
axs[1].scatter(arm1c.lon, arm1c.lat, label="arm 1", **ARM1_KW)
plot_origin(axs[1], origin.lon, origin.lat)
axs[1].scatter(arm2c.lon, arm2c.lat, label="arm 2", **ARM2_KW)

axs[1].set_xlabel(r"$\ell$ (Stream) [deg]", **LABEL_KW)
axs[1].set_ylabel("$b$ (Stream) [deg]", **LABEL_KW)
axs[1].legend(handler_map=handler_map)


# ===================================
# SOM plot

som1 = stream["arm1"].track.som
ps1 = som1.prototypes.transform_to(frame)
som2 = stream["arm2"].track.som
ps2 = som2.prototypes.transform_to(frame)

# arms
axs[2].scatter(arm1c.lon, arm1c.lat, label="arm 1", **ARM1_KW)
plot_origin(axs[2], origin.lon, origin.lat)
axs[2].scatter(arm2c.lon, arm2c.lat, label="arm 2", **ARM2_KW)

# soms
axs[2].plot(ps1.lon, ps1.lat, **SOM_KW, label="SOM")
axs[2].plot(ps2.lon, ps2.lat, **SOM_KW)

axs[2].set_xlabel(r"$\ell$ (Stream) [deg]", **LABEL_KW)
axs[2].set_ylabel("$b$ (Stream) [deg]", **LABEL_KW)
axs[2].legend(ncol=2, handler_map=handler_map)

# ===================================
# Kalman Filter

axs[3].scatter(arm1c.lon, arm1c.lat, label="arm 1", **ARM1_KW)
plot_origin(axs[3], origin.lon, origin.lat)
axs[3].scatter(arm2c.lon, arm2c.lat, label="arm 2", **ARM2_KW)

# kalmans
plot_kalman(axs[3], stream["arm1"], kind="positions", label="track", step=1)
plot_kalman(axs[3], stream["arm2"], kind="positions", step=1)

axs[3].set_xlabel(r"$\ell$ (Stream) [deg]", **LABEL_KW)
axs[3].set_ylabel("$b$ (Stream) [deg]", **LABEL_KW)
axs[3].legend(ncol=2, handler_map=handler_map)

# ===================================

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

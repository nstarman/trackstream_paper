"""Plot results of running TrackStream on Palomar 5."""

##############################################################################
# IMPORTS

# STDLIB
import pathlib

# THIRD-PARTY
import astropy.units as u
import matplotlib.pyplot as plt

# FIRST-PARTY
from trackstream.frame import fit_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.fit import Times
from trackstream.track.width import UnitSphericalWidth, Widths

# LOCAL
import paths
from conf import LENGTH, color1, color2, get_pal5_stream, plot_kalman

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

frame = stream["arm1"].frame
arm1c = stream["arm1"].coords.transform_to(frame)
arm2c = stream["arm2"].coords.transform_to(frame)
origin = stream["arm1"].origin.transform_to(frame)

# arm1
axs[0, 0].scatter(arm1c.icrs.ra, arm1c.icrs.dec, s=1, color=color1, label="arm 1", marker="*")
# origin
axs[0, 0].scatter(origin.icrs.ra, origin.icrs.dec, s=10, color="red", label="origin")
axs[0, 0].scatter(origin.icrs.ra, origin.icrs.dec, s=800, facecolor="None", edgecolor="red")
# arm2
axs[0, 0].scatter(arm2c.icrs.ra, arm2c.icrs.dec, s=1, color=color2, label="arm 2", marker="*")

axs[0, 0].set_xlabel("RA (ICRS) [deg]")
axs[0, 0].set_ylabel("Dec (ICRS) [deg]")

# ===================================

# arm1
axs[1, 0].scatter(arm1c.lon, arm1c.lat, s=1, color=color1, label="arm 1", marker="*")
# origin
axs[1, 0].scatter(origin.lon, origin.lat, s=10, color="red", label="origin")
axs[1, 0].scatter(origin.lon, origin.lat, s=800, facecolor="None", edgecolor="red")
# arm2
axs[1, 0].scatter(arm2c.lon, arm2c.lat, s=1, color=color2, label="arm 2", marker="*")

axs[1, 0].set_xlabel(r"$\ell$ (Stream) [deg]")
axs[1, 0].set_ylabel("$b$ (Stream) [deg]")

# ===================================
# SOM plot

som1 = stream["arm1"].track.som
ps1 = som1.prototypes.transform_to(frame)
som2 = stream["arm2"].track.som
ps2 = som2.prototypes.transform_to(frame)

# arm1
axs[2, 0].scatter(arm1c.lon, arm1c.lat, s=1, color=color1, label="arm 1", marker="*")
# origin
axs[2, 0].scatter(origin.lon, origin.lat, s=10, color="red", label="origin")
axs[2, 0].scatter(origin.lon, origin.lat, s=800, facecolor="None", edgecolor="red")
# arm2
axs[2, 0].scatter(arm2c.lon, arm2c.lat, s=1, color=color2, label="arm 2", marker="*")

# som 1
axs[2, 0].plot(ps1.lon, ps1.lat, c="k")
axs[2, 0].scatter(ps1.lon, ps1.lat, marker="P", edgecolors="black", facecolor="none")
# som 2
axs[2, 0].plot(ps2.lon, ps2.lat, c="k")
axs[2, 0].scatter(ps2.lon, ps2.lat, marker="P", edgecolors="black", facecolor="none")

axs[2, 0].set_xlabel(r"$\ell$ (Stream) [deg]")
axs[2, 0].set_ylabel("$b$ (Stream) [deg]")


# ===================================
# Kalman Filter

# arm1
axs[3, 0].scatter(arm1c.lon, arm1c.lat, s=1, color=color1, label="arm 1", marker="*")
# origin
axs[3, 0].scatter(origin.lon, origin.lat, s=10, color="red", label="origin")
axs[3, 0].scatter(origin.lon, origin.lat, s=800, facecolor="None", edgecolor="red")
# arm2
axs[3, 0].scatter(arm2c.lon, arm2c.lat, s=1, color=color2, label="arm 2", marker="*")

# kalman 1
for arm in ("arm1", "arm2"):
    plot_kalman(axs[3, 0], stream[arm], kind="positions")


# ===================================


for ax in fig.axes:
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.legend(fontsize=15)

axs[-1, 0].legend(fontsize=15, ncol=2)

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

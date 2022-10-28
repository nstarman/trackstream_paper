"""Plot an example of the SOM method."""

##############################################################################
# IMPORTS

# STDLIB
import pathlib
import warnings

# THIRD-PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from trackstream.frame import fit_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.width import UnitSphericalWidth, Widths

# FIRST-PARTY
import paths
from conf import LENGTH, cmap, cnorm, get_NGC5466_stream

##############################################################################
# SCRIPT

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
    som_kw={"nlattice": 10, "sigma": 2, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    track = stream.fit_track(
        fitters=fitters, som_kw={"num_iteration": int(1e3), "progress": False}, composite=True, force=True
    )


# ===================================================================
# Plot

fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(
    2, 2, width_ratios=(30, 1), height_ratios=(2, 7), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05
)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_cbar = fig.add_subplot(gs[1, 1])

ax_histx.tick_params(axis="x", labelbottom=False)

# ----

display_offset = u.Quantity(10, u.deg)

pts1 = ax.scatter(
    stream["arm1"].coords.lon,
    stream["arm1"].coords.lat,
    c=np.linspace(0, -1, len(stream["arm1"])),
    norm=cnorm,
    cmap=cmap,
    s=40,
    label="mock stream (arm1)",
    marker="*",
)
ax.scatter(
    stream["arm1"].track.som.prototypes.lon,
    stream["arm1"].track.som.prototypes.lat,
    edgecolor="gray",
    facecolors="none",
    s=40,
    zorder=100,
    marker="o",
)
for p in stream["arm1"].track.som.prototypes:
    ax.plot((p.lon.value, p.lon.value), (p.lat.value, p.lat.value + display_offset.value), c="gray", alpha=0.5)

ax.scatter(
    stream["arm1"].track.som.prototypes.lon,
    stream["arm1"].track.som.prototypes.lat + display_offset,
    edgecolor="none",
    facecolors=cmap(-0.5),
    s=40,
    label="SOM prototypes (arm 1)",
    zorder=100,
    marker="<",
)

ax.scatter(
    stream["arm2"].coords.lon,
    stream["arm2"].coords.lat,
    c=np.linspace(0, 1, len(stream["arm2"])),
    norm=cnorm,
    cmap=cmap,
    s=40,
    label="mock stream (arm2)",
    marker="*",
)
ax.scatter(
    stream["arm2"].track.som.prototypes.lon,
    stream["arm2"].track.som.prototypes.lat,
    edgecolor="gray",
    facecolors="none",
    s=40,
    zorder=100,
    marker="o",
    alpha=0.5,
)
for p in stream["arm2"].track.som.prototypes:
    ax.plot(
        (p.lon.value, p.lon.value),
        (p.lat.value, p.lat.value + display_offset.value),
        c="gray",
        alpha=0.5,
    )
ax.scatter(
    stream["arm2"].track.som.prototypes.lon,
    stream["arm2"].track.som.prototypes.lat + display_offset,
    edgecolor="none",
    facecolors=cmap(0.8),
    s=40,
    label="SOM prototypes (arm2)",
    zorder=100,
    marker=">",
)

cbar = plt.colorbar(pts1, cax=ax_cbar)  # TODO? include pts2 in colorbar
cbar.solids.set_edgecolor("face")
cbar.ax.set_ylabel("SOM ordering", fontsize=14)

# separting line between stream arms
ax.axvline(0, c="black", ls=":")

# ax.set_aspect("equal")
ax.set_xlim(None, 30)  # FIXME! show the whole stream

ax.set_xlabel(r"$\phi_1$ (Stream) [$\degree$]", fontsize=16)
ax.set_ylabel(r"$\phi_2$ (Stream) [$\degree$]", fontsize=16)
ax.legend(loc="lower left", fontsize=13)
ax.grid()

# ----

arms_lon = stream.coords.lon.wrap_at("180d").value
_, bins, patches = ax_histx.hist(arms_lon, bins=50, density=True, log=True, color="gray")

ax_histx.set_ylabel("log density", y=0.2)

# ----

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

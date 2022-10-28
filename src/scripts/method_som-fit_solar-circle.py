"""Show SOM run on a solar-circle mock stream."""

##############################################################################
# IMPORTS

# STDLIB
import pathlib
import warnings

# THIRD-PARTY
import asdf
import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke
from trackstream import Stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.width import Cartesian3DWidth, Widths

# FIRST-PARTY
import paths
from conf import LENGTH, cmap

##############################################################################
# SCRIPT
##############################################################################

# Load datas
with asdf.open(paths.data / "solar_circle.asdf") as af:
    origin = af["origin"]

    data = af["data"]
    data = data.group_by("arm")
    data.add_index("arm")

# The Stream
frame = coords.Galactocentric()
stream = Stream.from_data(data, origin=origin, frame=frame, name="Solar Circle")

# The Stream width
stream_width0 = Widths(
    {LENGTH: Cartesian3DWidth(x=u.Quantity(100, u.pc), y=u.Quantity(100, u.pc), z=u.Quantity(100, u.pc))}
)

# Fitter
fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=False,
    kinematics=False,
    som_kw={"nlattice": 25, "sigma": 2, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    _ = stream.fit_track(
        fitters=fitters, som_kw={"num_iteration": int(1e4), "progress": False}, composite=True, force=True
    )


# ===================================================================
# Plot


fig = plt.figure(figsize=(8, 4))
gs = GridSpec(3, 3, width_ratios=[14, 14, 1], height_ratios=[1, 25, 1])
# doing this weird GridSpec to get the colorbar to look good

color_array = np.arange(len(stream.data_coords))

# ----
# Plot 1 : Unordered

ax1 = fig.add_subplot(gs[:, 0])

ax1.scatter(stream.data_coords.x, stream.data_coords.y, c=color_array, cmap=cmap, s=40, label="mock stream", marker="*")

for som in stream.track.som.values():
    ax1.scatter(
        som.init_prototypes.x, som.init_prototypes.y, edgecolor="black", facecolors="none", s=40, label="SOM prototypes"
    )

# origin
ax1.scatter(stream.origin.x, stream.origin.y, s=10, color="black")
circle = plt.Circle(
    (stream.origin.x.value, stream.origin.y.value),
    1,
    clip_on=False,
    zorder=10,
    linewidth=2.0,
    edgecolor="black",
    facecolor="none",
    path_effects=[withStroke(linewidth=7, foreground=(1, 1, 1, 1))],
)
ax1.add_artist(circle)
ax1.text(
    stream.origin.x.value,
    stream.origin.y.value - 1,
    "origin",
    zorder=100,
    ha="center",
    va="center",
    weight="bold",
    color="black",
    style="italic",
    fontfamily="monospace",
    path_effects=[withStroke(linewidth=5, foreground=(1, 1, 1, 1))],
)

ax1.set_aspect("equal")
ax1.set_xlabel("x (Galactocentric) [kpc]", fontsize=13)
ax1.set_ylabel("y (Galactocentric) [kpc]", fontsize=13)
ax1.legend(loc="lower left", fontsize=13)
ax1.grid()

# ----
# Plot 2 : Ordered by SOM

ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[1, 2])  # colorbar

pts = ax2.scatter(stream.coords.x, stream.coords.y, c=color_array, cmap=cmap, s=40, marker="*", label="mock stream")
for som in stream.track.som.values():
    ax2.scatter(som.prototypes.x, som.prototypes.y, edgecolor="black", facecolors="none", s=40, label="SOM prototypes")

cbar = plt.colorbar(pts, cax=ax3)
cbar.solids.set_edgecolor("face")
cbar.ax.set_ylabel("SOM ordering", fontsize=14)

ax2.set_aspect("equal")
ax2.set_xlabel("x (Galactocentric) [kpc]", fontsize=16)
ax2.set_ylabel(None)
ax2.tick_params("y", labelleft=False)
ax2.legend(loc="lower left", fontsize=13)
ax2.grid()

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

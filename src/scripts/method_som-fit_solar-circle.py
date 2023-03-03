"""Show SOM run on a solar-circle mock stream."""

import pathlib
import warnings

import asdf
import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from conf import ARM_KW, LABEL_KW, LENGTH, SOM_KW, fraction_format, handler_map, plot_origin
from matplotlib.gridspec import GridSpec
from trackstream import Stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.width import Cartesian3DWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")

LABEL_KW = LABEL_KW | {"fontsize": 16}

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
    {LENGTH: Cartesian3DWidth(x=100 * u.pc, y=100 * u.pc, z=100 * u.pc)},
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
        fitters=fitters,
        som_kw={"num_iteration": int(1e4), "progress": False},
        composite=True,
        force=True,
    )


# ===================================================================
# Plot


# doing this weird GridSpec to get the colorbar to look good
fig = plt.figure(figsize=(8, 4))
gs = GridSpec(3, 3, width_ratios=[14, 14, 1], height_ratios=[1, 25, 1])
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1], sharey=ax1)
ax_cbar = fig.add_subplot(gs[1, 2])  # colorbar

color_array = np.linspace(-1, 1, len(stream.data_coords))

# ----
# Plot 1 : Unordered

ax1.scatter(stream.data_coords.x, stream.data_coords.y, c=color_array, label="mock stream", **ARM_KW | {"s": 40})

for som in stream.track.som.values():
    ax1.plot(som.init_prototypes.x, som.init_prototypes.y, label="SOM prototypes", **SOM_KW)

# origin
plot_origin(ax1, stream.origin.x, stream.origin.y)

ax1.set_aspect("equal")
ax1.set_xlabel("x (GXYC) [kpc]", **LABEL_KW)
ax1.set_ylabel("y (GXYC) [kpc]", **LABEL_KW)
ax1.tick_params(axis="both", which="major", labelsize=14)
lgnd = ax1.legend(loc="lower left", handler_map=handler_map)

# ----
# Plot 2 : Ordered by SOM

pts = ax2.scatter(stream.coords.x, stream.coords.y, c=color_array, label="mock stream", **ARM_KW | {"s": 40})
plot_origin(ax2, stream.origin.x, stream.origin.y)
# prototypes
for som in stream.track.som.values():
    ax2.plot(som.prototypes.x, som.prototypes.y, label="SOM prototypes", **SOM_KW)

# colorbar
cbar = plt.colorbar(pts, cax=ax_cbar)
cbar.solids.set_edgecolor("face")
cbar.ax.set_ylabel("SOM ordering", **LABEL_KW)
cbar.ax.tick_params(axis="both", which="major", labelsize=14)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(fraction_format))

ax2.set_aspect("equal")
ax2.set_xlabel("x (GXYC) [kpc]", **LABEL_KW)
ax2.set_ylabel("")
ax2.tick_params("y", labelleft=False, labelsize=14)
ax2.tick_params(axis="x", which="major", labelsize=14)

fig.tight_layout()
fig.savefig(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf"), bbox_inches="tight")

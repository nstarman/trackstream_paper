"""Plot an example of the SOM method."""

import pathlib
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from conf import ARM_KW, LABEL_KW, LENGTH, SOM_KW, color1, color2, fraction_format, get_ngc5466_stream, handler_map
from trackstream.frame import fit_stream
from trackstream.track import FitterStreamArmTrack
from trackstream.track.width import UnitSphericalWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
LABEL_KW = LABEL_KW | {"fontsize": 16}

##############################################################################
# SCRIPT

# Get stream
stream = get_ngc5466_stream()
# Fit frame
stream = fit_stream(stream, force=True, rot0=110 * u.deg)

object.__setattr__(stream["arm1"], "data", stream["arm1"].data[stream["arm1"].coords.lon.value < 10])
object.__setattr__(stream["arm2"], "data", stream["arm2"].data[stream["arm2"].coords.lon.value < 50])

# Fit track. This is overkill for just showing the SOM, but fitting a full
# track also re-orders the stream and makes plotting much easier.
stream_width0 = Widths({LENGTH: UnitSphericalWidth(lon=2 * u.deg, lat=2 * u.deg)})

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
        fitters=fitters,
        som_kw={"num_iteration": int(1e3), "progress": True},
        composite=True,
        force=True,
    )


# ===================================================================
# Plot

fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=(30, 1),
    height_ratios=(2, 7),
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.05,
    hspace=0.05,
)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_cbar = fig.add_subplot(gs[1, 1])

ax_histx.tick_params(axis="x", labelbottom=False, labelsize=14)

# ----

display_offset = 8 * u.deg

pts1 = ax.scatter(
    stream["arm1"].coords.lon,
    stream["arm1"].coords.lat,
    c=np.linspace(0, -1, len(stream["arm1"])),
    label="mock stream (arm1)",
    **ARM_KW | {"s": 40},
)
ax.scatter(
    stream["arm2"].coords.lon,
    stream["arm2"].coords.lat,
    c=np.linspace(0, 1, len(stream["arm2"])),
    label="mock stream (arm2)",
    **ARM_KW | {"s": 40},
)

# -------------------------------------------------------------------
# Prototypes

ps1 = stream["arm1"].track.som.prototypes

ax.plot(ps1.lon, ps1.lat, label="SOM", **SOM_KW | {"markersize": 8, "markeredgecolor": (0.7, 0.7, 0.7, 0.75)})
for p in ps1:
    ax.plot((p.lon.value, p.lon.value), (p.lat.value, p.lat.value + display_offset.value), c="gray", alpha=0.5)

ax.scatter(
    ps1.lon,
    ps1.lat + display_offset,
    edgecolor="none",
    facecolors=color1,
    s=50,
    zorder=100,
    marker="<",
)

ps2 = stream["arm2"].track.som.prototypes

ax.plot(ps2.lon, ps2.lat, **SOM_KW | {"markersize": 8, "markeredgecolor": (0.7, 0.7, 0.7, 0.75)})
for p in ps2:
    ax.plot((p.lon.value, p.lon.value), (p.lat.value, p.lat.value + display_offset.value), c="gray", alpha=0.5)
ax.scatter(ps2.lon, ps2.lat + display_offset, edgecolor="none", facecolors=color2, s=50, zorder=100, marker=">")

# separating line between stream arms
ax.axvline(0, c="black", ls=":")

# colorbar
cbar = plt.colorbar(pts1, cax=ax_cbar)  # TODO? include pts2 in colorbar
cbar.solids.set_edgecolor("face")
cbar.ax.set_ylabel("SOM ordering", **LABEL_KW)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(fraction_format))
cbar.ax.tick_params("both", labelsize=14)

# adjust
ax.set_xlim(None, 30)
ax.set_xlabel(r"$\phi_1$ (Stream) [$\degree$]", **LABEL_KW)
ax.set_ylabel(r"$\phi_2$ (Stream) [$\degree$]", **LABEL_KW)
ax.tick_params("both", labelsize=14)

lgnd = ax.legend(loc="lower left", handler_map=handler_map)
lgnd.legend_handles[0].set_color(color1)
lgnd.legend_handles[0].set_facecolor(color1)
lgnd.legend_handles[1].set_color(color2)

# ----
# Upper histogram

arms_lon = stream.coords.lon.wrap_at("180d").value
_, bins, patches = ax_histx.hist(arms_lon, bins=50, density=True, log=True, color="gray")

ax_histx.set_ylabel("density", y=0.2, **LABEL_KW)

# ----
# Save

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

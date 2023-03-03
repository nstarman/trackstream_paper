"""Run TrackStream on GD-1 data."""

from __future__ import annotations

import pathlib

import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.table import QTable
from conf import ARM1_KW, ARM_KW, LABEL_KW, LENGTH, SOM_KW, color1, handler_map, plot_kalman, plot_origin
from gala import coordinates as gc
from trackstream import Stream
from trackstream.track.fit import FitterStreamArmTrack, Times
from trackstream.track.width import AngularWidth, UnitSphericalWidth, Widths

plt.style.use(pathlib.Path(__file__).parent / "paper.mplstyle")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
LABEL_KW = LABEL_KW | {"fontsize": 19}

##############################################################################
# SCRIPTS
##############################################################################

# origin
origin = coords.SkyCoord(
    ra=-135.01 * u.deg,
    dec=58 * u.deg,
    pm_ra_cosdec=-7.78 * u.Unit("mas / yr"),
    pm_dec=-7.85 * u.Unit("mas / yr"),
)

# read data
full_data = QTable.read(paths.data / "PWB18_thinsel.ecsv")
full_data = full_data[full_data["pm_mask"] & full_data["gi_cmd_mask"] & full_data["stream_track_mask"]]
data = full_data[full_data["thin_stream_sel"]]

dif = coords.UnitSphericalDifferential(d_lon=data["pm_phi1_cosphi2_no_reflex"], d_lat=data["pm_phi2_no_reflex"])
rep = coords.UnitSphericalRepresentation(lon=data["phi1"], lat=data["phi2"], differentials=dif)
data["coords"] = coords.SkyCoord(gc.GD1Koposov10(rep))
data["arm"] = "arm1"
data = data.group_by("arm")

data_err = QTable()
data_err["ra_err"] = 0 * data["ra"]  # (for the shape)
data_err["dec_err"] = 0 * u.deg
data_err["arm"] = data["arm"]

# stream
stream = Stream.from_data(data, data_err=data_err, origin=origin, frame=gc.GD1Koposov10(), name="GD-1")


stream_width0 = Widths.from_format({"length": UnitSphericalWidth(lat=1 * u.deg, lon=1 * u.deg)})

fitters = FitterStreamArmTrack.from_format(
    stream,
    onsky=True,
    kinematics=False,
    som_kw={"nlattice": 8, "sigma": 0.1, "learning_rate": 0.5},
    kalman_kw={"width0": stream_width0},
)

_ = stream.fit_track(
    fitters=fitters,
    force=True,
    tune=True,
    composite=True,
    som_kw={"num_iteration": int(5e3), "progress": True},
    kalman_kw={"dtmax": Times({LENGTH: 1 * u.deg}), "width_min": Widths({LENGTH: AngularWidth(2 * u.deg)})},
)


# ===================================================================
# Plot

frame = stream["arm1"].frame
arm1c = stream["arm1"].coords
origin = stream["arm1"].origin.transform_to(frame)

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
full_name = stream["arm1"].full_name or ""

# Plot stream in system frame
axs[0].scatter(full_data["phi1"], full_data["phi2"], **ARM1_KW | {"color": "k"})
axs[0].scatter(arm1c.phi1, arm1c.phi2, label=full_name, **ARM1_KW | {"s": 5})
plot_origin(axs[0], origin.phi1, origin.phi2)

# -------------------------------------------------------------
# SOM

axs[1].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")
axs[1].scatter(arm1c.phi1, arm1c.phi2, c=np.linspace(-1, 1, len(arm1c)), label=full_name, **ARM_KW)
plot_origin(axs[1], origin.phi1, origin.phi2)
# som
som1 = stream["arm1"].track.som
ps1 = som1.prototypes.transform_to(frame)
axs[1].plot(ps1.phi1, ps1.phi2, label="SOM", **SOM_KW, markeredgewidth=2)


# -------------------------------------------------------------
# Kalman filter plot

axs[2].scatter(full_data["phi1"], full_data["phi2"], s=1, c="k")
axs[2].scatter(arm1c.phi1, arm1c.phi2, c=np.linspace(-1, 1, len(arm1c)), label=full_name, **ARM_KW)
plot_origin(axs[2], origin.phi1, origin.phi2)
plot_kalman(axs[2], stream["arm1"], kind="positions", label="track", step=1, zorder=10)


# -------------------------------------------------------------

axs[0].set_xlabel("")
axs[0].set_xticklabels([])
axs[0].set_ylabel(r"$\phi_2$ (GD1) [deg]", **LABEL_KW)
lgnd = axs[0].legend(ncol=2, loc="lower center", handler_map=handler_map)

axs[1].set_xlabel("")
axs[1].set_xticklabels([])
axs[1].set_ylabel(r"$\phi_2$ (GD1) [deg]", **LABEL_KW)
lgnd = axs[1].legend(ncol=3, loc="lower center", handler_map=handler_map)
lgnd.legend_handles[0].set_color(color1)

axs[2].set_xlabel(r"$\phi_1$ (GD1) [deg]", **LABEL_KW)
axs[2].set_ylabel(r"$\phi_2$ (GD1) [deg]", **LABEL_KW)
lgnd = axs[2].legend(ncol=3, loc="lower center", handler_map=handler_map)
lgnd.legend_handles[0].set_color(color1)

# -------------------------------------------------------------

fig.tight_layout()
fig.savefig(str(paths.figures / pathlib.Path(__file__).name.replace(".py", ".pdf")), bbox_inches="tight")

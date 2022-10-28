"""Download data from Vasiliev (2019) and save as an ECSV."""

##############################################################################
# IMPORTS

# STDLIB
import copy

# THIRD-PARTY
import asdf
import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord, concatenate
from astropy.table import QTable
from galpy.actionAngle import actionAngle, actionAngleIsochroneApprox
from galpy.df import streamdf
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, Potential

# FIRST-PARTY
import paths
from conf import RO, VO, get_from_Vasiliev2019_table

##############################################################################
# PARAMETERS

np.random.seed(4)

##############################################################################
# CODE


def streamdf_botharms(
    pot: Potential | list[Potential],
    aA: actionAngle,
    prog: SkyCoord,
    sigv: u.Quantity = u.Quantity(0.365, u.km / u.s),
    tdisrupt: u.Quantity = u.Quantity(4.5, u.Gyr),
) -> tuple[streamdf, streamdf]:
    """Run StreamDF for both arms.

    Parameters
    ----------
    pot : Potential or list[Potential]
        The potential in which to integrate the StreamDF.
    aA : actionAngle
        The action-angle transformer for integrating the StreamDF.
    prog : SkyCoord
        The progenitor coordinates.
    sigv : Quantity, optional
        The dispersion, by default u.Quantity(0.365, u.km / u.s).
    tdisrupt : Quantity, optional
        The disruption time, by default u.Quantity(4.5, u.Gyr)

    Returns
    -------
    tuple[streamdf, streamdf]
        Two integrated stream-df.
    """
    # Progenitor Orbit
    progenitor = Orbit(prog, ro=RO, vo=VO)

    # Run stream DF for each arm
    arguments = {
        "sigv": sigv,
        "progenitor": progenitor,
        "pot": pot,
        "aA": aA,
        "nTrackChunks": 11,
        "tdisrupt": tdisrupt,
        "ro": RO,
        "vo": VO,
    }
    trailing = streamdf(**arguments, leading=False)
    leading = streamdf(**arguments, leading=True)

    return trailing, leading


def galactocentric_track_from_sdf(sdf: streamdf) -> SkyCoord:
    """Get the Galactocentric coordinates from StreamDF.

    Parameters
    ----------
    sdf : `galpy.df.streamdf`
        The StreamDF instance from which to extract the coordinates.

    Returns
    -------
    SkyCoord
        Coordinates from ``sdf``.
    """
    x = sdf._interpolatedObsTrackXY[:, 0] * RO
    y = sdf._interpolatedObsTrackXY[:, 1] * RO
    z = sdf._interpolatedObsTrackXY[:, 2] * RO

    v_x = sdf._interpolatedObsTrackXY[:, 3] * VO
    v_y = sdf._interpolatedObsTrackXY[:, 4] * VO
    v_z = sdf._interpolatedObsTrackXY[:, 5] * VO

    # galpy is left-handed
    sc = SkyCoord(Galactocentric(x=-x, y=y, z=z, v_x=-v_x, v_y=v_y, v_z=v_z))
    return sc


##############################################################################
# SCRIPT
##############################################################################

# The potential
pot = copy.deepcopy(MWPotential2014)
aAI = actionAngleIsochroneApprox(pot=pot, b=0.8)

# Read progenitor. Requires download_Vasiliev2019_table.py
prog_sc = get_from_Vasiliev2019_table("NGC 104")
prog_sc = prog_sc.transform_to("galactocentric")

# Run Stream DFs
sdft, sdfl = streamdf_botharms(pot, aAI, prog_sc, sigv=u.Quantity(0.365, u.km / u.s), tdisrupt=u.Quantity(4.5, u.Gyr))

# True track
true_sct = galactocentric_track_from_sdf(sdft)
true_scl = galactocentric_track_from_sdf(sdfl)

# Samples of each arm
x, y, z, v_x, v_y, v_z = sdft.sample(200, xy=True)
samplest = Galactocentric(x=-x, y=y, z=z, v_x=-v_x, v_y=v_y, v_z=v_z)

x, y, z, v_x, v_y, v_z = sdfl.sample(200, xy=True)
samplesl = Galactocentric(x=-x, y=y, z=z, v_x=-v_x, v_y=v_y, v_z=v_z)

# -------------------------------------------------------------------
# Save results

data = QTable()
crd = concatenate((samplest[::-1], samplesl))
data["coords"] = crd
data["arm"] = "arm1"
data["arm"][len(samplest) :] = "arm2"
data = data.group_by("arm")

af = asdf.AsdfFile()
af.tree["trailing"] = true_sct
af.tree["leading"] = true_scl
af.tree["samples"] = data
af.write_to(paths.data / "streamdf.asdf")

"""Simulate Stream-Spray Distribution Function."""

import copy

import asdf
import astropy.coordinates as coords
import astropy.units as u
import galpy.potential as gpot
import numpy as np
import paths
from astropy.table import QTable, vstack
from astropy.units import Quantity
from conf import RO, VO, get_from_vasiliev2019_table
from galpy.df import streamspraydf
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, Potential

##############################################################################
# PARAMETERS

GC_47TUC_Mass = 7e5 * u.solMass  # Marks and Kroupa '10

##############################################################################
# CODE


def streamspraydf_botharms(  # noqa: PLR0913
    progenitor_mass: Quantity,
    progenitor_sc: coords.SkyCoord,
    pot: Potential | list[Potential] = MWPotential2014,
    tdisrupt: Quantity = 4.5 * u.Gyr,
    meankvec: list[float] | tuple[float, ...] = (2.0, 0.0, 0.3, 0.0, 0.0, 0.0),
    sigkvec: list[float] | tuple[float, ...] = (0.4, 0.0, 0.4, 0.5, 0.5, 0.0),
) -> tuple[streamspraydf, streamspraydf, Orbit]:
    """Return a streamspraydf for both trailing and leading arms.

    Parameters
    ----------
    progenitor_mass : Quantity
        The mass of the progenitor.
    progenitor_sc : SkyCoord
        SkyCoord of the progenitor.
    pot : Potential | list[Potential], optional
        The galpy potential, by default ``MWPotential2014``.
    tdisrupt : Quantity, optional
        Time of disruption, by default ``4.5 * u.Gyr``.
    meankvec : list[float], optional
        Mean of k-vec, by default ``[2.0, 0.0, 0.3, 0.0, 0.0, 0.0]``.
    sigkvec : list[float], optional
        sigma of k-vec, by default ``[0.4, 0.0, 0.4, 0.5, 0.5, 0.0]``.

    Returns
    -------
    trailing, leading : streamspraydf, streamspraydf
    progenitor : Orbit
    """
    # Stream progenitor
    progenitor = Orbit(progenitor_sc, ro=RO, vo=VO)
    progenitor.turn_physical_off()

    # Run stream-spray for each arm
    arguments = {
        "pot": pot,
        "progenitor_mass": progenitor_mass,
        "progenitor": progenitor,
        "tdisrupt": tdisrupt,
        "meankvec": list(meankvec),
        "sigkvec": list(sigkvec),
        "ro": RO,
        "vo": VO,
    }
    trailing = streamspraydf(**arguments, leading=False)
    trailing.turn_physical_off()
    leading = streamspraydf(**arguments, leading=True)
    leading.turn_physical_off()

    return trailing, leading, progenitor


##############################################################################
# SCRIPT
##############################################################################

# Get stream
gc47t_sc = get_from_vasiliev2019_table("NGC 104")
pot = copy.deepcopy(MWPotential2014)
gpot.turn_physical_off(pot)

# Origin at progenitor -> frame (necessary b/c galpy's GC)
frame = coords.Galactocentric()
origin = gc47t_sc.transform_to(frame)

# Parameters
tdisrupt = Quantity(2, u.Gyr)

# Run stream-spray
gc47t_sst, gc47t_ssl, gc47t_prog = streamspraydf_botharms(
    pot=pot,
    progenitor_mass=GC_47TUC_Mass,
    progenitor_sc=gc47t_sc,
    tdisrupt=tdisrupt,
)

# Integrate
np.random.seed(4)  # noqa: NPY002
RvRl, dtl = gc47t_ssl.sample(n=200, returndt=True, integrate=True)
RvRt, dtt = gc47t_sst.sample(n=200, returndt=True, integrate=True)

# Get SkyCoord(s)
gc47t_ssl_sc = RvRl.SkyCoord(ro=RO, vo=VO).transform_to(frame)
gc47t_sst_sc = RvRt.SkyCoord(ro=RO, vo=VO).transform_to(frame)

# Make data table
datal = QTable()
datal["coords"] = gc47t_ssl_sc
datal["arm"] = "arm1"
datat = QTable()
datat["coords"] = gc47t_sst_sc
datat["arm"] = "arm2"
data = vstack((datal, datat))
data = data.group_by("arm")

# Save results
af = asdf.AsdfFile()
af.tree["data"] = data
af.tree["origin"] = origin
af.write_to(paths.data / "streamspraydf.asdf")

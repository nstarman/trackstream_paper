"""Mock stream at the solar circle."""

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import asdf
import astropy.units as u
import numpy as np
import paths
from astropy.coordinates import BaseRepresentation, CartesianRepresentation, SkyCoord
from astropy.table import QTable
from astropy.units import Quantity
from galpy import potential
from galpy.orbit import Orbit
from numpy.random import Generator, default_rng
from trackstream._typing import CoordinateType

__author__ = "Nathaniel Starkman"
__credits__ = ["Jo Bovy"]

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


##############################################################################
# PARAMETERS

stop: int = 200
num: int = 100
unit = u.Unit("Myr")

UnitType: TypeAlias = u.Unit | u.IrreducibleUnit | u.UnitBase | u.FunctionUnitBase
CoordinateLikeType: TypeAlias = CoordinateType | str
RepresentationLikeType: TypeAlias = BaseRepresentation | str

##############################################################################
# CODE


def get_orbit(stop: float = stop, num: int = num, unit: UnitType = unit) -> Orbit:
    """Get Orbit by integrating in a `galpy.potential.MWPotential2014`.

    Parameters
    ----------
    stop: float
        Stop time for integration.
    num : int
        Number of points in integration.
    unit : `~astropy.units.Unit`
        Unit of integration time.

    Returns
    -------
    rep: `~galpy.orbit.Orbit`
        Integrated orbit.
    """
    # create time integration array
    time = np.linspace(0, stop, num=num) * unit

    # integrate orbit
    o = Orbit()
    o.integrate(time, potential.MWPotential2014)

    return o


# -------------------------------------------------------------------


def make_ordered_orbit_data(
    stop: float = stop,
    num: int = num,
    unit: UnitType = unit,
    frame: CoordinateLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
) -> SkyCoord:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
        Stop time for integration.
    num : int
        Number of points in integration.
    unit : Unit
        Unit of integration time.

    frame : CoordinateLikeType
        Frame to transform to.
    representation_type : RepresentationLikeType
        Representation to transform to.

    Returns
    -------
    SkyCoord
        (`num`, 3) array
    """
    # Get orbit
    time = np.linspace(0, stop, num=num) * unit
    o = Orbit()
    o.integrate(time, potential.MWPotential2014)

    # Extract coordinates in correct frame
    sc = o.SkyCoord(o.time())
    tsc = sc.transform_to(frame)
    tsc.representation_type = representation_type

    return tsc


# -------------------------------------------------------------------


def make_unordered_orbit_data(
    stop: float = stop,
    num: int = num,
    unit: UnitType = unit,
    frame: CoordinateLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
) -> SkyCoord:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
        Stop time for integration.
    num : int
        Number of points in integration.
    unit : Unit
        Unit of integration time.

    frame : CoordinateLikeType
        Frame to transform to.
    representation_type : RepresentationLikeType
        Representation to transform to.

    Returns
    -------
    SkyCoord
        (`num`, 3) array
    """
    # Ordered data
    osc = make_ordered_orbit_data(stop=stop, num=num, unit=unit, frame=frame, representation_type=representation_type)
    # Shuffle the data
    rng = default_rng(seed=0)
    shuffler = np.arange(len(osc))  # start with index array
    rng.shuffle(shuffler)  # shuffle array in-place

    return cast("SkyCoord", osc[shuffler])


# -------------------------------------------------------------------


def make_noisy_orbit_data(  # noqa: PLR0913
    stop: float = stop,
    num: int = num,
    sigma: dict[str, u.Quantity] | None = None,
    unit: UnitType = unit,
    frame: CoordinateLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
    rnd: int | np.random.Generator | None = None,
) -> SkyCoord:
    """Make ordered orbit data.

    Parameters
    ----------
    stop : float
        Stop time for integration.
    num : int
        Number of points in integration.
    sigma : dict[str, Quantity] or None, optional
        Errors in Galactocentric Cartesian coordinates.
    unit : Unit
        Unit of integration time.

    frame : CoordinateLikeType
        Frame to transform to.
    representation_type : RepresentationLikeType
        Representation to transform to.
    rnd : int or `~numpy.random.Generator`, optional
        Random state.

    Returns
    -------
    SkyCoord
        (`num`, 3) array
    """
    # Get random state
    rnd = rnd if isinstance(rnd, Generator) else default_rng(seed=rnd)

    # Default error
    sig = {"x": 100 * u.pc, "y": 100 * u.pc, "z": 20 * u.pc} if sigma is None else sigma

    # Unordered data
    usc = make_unordered_orbit_data(
        stop=stop,
        num=num,
        unit=unit,
        frame="galactocentric",
        representation_type="cartesian",
    )

    # Noisy SkyCoord with gaussian-convolved values.
    noisy: dict[str, u.Quantity] = {}
    for n, unit in cast("u.StructuredUnit", cast("u.Quantity", usc.data)._units).items():  # noqa: SLF001
        mean = getattr(usc.data, n).to_value(unit)
        scale = cast("NDArray[floating[Any]]", sig[n].to_value(unit))
        noisy[n] = u.Quantity(rnd.normal(mean, scale=scale), unit=unit)

    nc = SkyCoord(usc.frame.realize_frame(CartesianRepresentation(**noisy))).transform_to(frame)
    nc.representation_type = representation_type

    return SkyCoord(nc)


##############################################################################
# SCRIPT
##############################################################################

# Load ordered mock stream
ordered = make_ordered_orbit_data()
origin = ordered[len(ordered) // 2]

# Make noisy and shuffled stream
orb_obs = make_noisy_orbit_data(rnd=0)
data = QTable()
data["coords"] = orb_obs.galactocentric
data["arm"] = "arm1"
data["x_err"] = Quantity(0, u.pc)
data["y_err"] = Quantity(0, u.pc)
data["z_err"] = Quantity(1, u.pc)
data = data.group_by("arm")
data.add_index("arm")

# Save results
af = asdf.AsdfFile()
af.tree["data"] = data
af.tree["origin"] = origin
af.write_to(paths.data / "solar_circle.asdf")

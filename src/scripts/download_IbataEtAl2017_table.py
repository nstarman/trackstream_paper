"""Configuration."""

import astropy.units as u
import paths
from astropy.coordinates import Angle
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1


catalog_list = Vizier.find_catalogs("J/ApJ/842/120")
catalogs = Vizier.get_catalogs(catalog_list.keys())
table = catalogs["J/ApJ/842/120/table2"]

# Munge
renames = {
    "RAJ2000": "ra",
    "DEJ2000": "dec",
    "HRV": "radial_velocity",
    "e_HRV": "rv_err",
}
table.rename_columns(tuple(renames.keys()), tuple(renames.values()))

table["ra"] = Angle(table["ra"], unit=u.hourangle)
table["dec"] = Angle(table["dec"], u.deg)

# Save
table.write(paths.data / "IbataEtAl2017.ecsv", overwrite=True)

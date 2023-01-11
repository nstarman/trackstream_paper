"""Download data from Vasiliev (2019) and save as an ECSV."""

##############################################################################
# IMPORTS

# THIRD-PARTY
from astroquery.vizier import Vizier

# LOCAL
import paths

##############################################################################
# PARAMETERS

Vizier.ROW_LIMIT = -1

##############################################################################
# SCRIPT
##############################################################################

catalog_list = Vizier.find_catalogs("J/MNRAS/484/2832")
catalogs = Vizier.get_catalogs(catalog_list.keys())
table = catalogs["J/MNRAS/484/2832/catalog"]

table.write(paths.data / "Vasiliev2019.ecsv", overwrite=True)

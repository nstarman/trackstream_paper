"""Download data from Starkman et al (2019) and save as an ECSV."""

import paths
from astropy.table import QTable

##############################################################################
# SCRIPT
##############################################################################

# I like the format of the table in this paper, so I'm going to download it
# over uploading my own.
table = QTable.read(
    "https://raw.githubusercontent.com/cmateu/galstreams/db32e10b4b8a00c93bb8bc83e09f5f4e1f992654/galstreams/tracks/track.st.Pal5.starkman2020.ecsv",
)

table.write(paths.data / "StarkmanEtAl19.ecsv", overwrite=True)

# FITS Star Finder & Catalog Cross-Match Tool

🪐 A command-line pipeline for automatic detection of stars in FITS images, WCS coordinate transformation, and object lookup in major astronomical catalogs (Gaia DR3, VSX, SIMBAD, Pan-STARRS, Hipparcos).

## 📦 Features

- ✅ Validates presence and quality of WCS headers in FITS files
- 🌟 Detects stars using `DAOStarFinder` from `photutils`
- 🔭 Converts pixel coordinates to sky coordinates (RA/Dec) using `astropy.wcs`
- 🔍 Queries the following catalogs for each detected star:
  - Gaia DR3
  - VSX (AAVSO Variable Star Index)
  - SIMBAD
  - Pan-STARRS DR2
  - Hipparcos
- 📈 Adds catalog matches to a result table and saves to CSV
- 🧪 Sigma clipping and aperture photometry with error estimation
- 🖥️ Simple CLI interface with parameter customization
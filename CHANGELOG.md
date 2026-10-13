# Changelog

All notable changes to gtellcorr are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.9.1] - 2026-10-13

### Changed

- The result columns of the wavelength-solution correction are renamed:
  `wave_shift`, `wave_shift_err` and `wave_shift_value_source` in
  `ORDER_SUMMARY` become `wavesol_coeffs`, `wavesol_coeffs_err` and
  `wavesol_source`; the `summary.csv` column `wave_shift` becomes
  `wavesol_coeffs`, and `load_exposure_order` returns the same keys. Result files
  written by 0.9.0 are still read.
- "Contract" becomes "LBLRTM setup": `gtellcorr.telluric.resources.setup` with
  `load_setup`, `check_setup` and `physics_fields`;
  `VerifiedResources.setup_id`; `lblrtm_setup` in the run metadata of a result
  file. The 0.9.0 names remain as aliases until 1.0.
- Documentation, messages and comments describe the wavelength correction as a
  correction of the wavelength solution, and use plain words instead of
  internal terms.

### Fixed

- Continuous integration on macOS (`pymsis` publishes macOS wheels for macOS 15
  and later) and with newer numpy type stubs on Python 3.12. The README says
  what macOS 14 needs to install `pymsis`.

## [0.9.0] - 2026-10-12

First public release, a preview of 1.0. It reproduces the results of the
accompanying paper (Luo et al., submitted to *Universe*).

### Added

- **Atmospheric profiles** for each exposure: the lower atmosphere from ERA5,
  MERRA-2, NCEP/NCAR R1, GFS or ESO GDAS, or a user-supplied profile; the upper
  atmosphere from NRLMSISE-00 (with CelesTrak space weather) or the MIPAS
  mid-latitude night profile; optional merging of the on-site weather recorded
  in the headers.
- **Line-by-line transmission** with LBLRTM v12.17 (double precision), LNFL v3.2,
  the AER v3.9 line file and MT_CKD 4.3. Optical depths are cached per order and
  molecule, so a fit does not rerun LBLRTM.
- **Fitting** of the molecular columns, the resolving power, the line-spread
  function (Gaussian, super-Gaussian or pseudo-Voigt), a continuum polynomial
  and, optionally, a correction of the wavelength solution on the telluric
  lines, order by order or jointly over all orders of an exposure.
- **Stellar lines** handled with a stellar-line mask (from the master spectrum
  or a template) or with a stellar template in the model (synthesised with
  pySME or iSpec, or supplied by the user); radial velocities from a
  cross-correlation, or from a user file.
- **Strong telluric mask**, activity and user windows, and an optional PCA of
  the fitted telluric models of a run.
- **Instruments**: built-in definitions for WIYN/NEID, LBT/PEPSI and
  VLT/ESPRESSO; other instruments are described in a TOML file.
- **Command line**: `gtellcorr configure`, `inspect`, `run` and `resources`
  (`status`, `verify`, `build`, `freeze`, `fetch`).
- **Results**: one FITS file per exposure, a run summary table, figures, and a
  Python API to read them (`gtellcorr.load_result`).

### Known limitations

- The LBLRTM bundle is built and tested on macOS. The Linux build is
  implemented but has not yet been verified on a Linux host.
- A synthesised stellar template is reused as long as its file exists; so are
  the RV table (if it covers all exposures) and the master spectrum. After
  changing the parameters they depend on, delete these files (`--reuse` applies
  to the exposure results only).
- No worked example with data is included yet; the spectra used in the paper
  will be published with it.

[0.9.1]: https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.9.1
[0.9.0]: https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.9.0

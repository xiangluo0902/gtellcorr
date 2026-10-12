# Changelog

All notable changes to gtellcorr are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.9.0] - 2026-10-13

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
  and, optionally, a wavelength offset, order by order or jointly over all
  orders of an exposure.
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

[0.9.0]: https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.9.0

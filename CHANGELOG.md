# Changelog

All notable changes to gtellcorr are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.10.0] - 2026-10-17

### Changed

- The orders of an exposure are fitted jointly by default (`relcol_mode =
  "joint_orders"`), as in the paper. Order-by-order fitting (`single_order`)
  remains available. Run files that set `relcol_mode` are not affected.
- The LBLRTM bundle build is now verified on Linux as well as macOS.
- The continuum of the telluric fit is no longer sigma-clipped by default
  (`continuum.niter = 1`, was 5). Clipping makes the fit objective
  discontinuous; where the stellar model fits poorly (e.g. M dwarfs) the fit
  then stopped away from the minimum and depended on the starting values. Run
  files that set `continuum.niter` are not affected.
- The optical-depth cache includes the LBLRTM continua by default
  (`telluric.continuum_mode = "full"`): the H2O / O2 / N2 continua and
  collision-induced absorption, Rayleigh extinction and the O3 cross sections,
  computed at the reference and fixed columns (two more LBLRTM runs per cache)
  and held fixed while the fitted columns are scaled. Without them the O2
  collision-induced absorption under the A band was fitted as extra O2.
  `continuum_mode = "none"` gives the previous line-only cache; caches built
  before this change are read as `none`. Run files imported from the paper
  version set `none`.
- Stellar RV measurements mask telluric lines with the predicted line transmission
  by default (`rv.telluric_mask = "model"`, threshold 0.99, padded by 2 km/s).
  `"bands"` retains the previous band windows. The CCF velocity grid is widened
  on each side by `rv.ccf_vsini_pad * vsini` (default 2 times the target's vsini);
  set the padding to 0 to keep the configured fixed grid. Imported paper
  configurations retain band masks and the fixed grid.
- Measured RV tables record the template, mask windows, velocity grid and selected
  orders beside the table, and are measured again when these records change.
  Tables without a record are reused only with band masks and a fixed CCF grid.
- Files are written with `schema_version = 5`. Instrument files lose
  `spectral.airmass_card`: the exposure airmass has always come from
  `header.airmass` (the first of its cards present), and the reader's single card,
  which named the wrong telescope for three of the four VLT units, was not used.
  Files with schema 4 or older load as before (the card is dropped). The fit
  configuration hash `CFG_HASH` includes the schema version, so results of earlier
  versions are fitted again once under `reuse = "matching"`; optical-depth caches,
  strong-telluric masks and stellar products are reused. `OrderSpectrum` and
  `RawOrder` no longer carry `airmass` (use `Exposure.airmass`).
- Result files (`S2DIMGV3`) hold each order on its delivered pixel grid: pixel `i`
  of an image row is pixel `i` of the order in the input file. Before, a row held
  the order's usable pixels from its start, so every pixel after one without
  usable flux, error or wavelength (e.g. NaN pixels of NEID) moved. The
  wavelengths, `FLUX_OBS` and `ERR` hold every delivered pixel; the telluric model,
  corrected flux, continuum and stellar model are NaN where the pixel has no usable
  data. `EXCLUSION` is 0 for pixels used (in the fit of the molecular columns in
  the fit orders, `ORDER_SUMMARY.used_in_fit`; in the other orders in the fit of
  that order's continuum only), 1 for masked pixels (strong telluric cores,
  activity lines, user bad regions, and stellar lines with `stellar = "mask"`;
  they are corrected) and 2 for pixels without usable data in the input and
  beyond the order (was 1; not corrected). COMMENT cards of the `EXCLUSION` header
  explain the values. `ORDER_SUMMARY` appends
  `n_input`, the pixels of the delivered order (`n_pixels` remains the usable
  ones). `load_exposure_order` returns the whole order and `kept`, the pixels with
  data; `ExposureResult.no_data` marks the others. The fitted values do not change.
  Files written by 0.9 are read as before. Readers record the delivered order in
  `OrderSpectrum.grid` / `RawOrder.grid` (`InputGrid`).

### Added

- Example data, attached to the release as `gtellcorr-example-pepsi.zip`: one PEPSI
  spectrum of TOI-2076 with the ERA5 data of its atmospheric profile, a pySME stellar
  template and a run file; it runs without a weather-archive account.
- `examples/paper/`: five notebooks, with script versions, that reproduce the
  corrections of the paper (PEPSI, NEID, ESPRESSO), published with their outputs.
- The gtellcorr logo (`docs/logo/`), shown in the README and the manual.
- `telluric.continuum_mode = "scaled"`: the LBLRTM continua follow the fitted
  columns instead of staying at the reference columns. Four more LBLRTM runs per
  fitted molecule (with and without continua, at zero and twice the reference
  column) give a quadratic approximation in each fitted column (the H2O self
  continuum and the O2 collision-induced absorption have quadratic contributions);
  cross terms between fitted molecules are left out. At
  the reference columns it gives the `full` cache. `full` remains the default.
  The strong-telluric mask of a `scaled` run uses a `full` cache (the mask
  needs the reference columns only): the same windows as a `full` run, and a
  quarter of the LBLRTM runs for its twelve-molecule caches.
- With `continuum_mode = "full"` or `"scaled"` the stored telluric
  transmission `T` holds the line absorption only: the broadband transmission of
  the continua is moved into the stored continuum `C`, so `C * T` is unchanged.
- `plot_order_fit` takes `flux_ylim` to set the range of the flux panel
  (default `(0, 1.25)` as before).
- `continuum.output_niter` (default 5): the continuum stored with the result
  (`CONTINUUM`) is fitted once more after the fit, sigma-clipped against the
  fitted model, so unmasked stellar lines and outliers do not pull it down. It
  normalises the stored spectra for display and quality metrics only; the
  corrected flux F / T and the fitted parameters do not use it. Equal to
  `continuum.niter`, the fit's own continuum is stored as before.
- The LBLRTM runs of an optical-depth cache are made several at a time (setting
  `lblrtm_workers`, environment variable `GTELLCORR_LBLRTM_WORKERS`). The default
  `"auto"` takes the CPUs the process may use (affinity mask and cgroup CPU limit
  on Linux), at most 8. When running several Python processes, set the worker
  count per process to share the available CPUs and temporary disk space. The
  caches are the same as when built one run at a time, and keep their keys. Each
  run needs about 1 GB in TMPDIR while it runs.
- Experimental `masks.stellar_linemask.blend` (default `"off"`, as before): lines
  measured on the master are checked against the run's telluric model; lines taken
  for telluric features are not masked, and blends of a stellar and a telluric line
  are masked (`"mask"`), kept (`"keep"`) or masked by their stellar / telluric
  equivalent width (`"ratio"`, `blend_ratio`). The line catalogue records each line's
  class. In the injection-recovery simulation every setting fitted H2O worse than
  `"off"` (G and K stars: +4.5 to +12 % against +0.9 to +3.8 %), because the model
  column is that of the atmospheric profile and weak stellar lines left on telluric
  lines bias the fit; the user manual ("Two strategies for stellar lines") describes
  the test. The run's telluric model is computed once and shared with the RV
  telluric windows (`telluric_mask.run_line_transmission`).
- `stellar_model.line_elements = []` takes every element of the iSpec line list.
- The RV figures (`plot_ccf_diagnostics`, `plot_rv_timeseries`,
  `plot_rv_master_vs_perorder`) and `plot_profiles` accept `out=False` (no file),
  as the other figures of `gtellcorr.plotting` do.
- `plotting.plot_profiles`: temperature, pressure and water vapour of one or more
  atmospheric profiles against altitude, with the observatory altitude, the on-site
  measurements and a zoom on the lower atmosphere.
- `[data] instrument_file`: a run file can name a TOML file of your own instrument
  definition instead of a registered `instrument`, so `gtellcorr run` reads
  spectrographs without a built-in definition; `gtellcorr inspect --instrument-file`
  does the same.
- Optional stellar-template velocity fitting (`stellar_model.fit_shift = true`),
  disabled by default: a velocity beyond the measured placement RV, shared by
  the orders of a joint fit and bounded by `shift_bounds_kms` (default +/-2 km/s).
  `ORDER_SUMMARY` appends `stellar_shift` and `stellar_shift_err` in km/s;
  `summary.csv` appends `stellar_shift`. When disabled the fields are NaN in
  FITS and empty in CSV. Older result files remain readable; the image HDUs and
  `S2DIMGV2` format identifier are unchanged.

### Fixed

- `gtellcorr resources build` patches the water-continuum module of LBLRTM v12.17
  (MT_CKD 4.3, `mt_ckd_h2o_module.f90`). Its search of the coefficient table, which
  ends at 20000 cm-1 (500 nm), had no bound: an LBLRTM run whose range crosses
  500 nm (with the 20 nm margin of a run, ranges between about 480 and 520 nm)
  read past the end of the table. The process then died with SIGBUS ("Access to an
  undefined portion of a memory object"): in a test, 28 of 40 cache builds of 2 nm
  ranges between 480 and 517 nm failed, so orders near 500 nm failed at random.
  When it did not die, the continuum near 500 nm used values from beyond the table
  (up to 6e-6 in optical depth from run to run). The patch stops the search at the
  last coefficient, as the MT_CKD 3.5 code of LBLRTM v12.11 did: the same 40 builds
  all succeed and are reproducible, and differ from the unpatched ones by 1.1e-6 in
  optical depth. Ranges whose run does not cross 500 nm (tested at 430-760 nm) give
  the same optical depths as before, byte for byte.
  The patch is recorded in the bundle's build recipe. A bundle built before this
  change still works but gives a warning; rebuild it with `gtellcorr resources
  build` (the new bundle has a new fingerprint, so its optical-depth caches are
  built again).
- VLT/ESPRESSO pixels flagged by the pipeline are left out. The S2D product is a
  fixed grid of 9111 pixels per order; beyond the extracted order (both ends of
  the blue orders, the red end of the last two orders at 788-790 nm) the flux is 0
  with a positive error and `QUALDATA` flags the pixel. These pixels entered the
  fit and the continuum of the orders that hold them; in the last two orders the
  continuum was off by up to a factor 2.5 and a fit including them gave H2O 1.2 %
  too high. A spectral setting `quality` names the flag HDU of an echelle
  instrument; pixels with a non-zero flag are treated as pixels without usable
  data (and are `EXCLUSION` = 2 in the results). The built-in ESPRESSO definition
  sets `quality = "QUALDATA"`. This also leaves out isolated bad pixels (two in
  the default fit orders of the GJ 436 spectrum of the paper).
- VLT/ESPRESSO spectra downloaded from the ESO archive (`ES_S2DA_*.fits`, the
  same S2D product of fibre A as the pipeline's `r.ESPRE.*_S2D_A.fits`) are
  recognised. A channel of an instrument file may now list several file-name
  globs (`all = ["r.ESPRE.*_S2D_A.fits", "ES_S2DA_*.fits"]`); a single string
  still works. ESPRESSO files are also checked for `ESO PRO CATG = S2D_A`.
  `InstrumentSpec.channel_glob` is replaced by `channel_patterns`, which returns
  all globs of the channel.
- A pySME template synthesis no longer prints pySME's message for every line-list file
  it reads (with its full path); pySME's warnings are still shown.
- Two selected files with the same exposure epoch (for example a pipeline file and
  its archive copy in one directory) stop the run with an error naming both,
  instead of the exposure being processed twice.
- The MERRA-2 backend (`merra2_gesdisc`) works again. GES DISC retired its own
  OPeNDAP server, which now answers HTTP 410; profiles are read from Earthdata
  Cloud OPeNDAP instead, with the same Earthdata token.
- A lower-atmosphere profile that starts more than one grid step (1 km) above sea
  level, such as MERRA-2 at a mountain site, no longer fails with "P_hPa must
  strictly decrease": below its lowest level the pressure is extrapolated with
  the scale height of the two lowest levels instead of repeating the lowest value.
  This also changes the pressure at 0 km of other profiles (e.g. ERA5); those
  levels lie below the site and do not enter the radiative transfer, but newly
  built full profiles have a different content hash. Cached profiles are reused
  as they are.
- `dumps_config(config, comments=False)` (and `save` with `comments=False`) no
  longer starts with "Each parameter is documented above its line": without the
  descriptions, the header only says that parameters left out keep their defaults.
  The configuration hash, and hence the reuse of results, is unchanged.
- The `EXPOSURE` and `INSTRUME` cards of a result file keep their comment only when
  it fits in the 80-character card. A long input file name, such as an ESO archive
  name (`ES_S2DA_..._HR_2x1_U3.fits`), made astropy truncate the comment with a
  `VerifyWarning`; the value itself was always complete.
- Result files and `run_manifest.json` name no local directory. `RUN_METADATA` gives
  `run_root` relative to the output directory (`<target>/<site>_<instrument>/<date>`)
  and `rv_table` relative to the run directory, the manifest gives `summary_path`
  relative to the run directory, and `lblrtm_bundle` and `tau_cache` are given by
  their directory name (the bundle is identified by `lblrtm_local_fingerprint`).
- A synthesised stellar template (pySME or iSpec), the strong-telluric mask and the
  telluric windows of the RV measurement use the start R of the fit:
  `lsf.resolution` if set, else the exposures' header R (the PEPSI fibre), else the
  instrument default (`start_resolution` in `gtellcorr.data.observation.instrument`).
  The two masks used the header R of the first exposure, ignoring `lsf.resolution`.
  The template was always broadened to the instrument default, so
  PEPSI exposures taken with the 300 um fibre (R 50 000) were fitted with a template
  at R 130 000, and fits that set `lsf.resolution` (such as the injection-recovery
  simulation at R 10 000, 50 000 and 250 000) with one at the default R. The R is part
  of the template's file name, so a template at a new R is synthesised once. A
  `template_file` with `template_resolution` is broadened to the same R (it ignored
  `lsf.resolution` before). Without `lsf.resolution`, exposures with different header
  R in one run stop with an error. Templates and masks of runs whose header R equals
  the instrument default (all paper data) are unchanged.

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

[0.10.0]: https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.10.0
[0.9.1]: https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.9.1
[0.9.0]: https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.9.0

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/gtellcorr-logo-dark.svg">
    <img src="docs/logo/gtellcorr-logo.svg" alt="gtellcorr" width="420">
  </picture>
</p>

# gtellcorr

Clean the Earth's atmosphere out of your high-resolution spectra. gtellcorr rebuilds
the atmosphere above your telescope for every exposure, models its absorption with
LBLRTM, and fits it alongside your star's own lines.

In more detail: the atmospheric profile of each exposure comes from meteorological
reanalyses, LBLRTM turns it into a line-by-line transmission, and the molecular
columns, the instrumental resolution and the continuum are fitted to the observed
spectrum. Stellar lines are either masked or modelled with a stellar template. You
get back the corrected spectrum together with the telluric model that was removed.

gtellcorr currently comes with definitions for **WIYN/NEID**, **LBT/PEPSI** and
**VLT/ESPRESSO**; other echelle spectrographs can be described in a TOML file, and
more instruments will follow.

> Version 0.10.1 is a public preview of 1.0. It includes changes to the scientific
> defaults since 0.9.1. See [Upgrading from 0.9.1](#upgrading-from-091) and
> [Known limitations](#known-limitations) before using it. The accompanying paper
> is Luo et al., submitted to *Universe*.

## Installation

gtellcorr needs Python 3.11 or later, and runs on macOS and Linux. On macOS 14 or
earlier, pip has to compile the `pymsis` dependency, which needs a Fortran compiler
(for example `conda install -c conda-forge gfortran`).

```bash
pip install "gtellcorr @ git+https://github.com/xiangluo0902/gtellcorr@v0.10.1"
```

Optional extras:

```bash
pip install "gtellcorr[era5] @ git+https://github.com/xiangluo0902/gtellcorr@v0.10.1"   # ERA5 profiles
pip install "gtellcorr[pysme] @ git+https://github.com/xiangluo0902/gtellcorr@v0.10.1"  # pySME stellar templates
```

`environment.yml` pins the exact library versions used for the paper:

```bash
git clone https://github.com/xiangluo0902/gtellcorr && cd gtellcorr
conda env create -f environment.yml && conda activate gtellcorr
pip install -e .
```

## Setting up the external resources

gtellcorr needs three things it does not ship in the Python package: an LBLRTM
bundle, two reference files for the atmosphere, and (for most reanalyses)
credentials for the data archive. `gtellcorr resources status` shows what is
missing.

### 1. The LBLRTM bundle

The bundle holds LBLRTM v12.17, LNFL v3.2, the line database built from the AER v3.9
line file, and MT_CKD 4.3 (about 1.6 GB). It is compiled on your machine from three
source archives, which are attached to the
[v0.9.0 release](https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.9.0):

| File | Size |
|---|---|
| `aerlbl_v12.17.tar` | 122 MB |
| `aerlnfl_v3.2.tar` | 2 MB |
| `aer_v_3.9.tgz` | 406 MB |

They are redistributed under the AER licences (scientific and research use only;
see [Acknowledgements](#acknowledgements)). gtellcorr checks their SHA-256 before
building.

```bash
gtellcorr resources build \
    --lblrtm-archive aerlbl_v12.17.tar \
    --lnfl-archive aerlnfl_v3.2.tar \
    --line-file-archive aer_v_3.9.tgz
```

The build needs `conda` or `mamba`: it creates a separate conda-forge environment
with gfortran and NetCDF-Fortran under the resource directory. To use an existing
NetCDF-Fortran and gfortran installation instead, pass `--netcdf-prefix`. The
bundle is checked once it is built; `gtellcorr resources verify` repeats the
check at any time. The build has been tested on macOS and Linux.

A bundle copied from another computer needs its file sizes and checksums recorded
once with `gtellcorr resources freeze --bundle PATH`.

### 2. Reference files

```bash
gtellcorr resources fetch
```

This downloads the MIPAS mid-latitude night profile (molecules other than H2O,
and the upper atmosphere if MSIS is not used) and the CelesTrak space-weather table
(inputs of the NRLMSISE-00 upper atmosphere). Fetch the CelesTrak table again when
your observations are later than its last day.

### 3. Atmosphere archive credentials

ERA5 (the default) and MERRA-2 need an account. Write the keys into a credentials
file, as one line (the file is read line by line):

```text
WEATHER_KEYS = {"era5_cdsapi": {"CDSAPI_URL": "https://cds.climate.copernicus.eu/api", "CDSAPI_KEY": "<your key>"}, "merra2_gesdisc": {"EARTHDATA_TOKEN": "<your token>"}}
```

and point gtellcorr to it with `credentials_file` in the settings or with the
`GTELLCORR_CREDENTIALS` environment variable. For ERA5, register at the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) and accept the
ERA5 licence once. NCEP/NCAR R1, GFS and ESO GDAS need no account.

### Settings

Machine-specific paths go in `~/.config/gtellcorr/settings.toml` (another location
can be given with `GTELLCORR_SETTINGS`), or in environment variables:

| Setting | Environment variable | Default |
|---|---|---|
| `resource_root` | `GTELLCORR_RESOURCES` | the platform's user data folder |
| `cache_dir` | `GTELLCORR_CACHE` | the platform's user cache folder |
| `lblrtm_bundle` | `GTELLCORR_LBLRTM_BUNDLE` | `<resource_root>/lblrtm/gnu_double` |
| `tau_cache` | `GTELLCORR_TAU_CACHE` | `<cache_dir>/lblrtm_tau_cache` |
| `atmosphere_cache` | `GTELLCORR_ATMOSPHERE_CACHE` | `<cache_dir>/atmos_profiles` |
| `credentials_file` | `GTELLCORR_CREDENTIALS` | none |
| `stellar_linelist` | `GTELLCORR_STELLAR_LINELIST` | none (VALD list for pySME) |
| `ispec_dir` | `GTELLCORR_ISPEC_DIR` | none (iSpec installation) |
| `lblrtm_timeout_s` | `GTELLCORR_LBLRTM_TIMEOUT` | 600 |
| `lblrtm_workers` | `GTELLCORR_LBLRTM_WORKERS` | `"auto"` (LBLRTM runs of a cache at the same time: the CPUs available, at most 8) |

iSpec is needed for stellar-line masks measured on the master spectrum, for the
iSpec continuum of the master spectrum, and for the iSpec synthesis backend.
Stellar templates synthesised with pySME, or supplied by you, do not need it.
iSpec is imported into gtellcorr's Python environment and needs `statsmodels` and
`lockfile` there (both are in `environment.yml`).

## Usage

### Command line

```bash
gtellcorr inspect /data/NEID                       # identify the instrument, list the exposures
gtellcorr configure -o run.toml --directory /data/NEID --instrument WIYN/NEID
gtellcorr run run.toml -o out/
```

`configure` writes a run file with every parameter and its documentation; edit
the `[data]` table (target star, exposures to include or exclude) and the `[fit]`
table (molecules, stellar treatment, line-spread function, masks) before running.

### Python

```python
import gtellcorr as gtc

star = gtc.Star(name="TOI-2076", teff=5180, logg=4.608, feh=0.0, vsini_kms=5.0, vsys_kms=-12.5)
obs = gtc.Observation.from_directory("/data/NEID", instrument="WIYN/NEID", target=star)
config = gtc.FitConfig(stellar="template", relcol_mode="joint_orders")

result = gtc.run(obs, config, output_dir="out/")
result.summary  # fitted parameters per exposure and order (a table)
exposure = result.exposures[0]  # one exposure: (order, pixel) arrays
exposure.wave, exposure.flux_corrected, exposure.telluric
```

### Output

A run writes to `<output>/<target>/<site>_<instrument>/<night>/`:

- `telluric_corr/data/`: one FITS file per exposure (observed and corrected flux,
  telluric model, continuum, stellar model, fit mask, per-order parameters) and
  `summary.csv`;
- `telluric_corr/figure/`: per-order figures;
- `Radial_Velocities/`, `template_stellar_spectrum/`, `mask_spectral_line/`: the
  RV table, stellar template or master spectrum, and the masks used in the fit;
- `run_config.toml` and `run_manifest.json`: the configuration and inputs of the run.

`gtc.load_result(path)` reads a finished run.

### Example data and notebooks

The [v0.10.1 release](https://github.com/xiangluo0902/gtellcorr/releases/tag/v0.10.1)
has an example data set, `gtellcorr-example-pepsi.zip`: one PEPSI spectrum of the K
dwarf TOI-2076 (LBT, 2025 June 19), the ERA5 data of its atmospheric profile and a pySME
stellar template, with a run file. Unzip it and, from its folder:

```bash
GTELLCORR_ATMOSPHERE_CACHE=atmos_profiles gtellcorr run run.toml -o out
```

No weather-archive account is needed, since the profile data are included; the LBLRTM
bundle and the reference files must be set up as above.

`examples/paper/` holds the notebooks (with script versions) that reproduce the
corrections of the paper: a night of PEPSI spectra of TOI-2076, the mask and template
treatments of the stellar lines, NEID spectra of WASP-33, ESPRESSO spectra of GJ 436
with a PHOENIX template, and three meteorological sources for HAT-P-26. They are
published with their outputs; apart from the example spectrum and the ESO and NEID
archive files they name, their data are not public yet.

### Scientific defaults and optional parameters

Orders of each exposure share their fitted molecular columns by default
(`relcol_mode = "joint_orders"`); `"single_order"` fits them independently.
The optical-depth cache includes the LBLRTM continua at the reference columns
(`telluric.continuum_mode = "full"`). `"none"` omits them; `"scaled"` is an
optional quadratic approximation that follows the fitted columns and omits cross
terms between fitted molecules. It has been checked against direct calculations
under the O2 A band; that check does not establish its accuracy over every band
or atmospheric condition.

The stored `TELL_MODEL` holds the line absorption. With `"full"` or `"scaled"`,
the broadband part is moved into `CONTINUUM`, keeping their product unchanged.
`FLUX_CORRECTED` and `ERR_CORRECTED` are the observed flux and its uncertainty
divided by `TELL_MODEL`; the uncertainty does not include fitted-model errors.

The fit solves the continuum once at each parameter evaluation
(`continuum.niter = 1`). The saved continuum is then refitted with clipping
(`continuum.output_niter = 5`); that final refit does not change the fitted
parameters or corrected flux.

Stellar RVs are measured while masking predicted telluric lines
(`rv.telluric_mask = "model"`, line transmission below 0.99, padded by 2 km/s).
The CCF grid extends beyond `rv.ccf_v_lo` / `rv.ccf_v_hi` by
`rv.ccf_vsini_pad * vsini` on each side (default factor 2); a factor of 0 keeps
the fixed grid. These RVs place the stellar template and masks. A separate
template velocity can be fitted with `stellar_model.fit_shift = true` (default
false); it does not move the telluric model or replace the wavelength-solution
correction. It is shared across orders in a joint fit.

### Upgrading from 0.9.1

Existing TOML files remain readable, but omitted fields take the new defaults.
To retain the earlier default choices, set these fields explicitly (keep your
existing `relcol_mode` and `continuum.niter` if you already chose them):

```toml
[fit]
relcol_mode = "single_order"

[fit.telluric]
continuum_mode = "none"

[fit.continuum]
niter = 5
output_niter = 5

[fit.rv]
telluric_mask = "bands"
ccf_vsini_pad = 0.0

[fit.stellar_model]
fit_shift = false
```

These settings preserve the earlier choices, not a guarantee of identical
results after bug fixes. Existing atmospheric-profile caches are reused by the
pressure-extrapolation fix; use a separate `atmosphere_cache` directory when
evaluating freshly constructed profiles. The optical-depth cache distinguishes
`none`, `full` and `scaled`; old line-only caches are read as `none`, and a cache
with a different physical identity is not substituted for the requested one.

New FITS files append `stellar_shift` and `stellar_shift_err` to `ORDER_SUMMARY`
(km/s, NaN when disabled); `summary.csv` appends `stellar_shift` (empty when
disabled). The image layout and `S2DIMGV2` identifier are unchanged. The reader
accepts older files without these columns; scripts requiring an exact column
list must allow the additions.

## Known limitations

- A synthesised stellar template and the master spectrum are reused while their
  files exist. After changing inputs they depend on, remove the affected products
  (and the corresponding template chunk directory) before rebuilding. Measured RV
  tables record the template, mask windows, CCF grid and orders, and are remeasured
  when that record changes; this is not a complete check of every upstream input.
  `--reuse` applies to exposure results only; those are refitted when the fit
  configuration changes.
- Broad-lined A-star CCFs can return an incorrect RV even with the wider grid.
  Inspect the CCF or supply RVs with `rv.source = "user_file"`. Template velocity
  fitting has not improved telluric recovery in the tested simulation grid and
  can hit its bounds at low resolution or for broad-lined stars; it is off by
  default. Stellar-template mismatch can still bias the fitted columns and
  wavelength-solution correction.

## Citation

If you use gtellcorr, please cite the paper (Luo et al., submitted to *Universe*;
see [CITATION.cff](CITATION.cff)) and the version you used.

## Acknowledgements

gtellcorr uses LBLRTM, LNFL, MT_CKD and the AER line file by Atmospheric and
Environmental Research, Inc. (AER), redistributed under their licences
(`src/gtellcorr/telluric/lblrtm/third_party_notices/`); the line file is based on
HITRAN2020 (Gordon et al. 2022, JQSRT 277, 107949). Upper-atmosphere profiles use
NRLMSISE-00 through pymsis, with space-weather indices from CelesTrak, and the MIPAS
reference atmospheres distributed with RFM (University of Oxford). The example data
contain ERA5 data (Hersbach et al. 2020, Copernicus Climate Change Service, CC-BY 4.0)
and a spectrum taken with PEPSI at the Large Binocular Telescope.

## License

MIT (see [LICENSE](LICENSE)). The AER software and data keep their own licences.

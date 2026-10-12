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

> Version 0.9.0 is a public preview of 1.0. It reproduces the results of the
> accompanying paper (Luo et al., submitted to *Universe*). See
> [Known limitations](#known-limitations) before using it.

## Installation

gtellcorr needs Python 3.11 or later, and runs on macOS and Linux.

```bash
pip install "gtellcorr @ git+https://github.com/xiangluo0902/gtellcorr@v0.9.0"
```

Optional extras:

```bash
pip install "gtellcorr[era5] @ git+https://github.com/xiangluo0902/gtellcorr@v0.9.0"   # ERA5 profiles
pip install "gtellcorr[pysme] @ git+https://github.com/xiangluo0902/gtellcorr@v0.9.0"  # pySME stellar templates
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
check at any time.

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

iSpec is needed for stellar-line masks measured on the master spectrum, for the
iSpec continuum of the master spectrum, and for the iSpec synthesis backend.
Stellar templates synthesised with pySME, or supplied by you, do not need it.

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

## Known limitations

- The LBLRTM bundle has been built and tested on macOS. The Linux build is
  implemented but not yet verified on a Linux host.
- A synthesised stellar template is reused while its file exists; so are the RV
  table (if it covers every exposure) and the master spectrum. After changing the
  parameters they depend on, delete them from the run directory. (`--reuse`
  applies to the exposure results only; those are refitted when the fit
  configuration changes.)
- No example data set is included yet; the spectra used in the paper will be
  published with it.

## Citation

If you use gtellcorr, please cite the paper (Luo et al., submitted to *Universe*;
see [CITATION.cff](CITATION.cff)) and the version you used.

## Acknowledgements

gtellcorr uses LBLRTM, LNFL, MT_CKD and the AER line file by Atmospheric and
Environmental Research, Inc. (AER), redistributed under their licences
(`src/gtellcorr/telluric/lblrtm/third_party_notices/`); the line file is based on
HITRAN2020 (Gordon et al. 2022, JQSRT 277, 107949). Upper-atmosphere profiles use
NRLMSISE-00 through pymsis, with space-weather indices from CelesTrak, and the MIPAS
reference atmospheres distributed with RFM (University of Oxford).

## License

MIT (see [LICENSE](LICENSE)). The AER software and data keep their own licences.

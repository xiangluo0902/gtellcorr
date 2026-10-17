"""Synthetic spectrum files shaped like the built-in instruments' products."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits

N_PIX = 40


def _echelle(path: Path, primary: dict[str, Any], hdus: dict[str, np.ndarray]) -> Path:
    hdr = fits.Header()
    for key, value in primary.items():
        hdr[key] = value
    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
    for name, data in hdus.items():
        hdul.append(fits.ImageHDU(data=data, name=name))
    hdul.writeto(path, overwrite=True)
    return path


def neid_primary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "INSTRUME": "NEID",
        "OBJECT": "TIC 27491137",
        "DATE-OBS": "2022-02-13T07:30:05.553",
        "EXPTIME": 600.0,
        "AIRMASS": 1.9,
        "OBSLAT": 31.958092,
        "OBSLON": -111.600562,
        "OBSALT": 2091.0,
        "ENVOTMP": 13.4,
        "ENVOHUM": 3.2,
        "DEWPOINT": -25.5,
        "ENVTIME": "2022-02-13T07:30:09",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def neid_arrays(n_orders: int = 3) -> dict[str, np.ndarray]:
    wave = np.array([np.linspace(6000 + 100 * o, 6050 + 100 * o, N_PIX) for o in range(n_orders)])
    flux = np.full((n_orders, N_PIX), 200.0)
    flux[0, 3] = np.nan  # unusable pixel
    flux[1, 5] = -4.0  # negative noise excursion: kept
    var = np.full((n_orders, N_PIX), 16.0)
    var[0, 7] = 0.0  # zero error: dropped
    blaze = np.full((n_orders, N_PIX), 2.0)
    wave[2, :] = np.nan  # an order without wavelengths
    return {"SCIFLUX": flux, "SCIVAR": var, "SCIWAVE": wave, "SCIBLAZE": blaze}


@pytest.fixture
def make_neid(tmp_path: Path) -> Callable[..., Path]:
    def make(
        name: str = "neidL2_20220213T073005.fits", folder: Path | None = None, **header: Any
    ) -> Path:
        target = (folder or tmp_path) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return _echelle(target, neid_primary(**header), neid_arrays())

    return make


@pytest.fixture
def make_espresso(tmp_path: Path) -> Callable[..., Path]:
    def make(name: str = "r.ESPRE.2022-07-02T23:50:42.400_S2D_A.fits", **header: Any) -> Path:
        primary: dict[str, Any] = {
            "INSTRUME": "ESPRESSO",
            "DATE-OBS": "2022-07-02T23:50:42.399",
            "EXPTIME": 700.0,
            "HIERARCH ESO TEL1 GEOLAT": -24.6276,
            "HIERARCH ESO TEL1 GEOLON": -70.4051,
            "HIERARCH ESO TEL1 GEOELEV": 2648.0,
            "HIERARCH ESO TEL1 AIRM START": 1.143,
            "HIERARCH ESO INS MODE": "SINGLEHR",
            "HIERARCH ESO QC BERV": 12.5,
        }
        primary.update(header)
        wave = np.array([np.linspace(5000, 5050, N_PIX), np.linspace(5100, 5150, N_PIX)])
        flux = np.full((2, N_PIX), 50.0)
        quality = np.zeros((2, N_PIX), np.uint16)
        flux[0, -5:], quality[0, -5:] = 0.0, 16384  # S2D grid beyond the extracted order
        quality[1, 7] = 256  # a bad pixel with ordinary flux
        hdus = {
            "SCIDATA": flux,
            "ERRDATA": np.full((2, N_PIX), 2.0),
            "QUALDATA": quality,
            "WAVEDATA_AIR_BARY": wave,
        }
        return _echelle(tmp_path / name, {k: v for k, v in primary.items() if v is not None}, hdus)

    return make


@pytest.fixture
def make_pepsi(tmp_path: Path) -> Callable[..., Path]:
    def make(name: str = "pepsir.20250101.001.dxt.nor", **header: Any) -> Path:
        hdr = fits.Header()
        primary: dict[str, Any] = {
            "INSTRUME": "PEPSI",
            "IMAGETYP": "object",
            "OBJECT": "TEST STAR",
            "DATE-OBS": "2025-01-01",
            "TIME-OBS": "03:00:00.000",
            "EXPTIME": 600.0,
            "AIRMASS": 1.10,
            "LATITUDE": "+32:42:04.71",
            "LONGITUD": "-07:19:33.38",
            "ALTITUDE": 3221.0,
            "FIBER": "200",
            "SSTVEL": -25000.0,
            "LBTT": 12.0,
            "LBTH": 20.0,
            "LBTP": 690.0,
        }
        primary.update(header)
        for key, value in primary.items():
            if value is not None:
                hdr[key] = value
        n = 60
        wave = np.linspace(6300.0, 6310.0, n)
        flux = np.full(n, 0.95)
        var = np.full(n, 1e-4)
        var[4] = 0.0  # dropped by var > 0
        mask = np.zeros(n, dtype=np.int16)
        mask[10] = 1  # flagged pixel
        cols = fits.ColDefs(
            [
                fits.Column(name="Arg", format="D", array=wave),
                fits.Column(name="Fun", format="D", array=flux),
                fits.Column(name="Var", format="D", array=var),
                fits.Column(name="Mask", format="I", array=mask),
            ]
        )
        fits.HDUList(
            [fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, name="DataVector")]
        ).writeto(tmp_path / name, overwrite=True)
        return tmp_path / name

    return make

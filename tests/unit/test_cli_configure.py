from __future__ import annotations

from pathlib import Path

import pytest

from gtellcorr.cli import main
from gtellcorr.config.run_config import RunConfig


def test_configure_from_defaults(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run.toml"
    assert (
        main(["configure", "-o", str(out), "--directory", "/data/x", "--instrument", "LBT/PEPSI"])
        == 0
    )
    config = RunConfig.load(out)
    assert config.data.directory == "/data/x" and config.data.instrument == "LBT/PEPSI"
    assert "# " in out.read_text()
    assert "wrote" in capsys.readouterr().out


def test_configure_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run.toml"
    out.write_text("keep")
    assert main(["configure", "-o", str(out), "--directory", "/d"]) == 1
    assert out.read_text() == "keep"
    assert "use --force" in capsys.readouterr().err
    assert main(["configure", "-o", str(out), "--directory", "/d", "--force", "--no-comments"]) == 0
    assert RunConfig.load(out).data.directory == "/d"


def test_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import numpy as np
    from astropy.io import fits

    hdr = fits.Header(
        {
            "INSTRUME": "NEID",
            "DATE-OBS": "2022-02-13T07:30:05",
            "EXPTIME": 600.0,
            "AIRMASS": 0.3,
            "OBSLAT": 31.9,
            "OBSLON": -111.6,
            "OBSALT": 2091.0,
            "OBJECT": "TOI-2076",
        }
    )
    data = np.ones((2, 5))
    fits.HDUList(
        [fits.PrimaryHDU(header=hdr)]
        + [fits.ImageHDU(data, name=n) for n in ("SCIFLUX", "SCIVAR", "SCIWAVE", "SCIBLAZE")]
    ).writeto(tmp_path / "neidL2_x.fits")
    assert main(["inspect", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "WIYN/NEID" in out and "neidL2_x.fits" in out and "2022-02-13 07:35:05" in out
    assert "1.5 is used" in out
    assert main(["inspect", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err


def test_resources_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GTELLCORR_SETTINGS", str(tmp_path / "none.toml"))
    monkeypatch.setenv("GTELLCORR_RESOURCES", str(tmp_path / "res"))
    monkeypatch.delenv("GTELLCORR_LBLRTM_BUNDLE", raising=False)
    assert main(["resources", "status"]) == 1
    out = capsys.readouterr().out
    assert "MISSING/INVALID" in out and "MIPAS:" in out
    assert main(["resources", "verify"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["resources", "freeze", "--bundle", str(tmp_path)]) == 1
    assert "cannot freeze" in capsys.readouterr().err

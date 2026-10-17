from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gtellcorr.config.fit_config import FitConfig
from gtellcorr.errors import DataError
from gtellcorr.preprocessing import prepare
from gtellcorr.preprocessing.config import FitMasks, StellarLineMask
from gtellcorr.preprocessing.prepare import StellarProducts
from gtellcorr.preprocessing.stellar_lines import judge_line
from gtellcorr.preprocessing.telluric_mask import LineTransmission
from gtellcorr.preprocessing.windows import C_KMS

WAVE = np.arange(600.0, 601.0, 0.001)


def _gauss(centre: float, depth: float, sigma: float = 0.01) -> np.ndarray:
    return 1.0 - depth * np.exp(-0.5 * ((WAVE - centre) / sigma) ** 2)


def _entry(centre: float, depth: float) -> dict[str, float]:
    return {
        "wave_obs_nm": centre, "depth": depth,
        "wave_base_nm": centre - 0.03, "wave_top_nm": centre + 0.03,
    }  # fmt: skip


def _judge(entry: dict[str, float], flux: np.ndarray, tell: np.ndarray, **kw: object) -> dict:
    options = {"min_depth": 0.1, "blend": "ratio", "blend_ratio": 0.3, "telluric_depth": 0.02}
    return judge_line(entry, WAVE, flux, tell, **(options | kw))  # type: ignore[arg-type]


def test_lines_are_judged_against_the_telluric_model() -> None:
    tell = _gauss(600.5, 0.5)
    # a stellar line where the model does not absorb: masked
    flux = _gauss(600.2, 0.4) * tell
    assert _judge(_entry(600.2, 0.4), flux, tell) | {"telluric_depth": 0} == {
        "telluric_depth": 0, "class": "clean", "masked": True,
    }  # fmt: skip
    # a telluric line taken for a stellar one: no stellar depth left, not masked
    out = _judge(_entry(600.5, 0.5), tell, tell)
    assert (out["class"], out["masked"]) == ("telluric", False)
    assert out["stellar_depth"] == pytest.approx(0.0, abs=1e-3)
    # a blend: stellar depth 0.3 under a telluric line of depth 0.5, same width
    flux = _gauss(600.5, 0.3) * tell
    out = _judge(_entry(600.5, 1 - 0.7 * 0.5), flux, tell)
    assert out["class"] == "blend" and out["masked"]
    assert out["stellar_depth"] == pytest.approx(0.3, abs=1e-3)
    assert out["ew_ratio"] == pytest.approx(0.6, rel=0.05)
    entry = _entry(600.5, 1 - 0.7 * 0.5)
    assert not _judge(entry, flux, tell, blend_ratio=1.0)["masked"]
    assert _judge(entry, flux, tell, blend="mask")["masked"]
    assert not _judge(entry, flux, tell, blend="keep")["masked"]


def test_saturated_model_core_is_not_a_stellar_line() -> None:
    tell = _gauss(600.5, 0.9)
    out = _judge(_entry(600.5, 0.95), _gauss(600.5, 0.5) * tell, tell)
    assert (out["class"], out["masked"]) == ("telluric", False)


def test_line_transmission_interpolates_and_scales_with_airmass() -> None:
    w = np.linspace(600.0, 601.0, 11)
    t = np.full(11, 0.81)
    lt = LineTransmission(chunks=((w, t),), airmass=2.0)
    np.testing.assert_allclose(lt.at([599.0, 600.55, 602.0]), [1.0, 0.81, 1.0])
    np.testing.assert_allclose(lt.at([600.5], airmass=1.0), [0.9])
    other = LineTransmission(chunks=((w, t),), airmass=1.5)
    assert lt.sha256() != other.sha256()
    assert lt.sha256() == LineTransmission(chunks=((w.copy(), t.copy()),), airmass=2.0).sha256()


def test_master_frame_model_follows_the_rvs() -> None:
    w = np.linspace(599.0, 602.0, 30001)
    t = 1.0 - 0.5 * np.exp(-0.5 * ((w - 600.5) / 0.01) ** 2)
    lt = LineTransmission(chunks=((w, t),), airmass=1.0)
    rv = 30.0
    at = prepare._master_telluric(lt, [{"rv_kms": str(rv), "airmass": "1.0"}])
    rest = np.array([600.5 / (1 + rv / C_KMS)])
    np.testing.assert_allclose(at(rest), [0.5], atol=1e-3)
    # no usable airmass in the table: the model's own
    no_airmass = prepare._master_telluric(lt, [{"rv_kms": "0.0", "airmass": ""}])
    np.testing.assert_allclose(no_airmass(np.array([600.5])), [0.5], atol=1e-3)


def test_master_lines_need_the_model_unless_off(tmp_path: Path) -> None:
    products = StellarProducts(
        layout=None, template_path=tmp_path / "t.dat", rv_table=tmp_path / "rv.csv",  # type: ignore[arg-type]
        rv_kms={}, activity_windows=[], bad_regions=[], status={},
    )  # fmt: skip
    cfg = FitConfig(stellar="mask", masks=FitMasks(stellar_linemask=StellarLineMask(blend="ratio")))
    with pytest.raises(DataError, match="needs the run's telluric model"):
        prepare._linemask_settings(cfg, products)
    lt = LineTransmission(chunks=((np.array([600.0, 601.0]), np.array([1.0, 0.5])),), airmass=1.2)
    record = prepare._linemask_settings(cfg, products, lt)
    assert record["blend"] == "ratio" and record["telluric_model_sha256"] == lt.sha256()
    assert record["blend_ratio"] == 0.3
    default = prepare._linemask_settings(FitConfig(stellar="mask"), products)
    assert default["blend"] == "off" and "telluric_model_sha256" not in default

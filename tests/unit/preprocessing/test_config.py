from __future__ import annotations

import pytest

from gtellcorr.errors import ConfigError
from gtellcorr.preprocessing.config import (
    BROAD_LINE_WINDOWS_OPTICAL,
    OPTICAL_ACTIVITY_WINDOWS,
    FitMasks,
    MasterContinuum,
    PysmeSettings,
    RVConfig,
    StellarLineMask,
    StellarModelConfig,
    StrongTelluricMask,
)


def test_reference_defaults() -> None:
    masks = FitMasks()
    assert masks.activity_windows_nm == OPTICAL_ACTIVITY_WINDOWS
    assert len(OPTICAL_ACTIVITY_WINDOWS) == 11 and len(BROAD_LINE_WINDOWS_OPTICAL) == 12
    assert masks.strong_telluric.threshold == 0.05
    assert StellarModelConfig().backend == "pysme"
    assert StellarModelConfig().ispec.resolution == 500000
    assert RVConfig().telluric_mask_molecules == ("h2o", "o2")


@pytest.mark.parametrize(
    ("group", "data", "message"),
    [
        (StellarLineMask, {"depth_below": 1.5}, r"^depth_below: must be in \(0, 1\]"),
        (StellarLineMask, {"pad_kms": -1}, r"^pad_kms: must be >= 0"),
        (StrongTelluricMask, {"threshold": 0}, r"^threshold: must be in \(0, 1\)"),
        (StrongTelluricMask, {"h2o_relcol": 0}, r"^h2o_relcol: must be > 0"),
        (StrongTelluricMask, {"chunk_nm": 0}, r"^chunk_nm: must be > 0"),
        (StrongTelluricMask, {"pad_nm": -1}, r"^pad_nm: must be >= 0"),
        (StrongTelluricMask, {"overwrite": True}, r"^overwrite: unknown parameter"),
        (FitMasks, {"bad_regions_nm": [[500, 400]]}, r"^bad_regions_nm\[0\]"),
        (MasterContinuum, {"method": "spline"}, r"^method: must be one of"),
        (MasterContinuum, {"nknots": 0}, r"^nknots: must be >= 1"),
        (MasterContinuum, {"max_wave_range": 0}, r"^max_wave_range: must be > 0"),
        (MasterContinuum, {"strong_line_probability": 2}, r"^strong_line_probability"),
        (PysmeSettings, {"wave_step_nm": 0.0}, r"^wave_step_nm: must be > 0"),
        (PysmeSettings, {"cscale_flag": "linear"}, r"cscale_flag"),
        (StellarModelConfig, {"backend": "phoenix"}, r"^backend: must be one of"),
        (StellarModelConfig, {"line_min_depth": 1}, r"^line_min_depth"),
        (RVConfig, {"telluric_mask_molecules": ["h2x"]}, r"^telluric_mask_molecules"),
        (RVConfig, {"pad_kms": -1}, r"^pad_kms: must be >= 0"),
        (RVConfig, {"ccf_v_lo": 10, "ccf_v_hi": -10}, r"^ccf_v_lo: must be smaller"),
        (RVConfig, {"ccf_dv": 0}, r"^ccf_dv: must be > 0"),
        (RVConfig, {"ccf_plot_max_orders": 0}, r"^ccf_plot_max_orders: must be >= 1"),
    ],
)
def test_validation(group: type, data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        group.from_dict(data)

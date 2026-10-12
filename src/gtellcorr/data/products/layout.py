"""Directory layout and file names of a run's products.

One run writes under

    <output_dir>/<target>/<site>_<instrument>/<date>/<product>/{data,figure}/

with the products ``Radial_Velocities``, ``mask_spectral_line``,
``template_stellar_spectrum`` and ``telluric_corr``, and file names built from the
stem ``<target>_<instrument>_<date>`` (e.g. ``TOI-2076_NEID_20220213``). Products
already in place are found by these names and reused when allowed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

__all__ = [
    "PRODUCTS",
    "PRODUCT_MASK",
    "PRODUCT_RV",
    "PRODUCT_TELLURIC",
    "PRODUCT_TEMPLATE",
    "RunLayout",
    "date_token",
    "safe_component",
]

PRODUCT_RV = "Radial_Velocities"
PRODUCT_MASK = "mask_spectral_line"
PRODUCT_TEMPLATE = "template_stellar_spectrum"
PRODUCT_TELLURIC = "telluric_corr"
PRODUCTS = (PRODUCT_RV, PRODUCT_MASK, PRODUCT_TEMPLATE, PRODUCT_TELLURIC)
Kind = Literal["data", "figure"]


def safe_component(value: object) -> str:
    """*value* as a path component: characters other than letters, digits, ``-``, ``_``
    and ``.`` become ``_``."""
    s = str(value).strip()
    kept = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)
    return kept.strip("_") or "unknown"


def date_token(times: Iterable[datetime | None]) -> str:
    """``YYYYMMDD``, or ``YYYYMMDD-YYYYMMDD`` for a multi-night set."""
    days = sorted({t.strftime("%Y%m%d") for t in times if t is not None})
    if not days:
        return "unknown-date"
    return days[0] if len(days) == 1 else f"{days[0]}-{days[-1]}"


@dataclass(frozen=True)
class RunLayout:
    """Product paths of one run (target x instrument x date)."""

    root: Path
    """Run directory ``<output_dir>/<target>/<site>_<instrument>/<date>``."""
    target: str
    """Target name."""
    instrument: str
    """Instrument token in file names (``NEID``; the part after ``telescope/``)."""
    date: str
    """Date token, ``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD`` (see :func:`date_token`)."""

    @classmethod
    def for_run(
        cls,
        output_dir: str | os.PathLike[str],
        *,
        target: str,
        site_token: str,
        instrument: str,
        date: str,
    ) -> RunLayout:
        """``<output_dir>/<target>/<site>_<instrument>/<date>``.

        ``instrument`` may be a registered ``telescope/instrument`` name; its
        instrument part is used.
        """
        token = instrument.split("/", 1)[-1]
        root = (
            Path(os.path.abspath(str(output_dir)))
            / safe_component(target)
            / f"{safe_component(site_token)}_{safe_component(token)}"
            / safe_component(date)
        )
        return cls(root=root, target=target, instrument=token, date=date)

    # -- directories ---------------------------------------------------------
    def product_dir(self, product: str, kind: Kind = "data", *, create: bool = True) -> Path:
        """``<root>/<product>/<kind>``, created unless ``create=False``."""
        if product not in PRODUCTS:
            raise ValueError(f"unknown product {product!r}; use one of {PRODUCTS}")
        if kind not in ("data", "figure"):
            raise ValueError(f"product kind must be 'data' or 'figure', got {kind!r}")
        path = self.root / product / kind
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def stem(self) -> str:
        """``<target>_<instrument>_<date>`` without spaces, the stem of the run's file names."""
        target = str(self.target).strip().replace(" ", "")
        instrument = str(self.instrument).strip().replace(" ", "")
        return f"{target}_{instrument}_{self.date}"

    # -- radial velocities ---------------------------------------------------
    @property
    def rv_table(self) -> Path:
        """Per-exposure RVs, ``<stem>_rv.csv``."""
        return self.product_dir(PRODUCT_RV) / f"{self.stem}_rv.csv"

    @property
    def rv_per_order_table(self) -> Path:
        """Per-order RVs of every exposure, ``<stem>_rv_per_order.csv``."""
        return self.product_dir(PRODUCT_RV) / f"{self.stem}_rv_per_order.csv"

    def ccf_path(self, obs_time: str, exposure: str | os.PathLike[str] | None = None) -> Path:
        """``CCF_<target>_<instrument>_<obstime>[_<exposure>].npz``."""
        suffix = ""
        if exposure:
            suffix = f"_{safe_component(os.path.splitext(os.path.basename(str(exposure)))[0])}"
        name = (
            f"CCF_{safe_component(self.target)}_{safe_component(self.instrument)}_"
            f"{safe_component(obs_time)}{suffix}.npz"
        )
        return self.product_dir(PRODUCT_RV) / name

    # -- stellar template and master spectrum --------------------------------
    @property
    def template_dir(self) -> Path:
        """Directory of the stellar template and master spectrum."""
        return self.product_dir(PRODUCT_TEMPLATE)

    @property
    def master_npz(self) -> Path:
        """Stellar master spectrum, ``<stem>_master.npz``."""
        return self.template_dir / f"{self.stem}_master.npz"

    @property
    def master_s1d(self) -> Path:
        """Stellar master spectrum as a text table, ``<stem>_master_s1d.dat``."""
        return self.template_dir / f"{self.stem}_master_s1d.dat"

    # -- spectral-line masks -------------------------------------------------
    @property
    def mask_dir(self) -> Path:
        """Directory of the stellar-line masks."""
        return self.product_dir(PRODUCT_MASK)

    @property
    def linemask_rest(self) -> Path:
        """Stellar-line mask in the stellar rest frame, ``<stem>_stellarlinemask_rest.txt``."""
        return self.mask_dir / f"{self.stem}_stellarlinemask_rest.txt"

    @property
    def linemask_settings(self) -> Path:
        """How the per-exposure stellar-line masks were built,
        ``<stem>_stellarlinemask_settings.json``."""
        return self.mask_dir / f"{self.stem}_stellarlinemask_settings.json"

    @property
    def lines_catalog(self) -> Path:
        """Detected stellar lines, ``<stem>_stellar_lines.json``."""
        return self.mask_dir / f"{self.stem}_stellar_lines.json"

    def exposure_linemask(
        self, exposure: str | os.PathLike[str], frame: str = "topocentric"
    ) -> Path:
        """Stellar-line mask of one exposure, ``<exposure stem>_stellarlinemask_<frame>.txt``."""
        stem = os.path.splitext(os.path.basename(str(exposure)))[0]
        return self.mask_dir / f"{stem}_stellarlinemask_{frame}.txt"

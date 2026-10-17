"""Raw observations: sites, instruments, targets, spectrum readers and ``Observation``."""

from __future__ import annotations

from gtellcorr.data.observation.config import DataConfig, Frames, SiteCoords
from gtellcorr.data.observation.frames import (
    FrameSpec,
    air_to_vac,
    from_air_topocentric,
    n_air,
    to_air_topocentric,
    vac_to_air,
)
from gtellcorr.data.observation.header import (
    HeaderField,
    HeaderSchema,
    read_on_site_weather,
    read_site_time,
)
from gtellcorr.data.observation.instrument import (
    InstrumentSpec,
    ResolutionSpec,
    SpectralSpec,
    identify,
    instruments,
    load_instrument_file,
    read_resolution,
)
from gtellcorr.data.observation.observation import Exposure, Observation
from gtellcorr.data.observation.readers import readers
from gtellcorr.data.observation.site import Site, resolve_site, sites
from gtellcorr.data.observation.spectrum import InputGrid, OrderSpectrum, RawOrder
from gtellcorr.data.observation.target import Star, Transit

__all__ = [
    "DataConfig",
    "Exposure",
    "FrameSpec",
    "Frames",
    "HeaderField",
    "HeaderSchema",
    "InputGrid",
    "InstrumentSpec",
    "Observation",
    "OrderSpectrum",
    "RawOrder",
    "ResolutionSpec",
    "Site",
    "SiteCoords",
    "SpectralSpec",
    "Star",
    "Transit",
    "air_to_vac",
    "from_air_topocentric",
    "identify",
    "instruments",
    "load_instrument_file",
    "n_air",
    "read_on_site_weather",
    "read_resolution",
    "read_site_time",
    "readers",
    "resolve_site",
    "sites",
    "to_air_topocentric",
    "vac_to_air",
]

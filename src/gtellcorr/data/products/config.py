"""Parameter group of the products computed after the fit."""

from __future__ import annotations

from dataclasses import dataclass

from gtellcorr.config.base import ConfigGroup

__all__ = ["Postprocessing"]


@dataclass(frozen=True, kw_only=True)
class Postprocessing(ConfigGroup):
    """Products computed after all exposures are fitted."""

    telluric_pca: bool = False
    """Compute a PCA of the fitted telluric models of each order across the exposures."""
    telluric_pca_components: int = 3
    """Number of principal components kept (cumulative models ``mean + PC1 + ...``)."""

    def validate(self) -> None:
        """At least one component."""
        if self.telluric_pca_components < 1:
            self.fail("telluric_pca_components", "must be >= 1")

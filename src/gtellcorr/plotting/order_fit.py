"""plot_order_fit — one order's telluric fit from a saved result.

Give it the path to an exposure-level result FITS written by the pipeline and it
renders one order as a 2-panel figure, dispatching on the stored ``treatment``:

  'mask'  — obs + best-fit continuum C + telluric C·T, excluded fit pixels
            shaded (the masked-out region); corrected obs/(C·T) below.
  'template' (``'model'`` in results of earlier versions) — obs + continuum C +
            stellar component C·M + telluric correction
            model C·T; the full C·M·T product is not drawn by default.
            If plot_pca=True and saved PCA models exist, the selected cumulative
            PCA telluric model is overlaid as C·T_pca and obs/(C·T_pca).

Returns ``(fig, (ax_top, ax_bot))`` for further composition, and (unless
``out=False``) saves a PNG.  No instrument / target hardcoding — everything is
read from the result file.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from gtellcorr.data.products.fits_schema import load_exposure_order as load_order_result

#: ``treatment`` values of the template treatment in result files.
_TEMPLATE_TREATMENTS = ("template", "model")


def _default_out(result_path: str, *, order=None, plot_pca: bool = False) -> str:
    """<result_dir>/figure/<stem>[_ord###].png, sibling to result data dir."""
    d, base = os.path.split(os.path.abspath(result_path))
    stem = os.path.splitext(base)[0]
    if order is not None:
        order_token = f"_ord{int(order):03d}"
        if order_token not in stem:
            stem += order_token
    if plot_pca:
        stem += "_pca"
    root = os.path.dirname(d) if os.path.basename(d) == "data" else d
    return os.path.join(root, "figure", stem + ".png")


def _select_pca_model(pca_models, n_pca, component=None, *, source_label="result"):
    """The cumulative PCA model with *component* components (default: all saved),
    or ``(None, None)`` with a warning if none is saved."""
    pca_models = np.asarray(pca_models, float)
    n_avail = min(int(n_pca), pca_models.shape[0])
    if n_avail <= 0:
        warnings.warn(
            f"{source_label} has an empty saved PCA telluric model; plotting standard fit only",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None
    if component is None:
        component = n_avail
    component = int(component)
    if component < 1 or component > n_avail:
        raise ValueError(f"pca_component must be in 1..{n_avail}; got {component}")
    return pca_models[component - 1], component


def _load_saved_pca_model_from_result_dict(result_path: str, result: dict, component=None):
    """The saved cumulative PCA telluric model of one order, or ``(None, None)`` with a
    warning if the result has none."""
    order = result.get("order")
    source_label = f"{result_path} order {order}"
    if "pca_models" not in result or "n_pca" not in result:
        warnings.warn(
            f"{source_label} has no saved PCA telluric model; plotting standard fit only",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None
    pca_mode = str(result.get("pca_mode", ""))
    if pca_mode != "cumulative_model":
        warnings.warn(
            f"{source_label} has PCA mode {pca_mode!r}, expected 'cumulative_model'; "
            "plotting standard fit only",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None
    return _select_pca_model(
        result["pca_models"], result["n_pca"], component, source_label=source_label
    )


def plot_order_fit(
    result_path: str, *, order=None, out=None, ax=None, plot_pca=False, pca_component=None
):
    """Plot one order's fit from an exposure-level FITS ``result_path``.

    out : None → save to the sibling ``figure/`` dir; a path → save there;
          False → do not save.  ax : a length-2 array of axes to draw into
          (shares them instead of making a figure).  If ``plot_pca`` is True and
          the result file carries saved PCA telluric models, add C*T_pca and
          obs/(C*T_pca).  Returns ``(fig, axes)``.
    """
    # One order of the exposure-level result FITS as a plain dictionary.
    # Instrument name, target, wavelength range, fitted molecules, and
    # stellar-treatment route all come from stored metadata rather than from config
    # or hardcoded device branches.
    r = load_order_result(result_path, order=order)

    # Core arrays on the fitting grid.  x is already the working air/topocentric
    # wavelength grid; obs is the flux that was fitted.  C is the continuum that
    # was projected out during the fit, M is the optional stellar template of the
    # template treatment, excl marks all pixels excluded from the fit, and T is the best-fit
    # telluric transmission on x.
    x, obs = r["wave_fit"], r["flux_obs"]
    C = r["continuum"]
    M = r["stellar_model"]
    excl = r["exclusion"]
    T = r["tell_model"]
    treatment = r["treatment"]

    # Optional PCA overlay.  PCA models are post-processing products appended to
    # the saved result after all exposures of the order are available.  They are complete
    # cumulative telluric models on this exposure's order wave_fit grid, not
    # isolated principal components.  Missing PCA fields simply leave the normal
    # two-panel plot unchanged.
    T_pca = pca_k = None
    if plot_pca:
        T_pca, pca_k = _load_saved_pca_model_from_result_dict(result_path, r, pca_component)

    # Older or no-fit products may not carry a continuum; plotting uses a flat
    # continuum in that case so the diagnostic remains readable.
    if C is None:
        C = np.ones_like(x)

    # Compact title metadata: fitted relcol values, fitted/fixed LSF shape if
    # present, resolving power, and reduced chi-square.
    rl = ", ".join(f"{m}={r['relcol'][m]:.3f}" for m in r["molecules"])
    shp = "" if not np.isfinite(r["lsf_shape"]) else f" {r['lsf_kind']}={r['lsf_shape']:.2f}"
    head = (
        f"{r['exposure']}  order {r['order']} ({r['wave_lo']:.1f}-{r['wave_hi']:.1f} nm) — "
        f"treatment={treatment.upper()}  [{rl}] R={r['resolution']:.0f}{shp} "
        f"χ²={r['chi2_reduced']:.2f}"
    )

    # Allow callers to draw into existing axes for composite figures; otherwise
    # create the standard upper-model / lower-correction layout.
    if ax is None:
        fig, ax = plt.subplots(
            2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.3])
        )
    else:
        fig = ax[0].figure

    # The grey mask region is drawn at the lowest z-order.  It is diagnostic:
    # these pixels were excluded from the residual but are still displayed.  The
    # saved result stores the union of active masks, e.g. stellar-line, activity,
    # user bad-region, and strong telluric-core exclusions.
    if excl.any():
        ax[0].fill_between(
            x, 0, 1.3, where=excl, color="0.85", step="mid", zorder=0, label="masked fit pixels"
        )

    # Route-dependent upper panel.  In the 'mask' route there is no stellar
    # template, so the fitted model is C*T.  In the template route the stellar
    # template M is shown separately as C*M while the telluric correction model
    # remains C*T; the full C*M*T product is deliberately not drawn by default to
    # keep the panel focused on components that are useful for telluric QC.
    # template treatment: "template" (earlier versions wrote "model")
    is_mask = not (treatment in _TEMPLATE_TREATMENTS and M is not None)
    ax[0].plot(x, obs, color="#111111", lw=0.5, zorder=1, label="observed")
    if not is_mask:
        ax[0].plot(x, C * M, color="#E69F00", lw=0.7, alpha=0.95, zorder=2, label="stellar  C·M")
        ax[0].plot(
            x, C * T, color="#009E73", lw=0.75, alpha=0.95, zorder=3, label="best-fit telluric  C·T"
        )
    else:
        ax[0].plot(
            x, C * T, color="#009E73", lw=0.75, alpha=0.95, zorder=3, label="best-fit telluric  C·T"
        )

    # PCA comparison curve: same continuum C, but replacing the fitted telluric
    # model T with the selected cumulative PCA telluric model T_pca.
    if T_pca is not None:
        ax[0].plot(
            x,
            C * T_pca,
            color="#003B8E",
            lw=0.85,
            alpha=0.95,
            zorder=3.5,
            label=f"PCA telluric  C·T_pca{pca_k}",
        )
    ax[0].plot(x, C, color="#0072B2", lw=0.85, ls="--", zorder=4, label="best-fit continuum C")
    ax[0].set_ylim(0, 1.25)
    ax[0].set_ylabel("flux")
    ax[0].legend(fontsize=8, loc="lower left", ncol=2)
    ax[0].set_title(head)

    # Lower panel: corrected spectra.  r['flux_corrected'] is obs/T from the
    # saved result; dividing by C displays the continuum-normalised correction
    # obs/(C*T).  The reference line at 1 highlights residual structure.
    ax[1].axhline(1, color=".7", lw=0.5, ls=":")
    if excl.any():
        ax[1].fill_between(
            x, 0, 2.0, where=excl, color="0.85", step="mid", zorder=0, label="masked fit pixels"
        )
    corr = r["flux_corrected"]
    ecorr = r.get("err_corrected")
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_norm = corr / C

    # Reliability band from the propagated observational uncertainty.  It grows
    # in deep telluric cores because err_corrected = err_obs / T; infinities are
    # converted to gaps for plotting.
    if ecorr is not None:
        fin = np.isfinite(ecorr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ecorr_norm = ecorr / C
        ax[1].fill_between(
            x,
            np.where(fin, corr_norm - ecorr_norm, np.nan),
            np.where(fin, corr_norm + ecorr_norm, np.nan),
            color="#0072B2",
            alpha=0.18,
            lw=0,
            zorder=1.5,
            label="±1σ (σ = obs err / (C·T))",
        )
    ax[1].plot(
        x, corr_norm, color="#111111", lw=0.5, zorder=2, label="telluric-corrected  obs/(C·T)"
    )

    # PCA-corrected spectrum for apples-to-apples visual comparison with the
    # fitted telluric correction.  It is not stored back into the result file.
    if T_pca is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_pca = obs / (C * T_pca)
        ax[1].plot(
            x,
            corr_pca,
            color="#003B8E",
            lw=0.65,
            zorder=3,
            label=f"PCA-corrected  obs/(C·T_pca{pca_k})",
        )
    ax[1].set_ylim(0, 1.6)
    ax[1].set_ylabel("corrected  obs/(C·T)")
    ax[1].set_xlabel("wavelength (air) [nm]")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    # Saving works as in the other plotting functions: None means the sibling
    # figure directory, False means caller handles the figure, and explicit path
    # means save exactly there.  PCA overlays get a _pca suffix by default.
    if out is not False:
        out = (
            _default_out(
                result_path, order=r.get("order"), plot_pca=bool(plot_pca and T_pca is not None)
            )
            if out is None
            else out
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130)
    return fig, ax

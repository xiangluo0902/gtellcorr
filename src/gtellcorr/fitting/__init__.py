"""The fitting engine: continuum, LSF, parameter blocks, solver, single-order and
joint-order fits."""

from __future__ import annotations

from gtellcorr.fitting.blocks import ModelState, ParameterBlock, default_blocks, parameter_blocks
from gtellcorr.fitting.config import (
    LSF,
    Continuum,
    Molecule,
    Molecules,
    MoleculeSelection,
    OrderSelection,
    Wavelength,
)
from gtellcorr.fitting.continuum import fit_continuum
from gtellcorr.fitting.fit_order import JointOrderInput, fit_joint_orders, fit_single_order
from gtellcorr.fitting.fit_spec import (
    FitSpec,
    build_fit_spec,
    configured_fit_molecules,
    select_fit_molecules,
    select_fit_orders_for_molecules,
)
from gtellcorr.fitting.lsf import apply_lsf
from gtellcorr.fitting.results import JointOrdersFitResult, OrderFitResult
from gtellcorr.fitting.solver import (
    OptResult,
    OrderEvaluation,
    build_residual,
    evaluate_order,
    minimize,
    pack,
    unpack,
)

__all__ = [
    "LSF",
    "Continuum",
    "FitSpec",
    "JointOrderInput",
    "JointOrdersFitResult",
    "ModelState",
    "Molecule",
    "MoleculeSelection",
    "Molecules",
    "OptResult",
    "OrderEvaluation",
    "OrderFitResult",
    "OrderSelection",
    "ParameterBlock",
    "Wavelength",
    "apply_lsf",
    "build_fit_spec",
    "build_residual",
    "configured_fit_molecules",
    "default_blocks",
    "evaluate_order",
    "fit_continuum",
    "fit_joint_orders",
    "fit_single_order",
    "minimize",
    "pack",
    "parameter_blocks",
    "select_fit_molecules",
    "select_fit_orders_for_molecules",
    "unpack",
]

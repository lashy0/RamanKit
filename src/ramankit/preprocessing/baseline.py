from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, cast

import numpy as np
from pybaselines import Baseline  # type: ignore[import-untyped]

from ramankit.pipelines.pipeline import PreprocessingStep
from ramankit.preprocessing._types import Array1D


def _resolve_baseline_method(method_name: str):
    """Return a pybaselines method for the configured baseline step."""

    baseline_fitter = Baseline()
    if not hasattr(baseline_fitter, method_name):
        raise ValueError(f"Unsupported pybaselines method '{method_name}'.")
    return getattr(baseline_fitter, method_name)


def _coerce_parameter_array(name: str, value: object, axis: Array1D) -> Array1D:
    """Return a validated 1D float array for an axis-shaped parameter."""

    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"Expected {name} to be one-dimensional.")
    if array.shape != axis.shape:
        raise ValueError(f"Expected {name} to match the spectral axis shape.")
    return array


def _summarize_parameter_value(value: object) -> object:
    """Return a provenance-safe representation of a parameter value."""

    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def _as_float(value: object) -> float:
    """Return a float from a validated numeric parameter."""

    return float(cast(float, value))


def _as_int(value: object) -> int:
    """Return an int from a validated integer-like parameter."""

    return int(cast(int, value))


class BaselineStep(PreprocessingStep):
    """Define shared behavior for pybaselines-backed baseline correction steps."""

    function_name = "baseline_correct"
    method_name: ClassVar[str]
    positive_fields: ClassVar[tuple[str, ...]] = ()
    non_negative_fields: ClassVar[tuple[str, ...]] = ()
    positive_integer_fields: ClassVar[tuple[str, ...]] = ()
    non_negative_integer_fields: ClassVar[tuple[str, ...]] = ()
    open_unit_interval_fields: ClassVar[tuple[str, ...]] = ()
    closed_unit_interval_fields: ClassVar[tuple[str, ...]] = ()
    allowed_values: ClassVar[dict[str, tuple[object, ...]]] = {}

    def __post_init__(self) -> None:
        """Validate the configured method name and scalar parameters."""

        _resolve_baseline_method(self.method_name)
        self._validate_scalar_parameters(self._scalar_parameters())

    def parameters(self) -> dict[str, object]:
        """Return provenance-safe step parameters."""

        return {
            field.name: _summarize_parameter_value(getattr(self, field.name))
            for field in fields(cast(Any, self))
        }

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        """Apply the configured pybaselines method to one spectrum."""

        baseline_fitter = Baseline(x_data=axis)
        baseline, _ = getattr(baseline_fitter, self.method_name)(
            intensity,
            **self._backend_parameters(axis),
        )
        return intensity - baseline

    def _scalar_parameters(self) -> dict[str, object]:
        """Return non-axis-dependent parameters for eager validation."""

        return {
            field.name: getattr(self, field.name)
            for field in fields(cast(Any, self))
            if field.name not in {"weights", "alpha"}
        }

    def _backend_parameters(self, axis: Array1D) -> dict[str, object]:
        """Return validated backend parameters for pybaselines."""

        parameters = self._scalar_parameters()
        for field in fields(cast(Any, self)):
            if field.name not in {"weights", "alpha"}:
                continue
            value = getattr(self, field.name)
            if value is not None:
                parameters[field.name] = _coerce_parameter_array(field.name, value, axis)
            else:
                parameters[field.name] = None
        return parameters

    def _validate_scalar_parameters(self, parameters: dict[str, object]) -> None:
        """Validate common scalar parameter constraints."""

        for name in self.positive_fields:
            if _as_float(parameters[name]) <= 0:
                raise ValueError(f"Expected {name} to be positive.")

        for name in self.non_negative_fields:
            value = parameters[name]
            if value is not None and _as_float(value) < 0:
                raise ValueError(f"Expected {name} to be non-negative.")

        for name in self.positive_integer_fields:
            if _as_int(parameters[name]) <= 0:
                raise ValueError(f"Expected {name} to be a positive integer.")

        for name in self.non_negative_integer_fields:
            if _as_int(parameters[name]) < 0:
                raise ValueError(f"Expected {name} to be a non-negative integer.")

        for name in self.open_unit_interval_fields:
            value = _as_float(parameters[name])
            if not 0.0 < value < 1.0:
                raise ValueError(f"Expected {name} to be in the open interval (0, 1).")

        for name in self.closed_unit_interval_fields:
            value = _as_float(parameters[name])
            if not 0.0 < value <= 1.0:
                raise ValueError(f"Expected {name} to be in the interval (0, 1].")

        for name, allowed in self.allowed_values.items():
            value = parameters[name]
            if value not in allowed:
                joined = ", ".join(str(item) for item in allowed)
                raise ValueError(f"Expected {name} to be one of: {joined}.")

        self._validate_custom_parameters(parameters)

    def _validate_custom_parameters(self, parameters: dict[str, object]) -> None:
        """Validate method-specific scalar parameters beyond the shared rules."""


@dataclass(frozen=True, slots=True)
class ASLS(BaselineStep):
    """Good default for smooth baselines when tuning `p` is acceptable."""

    method_name = "asls"
    positive_fields = ("lam", "tol")
    positive_integer_fields = ("diff_order", "max_iter")
    open_unit_interval_fields = ("p",)

    lam: float = 1e6
    p: float = 1e-2
    diff_order: int = 2
    max_iter: int = 50
    tol: float = 1e-3
    weights: Array1D | None = None


@dataclass(frozen=True, slots=True)
class IASLS(BaselineStep):
    """Use when ASLS needs stronger control over derivative smoothing."""

    method_name = "iasls"
    positive_fields = ("lam", "lam_1", "tol")
    positive_integer_fields = ("diff_order", "max_iter")
    open_unit_interval_fields = ("p",)

    lam: float = 1e6
    p: float = 1e-2
    lam_1: float = 1e-4
    max_iter: int = 50
    tol: float = 1e-3
    weights: Array1D | None = None
    diff_order: int = 2


@dataclass(frozen=True, slots=True)
class AIRPLS(BaselineStep):
    """More robust than ASLS when peaks are strong or dense."""

    method_name = "airpls"
    positive_fields = ("lam", "tol")
    positive_integer_fields = ("diff_order", "max_iter")

    lam: float = 1e6
    diff_order: int = 2
    max_iter: int = 50
    tol: float = 1e-3
    weights: Array1D | None = None
    normalize_weights: bool = False


@dataclass(frozen=True, slots=True)
class ARPLS(BaselineStep):
    """Strong general-purpose least-squares baseline corrector."""

    method_name = "arpls"
    positive_fields = ("lam", "tol")
    positive_integer_fields = ("diff_order", "max_iter")

    lam: float = 1e5
    diff_order: int = 2
    max_iter: int = 50
    tol: float = 1e-3
    weights: Array1D | None = None


@dataclass(frozen=True, slots=True)
class DRPLS(BaselineStep):
    """Use when ARPLS needs stronger reweighting control."""

    method_name = "drpls"
    positive_fields = ("lam", "eta", "tol")
    positive_integer_fields = ("diff_order", "max_iter")

    lam: float = 1e5
    eta: float = 0.5
    max_iter: int = 50
    tol: float = 1e-3
    weights: Array1D | None = None
    diff_order: int = 2


@dataclass(frozen=True, slots=True)
class IARPLS(BaselineStep):
    """Improved ARPLS variant for difficult asymmetric baselines."""

    method_name = "iarpls"
    positive_fields = ("lam", "tol")
    positive_integer_fields = ("diff_order", "max_iter")

    lam: float = 1e5
    diff_order: int = 2
    max_iter: int = 50
    tol: float = 1e-3
    weights: Array1D | None = None


@dataclass(frozen=True, slots=True)
class ASPLS(BaselineStep):
    """Use when smoothness should adapt across the spectral axis."""

    method_name = "aspls"
    positive_fields = ("lam", "tol", "asymmetric_coef")
    positive_integer_fields = ("diff_order", "max_iter")

    lam: float = 1e5
    diff_order: int = 2
    max_iter: int = 100
    tol: float = 1e-3
    weights: Array1D | None = None
    alpha: Array1D | None = None
    asymmetric_coef: float = 0.5


@dataclass(frozen=True, slots=True)
class Poly(BaselineStep):
    """Simple polynomial baseline for smooth low-order trends."""

    method_name = "poly"
    non_negative_integer_fields = ("poly_order",)

    poly_order: int = 2
    weights: Array1D | None = None


@dataclass(frozen=True, slots=True)
class ModPoly(BaselineStep):
    """Polynomial baseline fit that works well for simple fluorescence trends."""

    method_name = "modpoly"
    positive_fields = ("tol",)
    non_negative_integer_fields = ("poly_order",)
    positive_integer_fields = ("max_iter",)

    poly_order: int = 2
    tol: float = 1e-3
    max_iter: int = 250
    weights: Array1D | None = None
    use_original: bool = False
    mask_initial_peaks: bool = False


@dataclass(frozen=True, slots=True)
class PenalisedPoly(BaselineStep):
    """Polynomial baseline with explicit robust loss selection."""

    method_name = "penalized_poly"
    positive_fields = ("tol", "alpha_factor")
    non_negative_fields = ("threshold",)
    non_negative_integer_fields = ("poly_order",)
    positive_integer_fields = ("max_iter",)
    closed_unit_interval_fields = ("alpha_factor",)
    allowed_values = {
        "cost_function": (
            "asymmetric_truncated_quadratic",
            "symmetric_truncated_quadratic",
            "asymmetric_huber",
            "symmetric_huber",
            "asymmetric_indec",
        )
    }

    poly_order: int = 2
    tol: float = 1e-3
    max_iter: int = 250
    weights: Array1D | None = None
    cost_function: str = "asymmetric_truncated_quadratic"
    threshold: float | None = None
    alpha_factor: float = 0.99


@dataclass(frozen=True, slots=True)
class IModPoly(BaselineStep):
    """Improved ModPoly for stronger peak masking during polynomial fitting."""

    method_name = "imodpoly"
    positive_fields = ("tol", "num_std")
    non_negative_integer_fields = ("poly_order",)
    positive_integer_fields = ("max_iter",)

    poly_order: int = 2
    tol: float = 1e-3
    max_iter: int = 250
    weights: Array1D | None = None
    use_original: bool = False
    mask_initial_peaks: bool = True
    num_std: float = 1.0


@dataclass(frozen=True, slots=True)
class Goldindec(BaselineStep):
    """Robust polynomial baseline method for asymmetric backgrounds."""

    method_name = "goldindec"
    positive_fields = ("tol", "alpha_factor", "tol_2", "tol_3")
    non_negative_integer_fields = ("poly_order",)
    positive_integer_fields = ("max_iter", "max_iter_2")
    closed_unit_interval_fields = ("peak_ratio", "alpha_factor")
    allowed_values = {"cost_function": ("asymmetric_indec", "indec")}

    poly_order: int = 2
    tol: float = 1e-3
    max_iter: int = 250
    weights: Array1D | None = None
    cost_function: str = "asymmetric_indec"
    peak_ratio: float = 0.5
    alpha_factor: float = 0.99
    tol_2: float = 1e-3
    tol_3: float = 1e-6
    max_iter_2: int = 100


@dataclass(frozen=True, slots=True)
class IRSQR(BaselineStep):
    """Spline-based baseline for difficult, slowly varying backgrounds."""

    method_name = "irsqr"
    positive_fields = ("lam", "tol")
    non_negative_fields = ("eps",)
    positive_integer_fields = ("num_knots", "spline_degree", "diff_order", "max_iter")
    open_unit_interval_fields = ("quantile",)

    lam: float = 100.0
    quantile: float = 0.05
    num_knots: int = 100
    spline_degree: int = 3
    diff_order: int = 3
    max_iter: int = 100
    tol: float = 1e-6
    weights: Array1D | None = None
    eps: float | None = None


@dataclass(frozen=True, slots=True)
class CornerCutting(BaselineStep):
    """Fast geometric baseline estimate for simple envelopes."""

    method_name = "corner_cutting"
    positive_integer_fields = ("max_iter",)

    max_iter: int = 100


@dataclass(frozen=True, slots=True)
class FABC(BaselineStep):
    """Automatic baseline correction when manual tuning should be minimal."""

    method_name = "fabc"
    positive_fields = ("lam", "num_std")
    non_negative_fields = ("scale",)
    positive_integer_fields = ("diff_order", "min_length")

    lam: float = 1e6
    scale: float | None = None
    num_std: float = 3.0
    diff_order: int = 2
    min_length: int = 2
    weights: Array1D | None = None
    weights_as_mask: bool = False
    pad_kwargs: dict[str, object] | None = None

    def _validate_custom_parameters(self, parameters: dict[str, object]) -> None:
        """Validate FABC-specific parameter combinations."""

        pad_kwargs = parameters["pad_kwargs"]
        if pad_kwargs is not None and not isinstance(pad_kwargs, dict):
            raise ValueError("Expected pad_kwargs to be a mapping or None.")

from __future__ import annotations

from dataclasses import dataclass

from pybaselines import Baseline  # type: ignore[import-untyped]

from ramankit.preprocessing._base import PreprocessingStep
from ramankit.preprocessing._types import Array1D


@dataclass(frozen=True, slots=True)
class ASLS(PreprocessingStep):
    """Asymmetric least-squares baseline correction."""

    function_name = "baseline_correct"
    method_name = "asls"

    lam: float = 1e6
    p: float = 1e-2
    diff_order: int = 2
    max_iter: int = 50
    tol: float = 1e-3

    def parameters(self) -> dict[str, object]:
        return {
            "lam": self.lam,
            "p": self.p,
            "diff_order": self.diff_order,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def _transform(self, intensity: Array1D, axis: Array1D) -> Array1D:
        baseline_fitter = Baseline(x_data=axis)
        baseline, _ = baseline_fitter.asls(
            intensity,
            lam=self.lam,
            p=self.p,
            diff_order=self.diff_order,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return intensity - baseline

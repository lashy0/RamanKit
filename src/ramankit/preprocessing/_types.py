from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ramankit.core.collection import SpectrumCollection
from ramankit.core.image import RamanImage
from ramankit.core.spectrum import Spectrum

type Array1D = npt.NDArray[np.float64]
type Array2D = npt.NDArray[np.float64]
type FloatArray = npt.NDArray[np.float64]

type SpectralData = Spectrum | SpectrumCollection | RamanImage
SpectralDataT = TypeVar("SpectralDataT", Spectrum, SpectrumCollection, RamanImage)
type AxisTransform1D = Callable[[Array1D, Array1D], tuple[Array1D, Array1D]]
type AxisTransform2D = Callable[[Array2D, Array1D], tuple[Array1D, Array2D] | None]
type Transform1D = Callable[[Array1D, Array1D], Array1D]
type Transform2D = Callable[[Array2D, Array1D], Array2D | None]

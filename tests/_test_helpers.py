from __future__ import annotations

import numpy as np

from ramankit import RamanImage, SpectrumCollection
from ramankit.preprocessing import PreprocessingStep


def apply_collection_row_by_row(
    step: PreprocessingStep,
    collection: SpectrumCollection,
) -> SpectrumCollection:
    spectra = [step.apply(collection[index]) for index in range(collection.n_spectra)]
    return SpectrumCollection.from_spectra(spectra)


def apply_image_pixel_by_pixel(step: PreprocessingStep, image: RamanImage) -> RamanImage:
    rows: list[list[np.ndarray]] = []
    for row_index in range(image.spatial_shape[0]):
        row: list[np.ndarray] = []
        for column_index in range(image.spatial_shape[1]):
            row.append(step.apply(image.pixel(row_index, column_index)).intensity)
        rows.append(row)
    return RamanImage(
        axis=image.axis,
        intensity=np.array(rows, dtype=np.float64),
        metadata=image.metadata,
        provenance=image.provenance,
        spectral_axis_name=image.spectral_axis_name,
        spectral_unit=image.spectral_unit,
    )

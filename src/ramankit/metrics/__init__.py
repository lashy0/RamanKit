"""Expose general spectral similarity and quality metrics."""

from ramankit.metrics.quality import band_area, snr
from ramankit.metrics.similarity import cosine_similarity, mse, pearson_correlation

__all__ = [
    "band_area",
    "cosine_similarity",
    "mse",
    "pearson_correlation",
    "snr",
]

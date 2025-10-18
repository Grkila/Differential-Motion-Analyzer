"""Core motion detection module."""

from motioneye.core.detection import (
    get_detections,
    get_mask,
    compute_centroid,
    get_contour_detections,
    non_max_suppression,
    apply_kernel,
)

__all__ = [
    "get_detections",
    "get_mask",
    "compute_centroid",
    "get_contour_detections",
    "non_max_suppression",
    "apply_kernel",
]

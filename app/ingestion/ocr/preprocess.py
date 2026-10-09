"""Image preprocessing for OCR: grayscale conversion, Otsu's thresholding,
deskew via Hough line transform, and contrast normalization via CLAHE.

Responsibilities (Section 4.2 – app.ingestion.ocr):
- Accept a raw PIL Image or numpy array.
- Return a preprocessed numpy uint8 array ready for Tesseract.
- No OCR, chunking, or vector-store operations here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (any mode) to a BGR numpy array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR array to single-channel grayscale."""
    if len(img.shape) == 2:
        return img  # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur then Otsu's global thresholding.

    The blur reduces high-frequency noise before thresholding,
    which typically improves Tesseract word-boundary detection.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _compute_skew_angle(binary: np.ndarray) -> float:
    """Estimate page skew angle in degrees using probabilistic Hough lines.

    Returns a value in [-45, 45] degrees. Returns 0.0 when no lines are
    detected (e.g., for blank or nearly-blank pages).
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return 0.0

    angles: list[float] = []
    flat_lines = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in flat_lines:
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        # Only use near-horizontal lines (text baselines)
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    return median_angle


def deskew(binary: np.ndarray) -> np.ndarray:
    """Rotate the binary image to correct detected skew.

    The rotation is bounded to ±45° so that vertically-oriented content
    is never accidentally rotated 90°. For angles smaller than 0.5°, no
    rotation is applied to avoid introducing unnecessary interpolation
    artifacts.
    """
    angle = _compute_skew_angle(binary)
    if abs(angle) < 0.5:
        return binary

    h, w = binary.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Normalize contrast using CLAHE (Contrast-Limited Adaptive Histogram Equalization).

    CLAHE works on the L-channel; for already-grayscale inputs we apply it
    directly. This improves legibility of faded ink or uneven scan lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(image: Image.Image | np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: grayscale → CLAHE → Otsu threshold → deskew.

    Args:
        image: Input image as either a PIL Image (any mode) or a BGR numpy array.

    Returns:
        A binary (0/255) uint8 numpy array suitable for Tesseract ingestion.
    """
    if isinstance(image, Image.Image):
        bgr = pil_to_bgr(image)
    else:
        bgr = image  # assume already BGR numpy array

    gray = to_grayscale(bgr)
    enhanced = apply_clahe(gray)
    binary = otsu_threshold(enhanced)
    deskewed = deskew(binary)
    return deskewed

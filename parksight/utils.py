"""
Image preprocessing utilities for parking space detection.

Provides CLAHE enhancement, morphological cleanup, stall-line enhancement,
and bounding-box annotation — all operating on OpenCV / PIL images.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline on a BGR OpenCV image.

    Steps:

    1. Grayscale conversion + CLAHE contrast enhancement.
    2. Otsu thresholding to isolate white stall paint.
    3. Morphological closing / opening to clean the stripe mask.
    4. Dilation + contour detection to find the largest stripe cluster.
    5. Crop the original BGR image to that cluster's bounding box.

    If no cluster is found the original image is returned unchanged.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Input image in BGR colour order (OpenCV convention).

    Returns
    -------
    numpy.ndarray
        Cropped (or original) BGR image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    _, thresh_otsu = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(cleaned, kernel_dil, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        pad = 10
        x_min = max(0, x - pad)
        y_min = max(0, y - pad)
        x_max = min(image_bgr.shape[1], x + w + pad)
        y_max = min(image_bgr.shape[0], y + h + pad)

        return image_bgr[y_min:y_max, x_min:x_max]

    return image_bgr


def enhance_white_stalls(image_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance white parking-stall markings in a BGR image.

    Returns a binary mask (uint8, 0/255) where white stall lines are
    highlighted after CLAHE + Otsu + morphological cleanup.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Input image in BGR colour order.

    Returns
    -------
    numpy.ndarray
        Binary mask highlighting stall lines.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    _, thresh_otsu = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return clean


def draw_bounding_boxes_pil(
    pil_image: Image.Image,
    boxes: list[list[float]],
    outline: str = "red",
    width: int = 2,
) -> Image.Image:
    """
    Draw rectangular bounding boxes on a PIL Image.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Image to annotate (modified in place).
    boxes : list of [x0, y0, x1, y1]
        Bounding box coordinates.
    outline : str
        Box colour.
    width : int
        Line thickness in pixels.

    Returns
    -------
    PIL.Image.Image
        The same image, with boxes drawn.
    """
    draw = ImageDraw.Draw(pil_image)
    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=outline, width=width)
    return pil_image

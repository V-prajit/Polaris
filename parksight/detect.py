"""
ML baseline: zero-shot parking detection with Grounding DINO.

Wraps the ``IDEA-Research/grounding-dino-tiny`` model from HuggingFace
Transformers to detect parking spaces and cars in satellite imagery
without any fine-tuning.

Students are expected to **beat this baseline** — via fine-tuning,
better prompts, segmentation models, or entirely different architectures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Default labels for zero-shot detection
DEFAULT_LABELS = ["parking space", "car"]


@dataclass
class DetectionResult:
    """Container for detection outputs."""

    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class ParkingDetector:
    """
    Zero-shot parking detector powered by Grounding DINO.

    The model is loaded lazily on first call to :meth:`detect` so that
    importing the module is cheap.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    device : str or None
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  If *None*, auto-detected.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or self._pick_device()
        self._processor = None
        self._model = None

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        """Lazy-load the model and processor."""
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info("Loading %s on %s …", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

    def detect(
        self,
        pil_image: Image.Image,
        labels: list[str] | None = None,
        threshold: float = 0.25,
    ) -> DetectionResult:
        """
        Run zero-shot detection on a PIL image.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            RGB satellite tile.
        labels : list[str], optional
            Text prompts for zero-shot detection (default: ``["parking space", "car"]``).
        threshold : float
            Confidence threshold for keeping detections (default 0.25).

        Returns
        -------
        DetectionResult
            Detected bounding boxes, confidence scores, and label strings.
        """
        self._load()

        if labels is None:
            labels = DEFAULT_LABELS

        inputs = self._processor(
            images=pil_image, text=labels, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        boxes = []
        scores = []
        det_labels = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            if float(score) >= threshold:
                boxes.append(box.tolist())
                scores.append(float(score))
                det_labels.append(label)

        return DetectionResult(boxes=boxes, scores=scores, labels=det_labels)

    def count_spots(self, pil_image: Image.Image, **kwargs) -> int:
        """
        End-to-end: preprocess → detect → count unique detections.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            RGB satellite tile.
        **kwargs
            Forwarded to :meth:`detect`.

        Returns
        -------
        int
            Number of detected parking spots / cars.
        """
        from .utils import preprocess

        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cropped_bgr = preprocess(bgr)
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)

        result = self.detect(cropped_pil, **kwargs)
        return len(result.boxes)

    @staticmethod
    def annotate(
        pil_image: Image.Image,
        result: DetectionResult,
        outline: str = "lime",
        width: int = 2,
    ) -> Image.Image:
        """
        Draw detection boxes and labels on a copy of the image.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            Original image.
        result : DetectionResult
            Output from :meth:`detect`.
        outline : str
            Box colour.
        width : int
            Line thickness.

        Returns
        -------
        PIL.Image.Image
            Annotated copy of the image.
        """
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        for box, score, label in zip(result.boxes, result.scores, result.labels):
            x0, y0, x1, y1 = [int(c) for c in box]
            draw.rectangle([x0, y0, x1, y1], outline=outline, width=width)
            draw.text((x0, max(y0 - 12, 0)), f"{label} {score:.2f}", fill=outline)
        return annotated

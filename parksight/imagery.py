"""
Image tiling utilities and optional Google Earth Engine helpers.

Provides functions to split satellite images into non-overlapping crops
and display tile grids — useful for preparing training data or feeding
tiles to detection models.

GEE helpers are **optional**: they are only available when the ``ee``
package is installed and authenticated.
"""

from __future__ import annotations

import io
from math import ceil
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def crop_non_overlapping(
    img_bytes: bytes,
    tile_size: Tuple[int, int] = (256, 256),
    keep_partial: bool = False,
) -> List[Image.Image]:
    """
    Split an image (given as raw bytes) into non-overlapping crops.

    Parameters
    ----------
    img_bytes : bytes
        Raw PNG / JPEG bytes (e.g. ``response.content``).
    tile_size : (int, int)
        ``(width, height)`` of each crop in pixels.
    keep_partial : bool
        If *True*, keep edge tiles that are smaller than *tile_size*.

    Returns
    -------
    list[PIL.Image.Image]
        One ``Image`` per tile.
    """
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    tw, th = tile_size
    tiles: List[Image.Image] = []

    cols = ceil(w / tw) if keep_partial else w // tw
    rows = ceil(h / th) if keep_partial else h // th

    for r in range(rows):
        for c in range(cols):
            left = c * tw
            upper = r * th
            right = min(left + tw, w)
            lower = min(upper + th, h)

            if not keep_partial and (right - left < tw or lower - upper < th):
                continue

            tile = img.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles


def show_tiles(tiles: List[Image.Image], cols: int | None = None) -> None:
    """
    Display tiles in a matplotlib grid.

    Parameters
    ----------
    tiles : list[PIL.Image.Image]
        Images to display.
    cols : int, optional
        Number of columns (defaults to ``min(len(tiles), 6)``).
    """
    if not tiles:
        print("No tiles to display")
        return

    cols = cols or min(len(tiles), 6)
    rows = ceil(len(tiles) / cols)

    fig, axarr = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    fig.suptitle("Cropped Tiles")
    axes = np.asarray(axarr).flat if isinstance(axarr, (list, np.ndarray)) else [axarr]

    for ax, tile in zip(axes, tiles):
        ax.imshow(tile)
        ax.axis("off")

    for ax in list(axes)[len(tiles):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ── Optional Google Earth Engine helpers ───────────────────────────
#
# These require:
#   pip install earthengine-api
#   ee.Authenticate()   (one-time browser login)
#   ee.Initialize()

try:
    import ee  # noqa: F401

    def make_geometry(
        *,
        latlon: Tuple[float, float] | None = None,
        buffer_m: float = 300,
        polygon: dict | None = None,
    ) -> "ee.Geometry":
        """
        Build an ``ee.Geometry`` from a lat/lon point (+ buffer) or GeoJSON polygon.
        """
        if polygon:
            return ee.Geometry(polygon)
        if latlon:
            lon, lat = latlon[1], latlon[0]
            return ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()
        raise ValueError("Supply latlon or polygon")

    def get_best_sentinel(
        geom: "ee.Geometry",
        *,
        start_date: str = "2025-05-01",
        end_date: str = "2025-06-01",
    ) -> "ee.Image":
        """
        Return the least-cloudy Sentinel-2 SR image over *geom* in the date window.
        """
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .sort("CLOUD_COVER")
        )
        return col.first()

    def quick_png_thumbnail(
        image: "ee.Image",
        geom: "ee.Geometry",
        *,
        vis: dict | None = None,
        scale: int = 10,
    ) -> bytes:
        """
        Return raw PNG bytes (up to 1280x1280) for a quick preview.
        """
        import requests as _requests

        url = image.clip(geom).getThumbURL(
            {
                "region": geom,
                "dimensions": 1024,
                "scale": scale,
                **(vis or {"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]}),
            }
        )
        resp = _requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

except ImportError:
    # GEE not installed — helpers unavailable (this is fine for the hackathon)
    pass

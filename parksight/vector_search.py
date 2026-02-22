"""
Polaris semantic vector search using Actian VectorAI DB and Gemini embeddings.

This module is intentionally async-first and uses AsyncCortexClient end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - handled at runtime with clear errors
    genai = None

try:
    from cortex import AsyncCortexClient, DistanceMetric
except Exception:  # pragma: no cover - handled at runtime with clear errors
    AsyncCortexClient = None
    DistanceMetric = None

logger = logging.getLogger(__name__)

VECTOR_COLLECTION = os.getenv("VECTORDB_COLLECTION", "polaris_hex_cells")
VECTOR_DIMENSION = 768
EMBED_MODEL = "models/gemini-embedding-001"
PLACES_RADIUS_METERS = 200
PLACES_NEW_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_LEGACY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

_VALID_TASK_TYPES = {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}
_POI_CATEGORY_TYPES: dict[str, tuple[str, ...]] = {
    "restaurants": ("restaurant",),
    "retail_stores": ("shopping_mall", "supermarket", "department_store"),
    "offices": ("corporate_office",),
    "transit_stops": ("transit_station",),
}

_GENAI_CONFIGURED = False
_GENAI_LOCK = asyncio.Lock()
_POI_CACHE: dict[tuple[float, float], dict[str, int]] = {}
_POI_CACHE_LOCK = asyncio.Lock()
_GEOCODE_CACHE: dict[tuple[float, float], str] = {}
_GEOCODE_CACHE_LOCK = asyncio.Lock()


def _vector_db_host() -> str:
    return os.getenv("VECTORDB_HOST", "localhost:50051")


def _reverse_geocode_sync(lat: float, lon: float) -> str:
    """Reverse geocode via Nominatim to get a human-readable location string."""
    try:
        params = urllib.parse.urlencode({
            "lat": lat, "lon": lon, "format": "json", "zoom": 18,
        })
        url = f"https://nominatim.openstreetmap.org/reverse?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Polaris/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("display_name", "")
    except Exception as exc:
        logger.warning("Reverse geocode failed for (%.5f, %.5f): %s", lat, lon, exc)
        return ""


async def reverse_geocode_cached(lat: float, lon: float) -> str:
    """Reverse geocode with caching to avoid duplicate Nominatim calls."""
    cache_key = (round(lat, 4), round(lon, 4))
    async with _GEOCODE_CACHE_LOCK:
        cached = _GEOCODE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = await asyncio.to_thread(_reverse_geocode_sync, lat, lon)
    async with _GEOCODE_CACHE_LOCK:
        _GEOCODE_CACHE[cache_key] = result
    return result


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _require_dependencies() -> None:
    if AsyncCortexClient is None:
        raise RuntimeError(
            "Actian client missing. Install the actiancortex wheel from "
            "https://github.com/hackmamba-io/actian-vectorAI-db-beta."
        )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 1)


def _density_class(total_spots: int) -> str:
    if total_spots >= 300:
        return "very high"
    if total_spots >= 180:
        return "high"
    if total_spots >= 80:
        return "moderate"
    if total_spots > 0:
        return "low"
    return "very low"


def _extract_centroid(hex_cell: dict[str, Any]) -> tuple[float | None, float | None]:
    centroid = hex_cell.get("centroid")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        try:
            return float(centroid[0]), float(centroid[1])
        except (TypeError, ValueError):
            return None, None
    if isinstance(centroid, dict):
        lat = centroid.get("lat")
        lon = centroid.get("lon")
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None, None
    return None, None


def _poi_summary_sentence(poi_summary: dict[str, int] | None) -> str:
    if not poi_summary:
        return "Nearby points of interest were not available during indexing."

    restaurants = _safe_int(poi_summary.get("restaurants"))
    retail_stores = _safe_int(poi_summary.get("retail_stores"))
    offices = _safe_int(poi_summary.get("offices"))
    transit = _safe_int(poi_summary.get("transit_stops"))
    return (
        f"Nearby context within roughly {PLACES_RADIUS_METERS} meters includes "
        f"{restaurants} restaurants, {retail_stores} retail stores, "
        f"{offices} office locations, and {transit} transit stops."
    )


def parking_profile_to_text(
    hex_cell: dict[str, Any],
    poi_summary: dict[str, int] | None = None,
    location_context: str = "",
) -> str:
    """
    Convert one macro hex cell into rich semantic text suitable for embeddings.
    """
    hex_id = str(hex_cell.get("hex_id", "unknown"))
    surface = _safe_int(hex_cell.get("surface"))
    structured = _safe_int(hex_cell.get("structured"))
    street = _safe_int(hex_cell.get("street"))
    total = _safe_int(hex_cell.get("total"), surface + structured + street)
    if total <= 0:
        total = surface + structured + street

    surface_pct = _pct(surface, total)
    structured_pct = _pct(structured, total)
    street_pct = _pct(street, total)
    density = _density_class(total)

    lat, lon = _extract_centroid(hex_cell)
    if lat is not None and lon is not None:
        location_phrase = f"The hex centroid is near latitude {lat:.6f}, longitude {lon:.6f}."
    else:
        location_phrase = "The hex centroid coordinates are not available."

    location_ctx = ""
    if location_context:
        location_ctx = f"This area is located near {location_context}. "

    return (
        f"Atlanta parking cell {hex_id}. "
        f"{location_ctx}"
        f"Estimated total parking capacity is {total} spots. "
        f"Surface parking contributes {surface} spots ({surface_pct} percent). "
        f"Structured garage or underground parking contributes {structured} spots "
        f"({structured_pct} percent). "
        f"Street parking contributes {street} spots ({street_pct} percent). "
        f"Overall parking density classification is {density}. "
        f"{_poi_summary_sentence(poi_summary)} "
        f"{location_phrase}"
    )


async def _ensure_genai_configured() -> None:
    global _GENAI_CONFIGURED
    if genai is None:
        return
    if _GENAI_CONFIGURED:
        return

    api_key = _require_env("GEMINI_API_KEY")

    async with _GENAI_LOCK:
        if _GENAI_CONFIGURED:
            return
        await asyncio.to_thread(genai.configure, api_key=api_key)
        _GENAI_CONFIGURED = True


def _extract_embedding(response: Any) -> list[float]:
    if isinstance(response, dict):
        embedding = response.get("embedding")
        if isinstance(embedding, dict):
            values = embedding.get("values")
            if isinstance(values, list):
                return [float(v) for v in values]
        if isinstance(embedding, list):
            return [float(v) for v in embedding]

        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, dict):
                values = first.get("values")
                if isinstance(values, list):
                    return [float(v) for v in values]

    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        if isinstance(embedding, list):
            return [float(v) for v in embedding]
        values = getattr(embedding, "values", None)
        if values is not None:
            return [float(v) for v in values]

    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return [float(v) for v in values]

    raise RuntimeError("Failed to parse embedding vector from Gemini response.")


def _embed_text_with_rest_sync(text: str, task_type: str) -> list[float]:
    api_key = _require_env("GEMINI_API_KEY")
    params = urllib.parse.urlencode({"key": api_key})
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:embedContent?{params}"
    body = {
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": VECTOR_DIMENSION,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data = _read_json_request(req)
    embedding = data.get("embedding", {})
    values = embedding.get("values") if isinstance(embedding, dict) else None
    if not isinstance(values, list):
        raise RuntimeError("Gemini REST embedding response did not include embedding values.")
    return [float(v) for v in values]


async def embed_text(text: str, task_type: str) -> list[float]:
    """
    Embed text with Gemini text-embedding-004 at 768 dimensions.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Text to embed cannot be empty.")
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. "
            f"Use one of: {sorted(_VALID_TASK_TYPES)}"
        )

    vector: list[float]
    if genai is not None:
        try:
            await _ensure_genai_configured()

            def _embed_sync() -> Any:
                return genai.embed_content(
                    model=EMBED_MODEL,
                    content=clean_text,
                    task_type=task_type,
                    output_dimensionality=VECTOR_DIMENSION,
                )

            response = await asyncio.to_thread(_embed_sync)
            vector = _extract_embedding(response)
        except Exception as exc:
            logger.warning(
                "google-generativeai embedding call failed; falling back to REST API: %s",
                exc,
            )
            vector = await asyncio.to_thread(_embed_text_with_rest_sync, clean_text, task_type)
    else:
        vector = await asyncio.to_thread(_embed_text_with_rest_sync, clean_text, task_type)

    if len(vector) != VECTOR_DIMENSION:
        raise RuntimeError(
            f"Expected embedding dimension {VECTOR_DIMENSION}, got {len(vector)}."
        )
    return vector


def _read_json_request(req: urllib.request.Request | str) -> dict[str, Any]:
    with urllib.request.urlopen(req, timeout=12) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _places_search_new(
    api_key: str, lat: float, lon: float, radius_m: int, place_type: str
) -> set[str]:
    body = {
        "includedTypes": [place_type],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius_m),
            }
        },
    }
    req = urllib.request.Request(
        PLACES_NEW_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.id",
        },
        method="POST",
    )
    data = _read_json_request(req)
    places = data.get("places", [])
    if not isinstance(places, list):
        return set()
    return {p.get("id") for p in places if isinstance(p, dict) and p.get("id")}


def _places_search_legacy(
    api_key: str, lat: float, lon: float, radius_m: int, place_type: str
) -> set[str]:
    params = urllib.parse.urlencode(
        {
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "type": place_type,
            "key": api_key,
        }
    )
    req = f"{PLACES_LEGACY_URL}?{params}"
    data = _read_json_request(req)
    status = data.get("status")
    if status not in {"OK", "ZERO_RESULTS"}:
        raise RuntimeError(f"Legacy Places API status: {status}")
    places = data.get("results", [])
    if not isinstance(places, list):
        return set()
    return {p.get("place_id") for p in places if isinstance(p, dict) and p.get("place_id")}


def _fetch_poi_counts_sync(lat: float, lon: float, radius_m: int) -> dict[str, int]:
    api_key = _require_env("GOOGLE_MAPS_API_KEY")
    summary: dict[str, int] = {}

    for category, types in _POI_CATEGORY_TYPES.items():
        category_ids: set[str] = set()
        for place_type in types:
            ids: set[str]
            try:
                ids = _places_search_new(api_key, lat, lon, radius_m, place_type)
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError):
                # Graceful fallback for projects still configured for legacy Places.
                try:
                    ids = _places_search_legacy(api_key, lat, lon, radius_m, place_type)
                except Exception:
                    ids = set()
            category_ids.update(ids)
        summary[category] = len(category_ids)

    summary["total_pois"] = (
        summary.get("restaurants", 0)
        + summary.get("retail_stores", 0)
        + summary.get("offices", 0)
        + summary.get("transit_stops", 0)
    )
    return summary


async def fetch_nearby_poi_counts(
    lat: float, lon: float, radius_m: int = PLACES_RADIUS_METERS
) -> dict[str, int]:
    cache_key = (round(float(lat), 5), round(float(lon), 5))

    async with _POI_CACHE_LOCK:
        cached = _POI_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    counts = await asyncio.to_thread(_fetch_poi_counts_sync, float(lat), float(lon), int(radius_m))
    async with _POI_CACHE_LOCK:
        _POI_CACHE[cache_key] = dict(counts)
    return counts


def _vector_id_for_hex(hex_id: str) -> int:
    try:
        return int(hex_id, 16)
    except ValueError:
        # fallback deterministic id if hex_id isn't parseable as hex
        return int(zlib.crc32(hex_id.encode("utf-8")))


def _payload_for_cell(
    hex_cell: dict[str, Any],
    profile_text: str,
    poi_summary: dict[str, int],
) -> dict[str, Any]:
    hex_id = str(hex_cell.get("hex_id", "unknown"))
    lat, lon = _extract_centroid(hex_cell)
    surface = _safe_int(hex_cell.get("surface"))
    structured = _safe_int(hex_cell.get("structured"))
    street = _safe_int(hex_cell.get("street"))
    total = _safe_int(hex_cell.get("total"), surface + structured + street)
    if total <= 0:
        total = surface + structured + street

    return {
        "hex_id": hex_id,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "total_spots": total,
        "surface_spots": surface,
        "structured_spots": structured,
        "street_spots": street,
        "surface_pct": _pct(surface, total),
        "structured_pct": _pct(structured, total),
        "street_pct": _pct(street, total),
        "density_class": _density_class(total),
        "poi_summary": poi_summary,
        "geometry": hex_cell.get("geometry"),
        "profile_text": profile_text,
    }


async def _ensure_collection(client: Any) -> None:
    has_collection = await client.has_collection(VECTOR_COLLECTION)
    if has_collection:
        return

    metric = getattr(DistanceMetric, "COSINE", "COSINE") if DistanceMetric is not None else "COSINE"
    await client.create_collection(
        name=VECTOR_COLLECTION,
        dimension=VECTOR_DIMENSION,
        distance_metric=metric,
    )


async def _flush_collection_if_supported(client: Any) -> None:
    flush = getattr(client, "flush", None)
    if not callable(flush):
        return
    try:
        await flush(collection_name=VECTOR_COLLECTION)
    except TypeError:
        await flush(VECTOR_COLLECTION)
    except Exception as exc:  # pragma: no cover
        logger.warning("Vector DB flush failed for %s: %s", VECTOR_COLLECTION, exc)


async def index_hex_cells(
    hex_cells: list[dict[str, Any]],
    batch_size: int = 24,
    location_hint: str = "",
) -> int:
    """
    Embed and index macro hex cells into Actian VectorAI DB.
    """
    _require_dependencies()
    _require_env("GEMINI_API_KEY")
    _require_env("GOOGLE_MAPS_API_KEY")

    if not hex_cells:
        return 0

    ids: list[int] = []
    vectors: list[list[float]] = []
    payloads: list[dict[str, Any]] = []

    for hex_cell in hex_cells:
        if not isinstance(hex_cell, dict):
            continue
        hex_id = str(hex_cell.get("hex_id", "")).strip()
        if not hex_id:
            continue

        lat, lon = _extract_centroid(hex_cell)
        poi_summary: dict[str, int]
        if lat is None or lon is None:
            poi_summary = {"restaurants": 0, "retail_stores": 0, "offices": 0, "transit_stops": 0, "total_pois": 0}
        else:
            poi_summary = await fetch_nearby_poi_counts(lat, lon)

        # Build location context: combine user-provided hint with reverse geocode
        geo_address = ""
        if lat is not None and lon is not None:
            geo_address = await reverse_geocode_cached(lat, lon)
        location_context = location_hint
        if geo_address:
            location_context = f"{location_hint}. {geo_address}" if location_hint else geo_address

        profile_text = parking_profile_to_text(hex_cell, poi_summary, location_context=location_context)
        vector = await embed_text(profile_text, task_type="RETRIEVAL_DOCUMENT")

        ids.append(_vector_id_for_hex(hex_id))
        vectors.append(vector)
        payloads.append(_payload_for_cell(hex_cell, profile_text, poi_summary))

    if not ids:
        return 0

    async with AsyncCortexClient(_vector_db_host()) as client:
        await client.health_check()
        await _ensure_collection(client)

        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            await client.batch_upsert(
                collection_name=VECTOR_COLLECTION,
                ids=ids[start:stop],
                vectors=vectors[start:stop],
                payloads=payloads[start:stop],
            )
        await _flush_collection_if_supported(client)

    return len(ids)


def _payload_matches_filters(
    payload: dict[str, Any],
    min_spots: int | None,
    require_garage: bool,
    require_street: bool,
) -> bool:
    total_spots = _safe_int(payload.get("total_spots"))
    structured_spots = _safe_int(payload.get("structured_spots"))
    street_spots = _safe_int(payload.get("street_spots"))

    if min_spots is not None and total_spots < min_spots:
        return False
    if require_garage and structured_spots <= 0:
        return False
    if require_street and street_spots <= 0:
        return False
    return True


def _unpack_search_result(result: Any) -> tuple[float, dict[str, Any]]:
    if isinstance(result, dict):
        score = float(result.get("score", 0.0))
        payload = result.get("payload") or {}
        return score, payload if isinstance(payload, dict) else {}

    score = float(getattr(result, "score", 0.0))
    payload = getattr(result, "payload", {})
    return score, payload if isinstance(payload, dict) else {}


async def semantic_search(
    query: str,
    top_k: int = 10,
    min_spots: int | None = None,
    require_garage: bool = False,
    require_street: bool = False,
) -> list[dict[str, Any]]:
    """
    Perform semantic search over indexed hex-cell parking profiles.
    """
    _require_dependencies()
    clean_query = (query or "").strip()
    if not clean_query:
        return []

    query_vector = await embed_text(clean_query, task_type="RETRIEVAL_QUERY")
    target_k = max(1, int(top_k))
    candidate_k = target_k if not any([min_spots is not None, require_garage, require_street]) else min(target_k * 6, 250)

    async with AsyncCortexClient(_vector_db_host()) as client:
        await client.health_check()
        if not await client.has_collection(VECTOR_COLLECTION):
            return []

        indexed_count = _safe_int(await client.count(VECTOR_COLLECTION))
        if indexed_count <= 0:
            return []

        results = await client.search(
            collection_name=VECTOR_COLLECTION,
            query=query_vector,
            top_k=min(candidate_k, indexed_count),
        )

    ranked: list[dict[str, Any]] = []
    for result in results:
        score, payload = _unpack_search_result(result)
        if not payload:
            continue
        if not _payload_matches_filters(
            payload=payload,
            min_spots=min_spots,
            require_garage=require_garage,
            require_street=require_street,
        ):
            continue

        lat = payload.get("centroid_lat")
        lon = payload.get("centroid_lon")
        ranked.append(
            {
                "hex_id": payload.get("hex_id"),
                "score": score,
                "lat": lat,
                "lon": lon,
                "centroid": [lat, lon],
                "total": _safe_int(payload.get("total_spots")),
                "surface": _safe_int(payload.get("surface_spots")),
                "structured": _safe_int(payload.get("structured_spots")),
                "street": _safe_int(payload.get("street_spots")),
                "density_class": payload.get("density_class"),
                "poi_summary": payload.get("poi_summary", {}),
                "geometry": payload.get("geometry"),
            }
        )
        if len(ranked) >= target_k:
            break

    return ranked


async def get_vector_index_status() -> dict[str, Any]:
    """
    Check VectorAI DB health and indexed hex count.
    """
    status: dict[str, Any] = {
        "db_ready": False,
        "vector_db_host": _vector_db_host(),
        "collection": VECTOR_COLLECTION,
        "indexed_count": 0,
    }

    if AsyncCortexClient is None:
        status["error"] = (
            "Actian client missing. Install the actiancortex wheel from "
            "https://github.com/hackmamba-io/actian-vectorAI-db-beta."
        )
        return status

    try:
        async with AsyncCortexClient(_vector_db_host()) as client:
            health = await client.health_check()
            status["db_ready"] = True
            if isinstance(health, (list, tuple)) and len(health) >= 2:
                status["server_version"] = health[0]
                status["uptime_seconds"] = health[1]

            has_collection = await client.has_collection(VECTOR_COLLECTION)
            status["collection_exists"] = bool(has_collection)
            if has_collection:
                status["indexed_count"] = _safe_int(await client.count(VECTOR_COLLECTION))
    except Exception as exc:
        status["error"] = str(exc)

    return status

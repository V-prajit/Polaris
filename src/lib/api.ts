/**
 * Polaris API client — connects the Next.js frontend to the FastAPI backend.
 * Zero backend changes needed — calls existing /api/estimate directly,
 * and Nominatim for reverse geocoding (browser-side, CORS-friendly).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Types matching /api/estimate response ───────────────────────────

export interface ConfidenceBand {
    value: number;
    low: number;
    high: number;
    method: string;
}

export interface UtilizationBand {
    rate: number;
    low: number;
    high: number;
}

export interface ParkingFeature {
    name: string;
    type: "surface" | "garage" | "underground" | "street";
    count: number;
    count_area?: number;
    count_yolo?: number;
    count_segformer?: number;
    spots: ConfidenceBand;
    cars?: ConfidenceBand;
    utilization?: UtilizationBand;
    centroid: [number, number]; // [lat, lon]
    geometry: GeoJSON.Geometry;
    levels?: number;
    floor_area_m2?: number;
    length_m?: number;
    sides?: number;
}

export interface EstimateResponse {
    lat: number;
    lon: number;
    radius: number;
    surface: { total: number; features: ParkingFeature[] };
    structured: { total: number; features: ParkingFeature[] };
    street: { total: number; features: ParkingFeature[] };
    grand_total: number;
    spots: ConfidenceBand;
    cars: ConfidenceBand;
    utilization: UtilizationBand;
    elapsed_seconds: number;
}

export interface GeocodeResult {
    name: string;
    address: string;
}

// ─── API Functions ───────────────────────────────────────────────────

export async function fetchEstimate(
    lat: number,
    lon: number,
    radius: number = 300
): Promise<EstimateResponse> {
    // 1. Attempt to load from lightning-fast static cache
    try {
        const cacheName = `${lat.toFixed(3)}_${lon.toFixed(3)}_${radius}.json`;
        const cacheUrl = `/precomputed/${cacheName}`;
        const cacheRes = await fetch(cacheUrl);
        if (cacheRes.ok) {
            console.log(`[ParkSight] Serving INSTANT cache: ${cacheUrl}`);
            return await cacheRes.json();
        }
    } catch (e) {
        // Ignore and fall back to live backend
    }

    // 2. Fall back to live API
    const url = `${API_BASE}/api/estimate?lat=${lat}&lon=${lon}&radius=${radius}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(60_000) });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`API error ${res.status}: ${text || res.statusText}`);
    }
    return res.json();
}

export async function reverseGeocode(
    lat: number,
    lon: number
): Promise<GeocodeResult> {
    try {
        // Call Nominatim directly from the browser (CORS-friendly, no backend needed)
        const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&zoom=18`;
        const res = await fetch(url, {
            signal: AbortSignal.timeout(5_000),
        });
        if (!res.ok) throw new Error(`Nominatim ${res.status}`);
        const data = await res.json();
        const addr = data.address || {};
        const name =
            addr.building ||
            addr.amenity ||
            addr.shop ||
            addr.leisure ||
            data.name ||
            addr.road ||
            `Location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
        const display = data.display_name || `${lat}, ${lon}`;
        return { name, address: display };
    } catch {
        return {
            name: `Location (${lat.toFixed(4)}, ${lon.toFixed(4)})`,
            address: `${lat.toFixed(5)}, ${lon.toFixed(5)}`,
        };
    }
}

// ─── Data Transformations ────────────────────────────────────────────

/**
 * Convert an EstimateResponse into a GeoJSON FeatureCollection
 * that MapView and chart components can consume.
 */
export function apiResponseToGeoJSON(
    data: EstimateResponse
): GeoJSON.FeatureCollection {
    const allFeatures = [
        ...data.surface.features,
        ...data.structured.features,
        ...data.street.features,
    ];

    const features: GeoJSON.Feature[] = allFeatures.map((f) => ({
        type: "Feature" as const,
        properties: {
            name: f.name,
            featureType: f.type,
            count: f.count,
            count_area: f.count_area,
            count_yolo: f.count_yolo,
            count_segformer: f.count_segformer,
            spots_low: f.spots?.low,
            spots_high: f.spots?.high,
            spots_method: f.spots?.method,
            cars: f.cars?.value ?? 0,
            // Map to legacy property names so existing MapView tooltip logic works
            amenity: f.type !== "street" ? "parking" : undefined,
            parking: f.type,
            highway: f.type === "street" ? "street" : undefined,
            building:
                f.type === "garage" || f.type === "underground" ? f.type : undefined,
            levels: f.levels,
            floor_area_m2: f.floor_area_m2,
            length_m: f.length_m,
            sides: f.sides,
        },
        geometry: f.geometry,
    }));

    return { type: "FeatureCollection", features };
}

/**
 * Build metadata object from EstimateResponse (replaces parking_metadata.json).
 */
export function apiResponseToMetadata(data: EstimateResponse) {
    const totalFeatures =
        data.surface.features.length +
        data.structured.features.length +
        data.street.features.length;

    return {
        center: [data.lat, data.lon] as [number, number],
        radius: data.radius,
        total_stalls: data.grand_total,
        features_count: totalFeatures,
    };
}

/**
 * Build stall breakdown items from the API response.
 */
export function buildStallItems(data: EstimateResponse) {
    return [
        ...data.surface.features.map((f) => ({
            name: f.name,
            count: f.count,
            type: "parking" as const,
        })),
        ...data.structured.features.map((f) => ({
            name: f.name,
            count: f.count,
            type: "parking" as const,
        })),
        ...data.street.features.map((f) => ({
            name: f.name,
            count: f.count,
            type: "road" as const,
        })),
    ];
}

/**
 * Compute a 0-100 parking score from the API response.
 */
export function computePolarisScore(data: EstimateResponse): number {
    let score = 50;

    const totalFeatures =
        data.surface.features.length +
        data.structured.features.length +
        data.street.features.length;

    if (totalFeatures > 0) {
        const density = data.grand_total / totalFeatures;
        if (density > 20) score += 15;
        else if (density > 10) score += 8;
    }

    if (data.structured.total > 0) score += 10;
    if (data.street.total > 0) score += 5;

    if (data.utilization && data.utilization.rate > 0) {
        if (data.utilization.rate > 0.8) score -= 10;
        else if (data.utilization.rate > 0.6) score -= 5;
    }

    const categories = [
        data.surface.features.length > 0,
        data.structured.features.length > 0,
        data.street.features.length > 0,
    ].filter(Boolean).length;
    score += categories * 3;

    return Math.max(0, Math.min(score, 100));
}

// ─── Types matching /api/polaris/search response ─────────────────────

export interface PolarisResult {
    hex_id: string;
    score: number;
    lat: number;
    lon: number;
    centroid: [number, number];
    total: number;
    surface: number;
    structured: number;
    street: number;
    density_class: string;
    poi_summary: {
        restaurants: number;
        retail_stores: number;
        offices: number;
        transit_stops: number;
        total_pois: number;
    };
    geometry: GeoJSON.Geometry;
}

export interface PolarisSearchResponse {
    status: string;
    query: string;
    top_k: number;
    filters: {
        min_spots: number | null;
        require_garage: boolean;
        require_street: boolean;
    };
    result_count: number;
    results: PolarisResult[];
}

export interface PolarisSearchFilters {
    top_k?: number;
    min_spots?: number;
    require_garage?: boolean;
    require_street?: boolean;
}

// ─── API Functions ────────────────────────────────────────────────────

/**
 * Semantic search over parking hex cells.
 * Calls the local Next.js proxy at /api/polaris/search,
 * which forwards the request to the GPU backend.
 */
export async function searchPolaris(
    query: string,
    filters?: PolarisSearchFilters
): Promise<PolarisSearchResponse> {
    const res = await fetch("/api/polaris/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, ...filters }),
        signal: AbortSignal.timeout(60_000),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`Polaris search error ${res.status}: ${text || res.statusText}`);
    }
    return res.json();
}

/**
 * Trigger a rebuild of the Polaris search index on the backend.
 * This can take several minutes; the 300s proxy timeout accommodates that.
 */
export async function indexPolaris(): Promise<unknown> {
    const res = await fetch("/api/polaris/index", {
        method: "POST",
        signal: AbortSignal.timeout(300_000),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`Polaris index error ${res.status}: ${text || res.statusText}`);
    }
    return res.json();
}

/**
 * Health-check the Polaris backend.
 * Returns quickly (10s timeout); throws if the backend is unreachable.
 */
export async function getPolarisStatus(): Promise<unknown> {
    const res = await fetch("/api/polaris/status", {
        signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`Polaris status error ${res.status}: ${text || res.statusText}`);
    }
    return res.json();
}

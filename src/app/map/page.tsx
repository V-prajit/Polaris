"use client";

import { useEffect, useState, useMemo, useCallback, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import {
  ArrowLeft,
  Loader2,
  AlertTriangle,
  Search,
  MapPin,
  Building2,
  Car,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ParkingScoreGauge } from "@/components/dashboard/ParkingScoreGauge";
import { StallBreakdownChart } from "@/components/dashboard/StallBreakdownChart";
import { TrafficAnalysis } from "@/components/dashboard/TrafficAnalysis";
import { MapLayersPanel } from "@/components/dashboard/MapLayersPanel";
import {
  fetchEstimate,
  reverseGeocode,
  forwardGeocode,
  apiResponseToGeoJSON,
  apiResponseToMetadata,
  buildStallItems,
  computePolarisScore,
  extractOverlayData,
  type EstimateResponse,
  type GeocodeResult,
} from "@/lib/api";
import { searchPolaris, type PolarisResult } from "@/lib/polaris";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

// ─── Compact metric pill ──────────────────────────────────────────────
function MetricPill({ value, label, suffix = "" }: { value: number; label: string; suffix?: string }) {
  return (
    <div className="flex-1 flex flex-col items-center py-3 px-2 rounded-xl bg-secondary/50 border border-border/60">
      <span className="text-[17px] font-bold tracking-tight tabular-nums">{value}{suffix}</span>
      <span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground mt-0.5">{label}</span>
    </div>
  );
}

// ─── Dashboard Content ────────────────────────────────────────────────
function MapDashboardContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const lat = searchParams.get("lat") || "33.8025746";
  const lng = searchParams.get("lng") || "-84.4106416";

  const [mounted, setMounted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiData, setApiData] = useState<EstimateResponse | null>(null);
  const [locationInfo, setLocationInfo] = useState<GeocodeResult>({ name: "Loading...", address: "" });
  const [layers, setLayers] = useState<Record<string, boolean>>({
    parking: true, roads: true, radius: true, satellite: true, labels: true,
    model_area: true, model_yolo: false, model_segformer: false,
  });

  // Search bar state
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Tab + Polaris search state
  const [activeTab, setActiveTab] = useState<"analysis" | "polaris">("analysis");
  const [polarisQuery, setPolarisQuery] = useState("");
  const [polarisResults, setPolarisResults] = useState<PolarisResult[] | null>(null);
  const [polarisLoading, setPolarisLoading] = useState(false);
  const [polarisError, setPolarisError] = useState<string | null>(null);
  const [polarisMinSpots, setPolarisMinSpots] = useState<number>(0);
  const [polarisRequireGarage, setPolarisRequireGarage] = useState(false);
  const [polarisRequireStreet, setPolarisRequireStreet] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  // Load data whenever lat/lng changes
  useEffect(() => {
    if (!mounted) return;
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const [estimateData, geocodeData] = await Promise.all([
          fetchEstimate(Number(lat), Number(lng)),
          reverseGeocode(Number(lat), Number(lng)),
        ]);
        if (cancelled) return;
        setApiData(estimateData);
        setLocationInfo(geocodeData);
      } catch (err: any) {
        if (cancelled) return;
        const msg = err.message || "Unknown error";
        setError(
          msg.includes("Failed to fetch") || msg.includes("NetworkError")
            ? "Cannot reach backend at localhost:8000. Make sure the backend server is running."
            : `Backend error: ${msg}`
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();
    return () => { cancelled = true; };
  }, [mounted, lat, lng]);

  // Search handler — accepts "lat, lng" or address string
  const handleSearch = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const q = searchQuery.trim();
    if (!q) return;
    setSearchError(null);
    setSearching(true);

    try {
      // Detect lat,lng pattern
      const latLngMatch = q.match(/^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$/);
      if (latLngMatch) {
        router.push(`/map?lat=${latLngMatch[1]}&lng=${latLngMatch[2]}`);
      } else {
        const result = await forwardGeocode(q);
        router.push(`/map?lat=${result.lat}&lng=${result.lon}`);
      }
      setSearchQuery("");
    } catch (err: any) {
      setSearchError(err.message || "Location not found");
    } finally {
      setSearching(false);
    }
  }, [searchQuery, router]);

  // Polaris search handler
  const handlePolarisSearch = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!polarisQuery.trim()) return;
    setPolarisError(null);
    setPolarisLoading(true);
    try {
      const filters = {
        min_spots: polarisMinSpots > 0 ? polarisMinSpots : undefined,
        require_garage: polarisRequireGarage || undefined,
        require_street: polarisRequireStreet || undefined,
      };
      const response = await searchPolaris(polarisQuery, filters);
      setPolarisResults(response.results);
    } catch (err: any) {
      setPolarisError(err.message || "Search failed");
    } finally {
      setPolarisLoading(false);
    }
  }, [polarisQuery, polarisMinSpots, polarisRequireGarage, polarisRequireStreet]);

  // Derived data
  const geojsonData = useMemo(() => apiData ? apiResponseToGeoJSON(apiData) : null, [apiData]);
  const metadata = useMemo(
    () => apiData
      ? apiResponseToMetadata(apiData)
      : { center: [Number(lat), Number(lng)] as [number, number], radius: 300, total_stalls: 0, features_count: 0 },
    [apiData, lat, lng]
  );
  const stallItems = useMemo(() => apiData ? buildStallItems(apiData) : [], [apiData]);
  const polarisScore = useMemo(() => apiData ? computePolarisScore(apiData) : 0, [apiData]);
  const parkingCount = useMemo(
    () => apiData ? apiData.surface.features.length + apiData.structured.features.length : 0,
    [apiData]
  );

  // ML overlay collections — extracted from surface features when ML data is present
  const overlayData = useMemo(() => apiData ? extractOverlayData(apiData) : null, [apiData]);
  const carBoxes = overlayData?.carBoxes ?? null;
  const spotBoxes = overlayData?.spotBoxes ?? null;
  const segformerMasks = overlayData?.segformerMasks ?? null;

  // Convert Polaris results to a GeoJSON FeatureCollection for the map
  const polarisGeoJSON = useMemo((): GeoJSON.FeatureCollection | null => {
    if (!polarisResults || polarisResults.length === 0) return null;
    const features: GeoJSON.Feature[] = polarisResults
      .filter((r) => r.geometry != null)
      .map((r) => ({
        type: "Feature" as const,
        properties: {
          hex_id: r.hex_id,
          total_spots: r.total,
          density_class: r.density_class,
          poi_summary: r.poi_summary ? `${r.poi_summary.total_pois} POIs nearby` : null,
        },
        geometry: r.geometry as GeoJSON.Geometry,
      }));
    return { type: "FeatureCollection", features };
  }, [polarisResults]);

  const trafficData = useMemo(() => {
    if (!geojsonData) return { avgLanes: 0, maxLanes: 0, speedLimits: [], surfaces: [], litPercentage: 0 };
    const roads = geojsonData.features.filter((f: any) => f.properties?.highway);
    const lanes = roads.map((f: any) => parseInt(f.properties?.lanes)).filter((n: number) => !isNaN(n));
    const speeds = [...new Set(roads.map((f: any) => f.properties?.maxspeed).filter(Boolean) as string[])];
    const surfaces = [...new Set(roads.map((f: any) => f.properties?.surface).filter(Boolean) as string[])];
    const litCount = roads.filter((f: any) => f.properties?.lit === "yes").length;
    return {
      avgLanes: lanes.length > 0 ? lanes.reduce((a: number, b: number) => a + b, 0) / lanes.length : 0,
      maxLanes: lanes.length > 0 ? Math.max(...lanes) : 0,
      speedLimits: speeds,
      surfaces,
      litPercentage: roads.length > 0 ? Math.round((litCount / roads.length) * 100) : 0,
    };
  }, [geojsonData]);

  const handleLayerToggle = useCallback((id: string, visible: boolean) => {
    setLayers((prev) => ({ ...prev, [id]: visible }));
  }, []);


  if (!mounted) return null;

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <span className="text-sm text-muted-foreground animate-pulse">Analyzing parking infrastructure...</span>
          <span className="text-[10px] text-muted-foreground/50 font-mono">{Number(lat).toFixed(5)}, {Number(lng).toFixed(5)}</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 max-w-md text-center">
          <AlertTriangle className="h-8 w-8 text-destructive" />
          <span className="text-sm text-muted-foreground">{error}</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => {
              setError(null); setLoading(true);
              fetchEstimate(Number(lat), Number(lng)).then((d) => { setApiData(d); setLoading(false); }).catch((e) => { setError(e.message); setLoading(false); });
              reverseGeocode(Number(lat), Number(lng)).then(setLocationInfo);
            }}>Retry</Button>
            <Link href="/"><Button variant="outline" size="sm"><ArrowLeft className="w-4 h-4 mr-2" />Go Back</Button></Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden relative">
      {/* ─── Map full bleed ─── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2, delay: 0.1 }}
        className="absolute inset-0 z-0"
      >
        <MapView
          lat={Number(lat)}
          lng={Number(lng)}
          radius={metadata.radius}
          layers={layers}
          geojsonData={geojsonData}
          carBoxes={carBoxes}
          spotBoxes={spotBoxes}
          segformerMasks={segformerMasks}
          polarisResults={polarisGeoJSON}
        />
      </motion.div>

      {/* ─── Left Panel ─── */}
      <motion.aside
        initial={{ x: -380, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 260, damping: 28, delay: 0.1 }}
        className="absolute left-0 top-0 bottom-0 w-[340px] z-20 bg-background border-r border-border flex flex-col overflow-hidden"
      >
        {/* ── Search bar ── */}
        <div className="px-4 pt-4 pb-3 border-b border-border/40">
          <form onSubmit={handleSearch} className="flex gap-2">
            <Input
              placeholder="Search address or lat, lng…"
              value={searchQuery}
              onChange={(e) => { setSearchQuery(e.target.value); setSearchError(null); }}
              className="flex-1 h-9 text-sm bg-background"
            />
            <Button type="submit" size="icon" className="h-9 w-9 flex-shrink-0" disabled={!searchQuery.trim() || searching}>
              {searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            </Button>
          </form>
          {searchError && (
            <p className="text-[11px] text-destructive mt-1.5">{searchError}</p>
          )}
        </div>

        {/* ── Tab switcher ── */}
        <div className="flex border-b border-border/40 flex-shrink-0">
          <button
            onClick={() => setActiveTab("analysis")}
            className={`flex-1 py-2 text-xs font-semibold transition-colors ${
              activeTab === "analysis"
                ? "text-foreground border-b-2 border-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Analysis
          </button>
          <button
            onClick={() => setActiveTab("polaris")}
            className={`flex-1 py-2 text-xs font-semibold transition-colors ${
              activeTab === "polaris"
                ? "text-foreground border-b-2 border-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Polaris Search
          </button>
        </div>

        {/* ── Location header ── */}
        <div className="px-4 pt-3 pb-3 border-b border-border/40">
          <div className="flex items-center gap-2.5 mb-1.5">
            <Link href="/">
              <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full bg-secondary/60 hover:bg-secondary flex-shrink-0">
                <ArrowLeft className="h-3.5 w-3.5" />
              </Button>
            </Link>
            <h1
              className="text-[17px] font-bold tracking-tight truncate"
              style={{
                background: "linear-gradient(180deg, hsl(var(--foreground)) 20%, hsl(var(--muted-foreground)) 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              {locationInfo.name}
            </h1>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <MapPin className="w-3 h-3 flex-shrink-0" />
            <span className="truncate">{locationInfo.address || `${Number(lat).toFixed(5)}, ${Number(lng).toFixed(5)}`}</span>
          </div>
        </div>

        {/* ── Scrollable content ── */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">

          {/* ── ANALYSIS TAB ── */}
          {activeTab === "analysis" && (
            <>
              {/* Polaris Score Gauge */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="flex flex-col items-center py-5 border-b border-border/40"
              >
                <ParkingScoreGauge score={polarisScore} label="Polaris Score" delay={0.35} />
              </motion.div>

              {/* Compact metrics strip */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="flex gap-2 px-4 py-4 border-b border-border/40"
              >
                <MetricPill value={metadata.total_stalls} label="Stalls" />
                <MetricPill value={metadata.features_count} label="Features" />
                <MetricPill value={metadata.radius} label="Radius" suffix="m" />
                <MetricPill value={parkingCount} label="Structures" />
              </motion.div>

              {/* Stall Breakdown Chart */}
              {stallItems.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                  className="px-4 py-4 border-b border-border/40"
                >
                  <StallBreakdownChart items={stallItems} delay={0.85} />
                </motion.div>
              )}

              {/* Traffic Analysis */}
              {(trafficData.avgLanes > 0 || trafficData.surfaces.length > 0) && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                  className="px-4 py-4 border-b border-border/40"
                >
                  <TrafficAnalysis data={trafficData} delay={1.15} />
                </motion.div>
              )}

              {/* Map Layers Panel — inline in sidebar */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="px-4 py-4"
              >
                <MapLayersPanel onToggle={handleLayerToggle} delay={1.3} />
              </motion.div>
            </>
          )}

          {/* ── POLARIS SEARCH TAB ── */}
          {activeTab === "polaris" && (
            <>
              {/* Search form */}
              <form onSubmit={handlePolarisSearch} className="px-4 py-4 border-b border-border/40 space-y-3">
                <div className="flex gap-2">
                  <Input
                    placeholder="Find parking near restaurants…"
                    value={polarisQuery}
                    onChange={(e) => { setPolarisQuery(e.target.value); setPolarisError(null); }}
                    className="flex-1 h-9 text-sm bg-background"
                  />
                  <Button
                    type="submit"
                    size="icon"
                    className="h-9 w-9 flex-shrink-0"
                    disabled={!polarisQuery.trim() || polarisLoading}
                  >
                    {polarisLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  </Button>
                </div>

                {/* Filters */}
                <div className="space-y-2 pt-1">
                  <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-muted-foreground/60 mb-1">
                    Filters
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-[11px] text-muted-foreground w-20 flex-shrink-0">Min Spots</label>
                    <Input
                      type="number"
                      min={0}
                      value={polarisMinSpots || ""}
                      onChange={(e) => setPolarisMinSpots(Number(e.target.value))}
                      className="h-7 text-xs w-20"
                      placeholder="0"
                    />
                  </div>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={polarisRequireGarage}
                      onChange={(e) => setPolarisRequireGarage(e.target.checked)}
                      className="w-3.5 h-3.5 accent-primary"
                    />
                    <span className="text-[11px] text-muted-foreground">Require garage</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={polarisRequireStreet}
                      onChange={(e) => setPolarisRequireStreet(e.target.checked)}
                      className="w-3.5 h-3.5 accent-primary"
                    />
                    <span className="text-[11px] text-muted-foreground">Require street parking</span>
                  </label>
                </div>

                {polarisError && (
                  <p className="text-[11px] text-destructive">{polarisError}</p>
                )}
              </form>

              {/* Empty state */}
              {!polarisResults && !polarisLoading && (
                <div className="px-4 py-10 flex flex-col items-center gap-2 text-center">
                  <Search className="w-7 h-7 text-muted-foreground/30" />
                  <p className="text-xs text-muted-foreground/60">
                    Search for areas by describing what you need.
                  </p>
                </div>
              )}

              {/* No results */}
              {polarisResults && polarisResults.length === 0 && (
                <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                  No results found.
                </div>
              )}

              {/* Results list */}
              {polarisResults && polarisResults.length > 0 && (
                <div className="px-4 py-3 space-y-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60 mb-2">
                    {polarisResults.length} result{polarisResults.length !== 1 ? "s" : ""}
                  </div>
                  {polarisResults.map((result) => (
                    <button
                      key={result.hex_id}
                      onClick={() => router.push(`/map?lat=${result.lat}&lng=${result.lon}`)}
                      className="w-full text-left rounded-xl bg-secondary/50 border border-border/60 px-3 py-3 hover:bg-secondary/80 transition-colors"
                    >
                      {/* Header row */}
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-semibold truncate max-w-[180px]">
                          {result.hex_id}
                        </span>
                        {result.density_class && (
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                            result.density_class === "high"
                              ? "bg-green-500/15 text-green-500"
                              : result.density_class === "medium"
                              ? "bg-yellow-500/15 text-yellow-500"
                              : "bg-red-500/15 text-red-500"
                          }`}>
                            {result.density_class}
                          </span>
                        )}
                      </div>

                      {/* Total spots */}
                      <div className="flex items-center gap-1 mb-1">
                        <Car className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                        <span className="text-[11px] text-muted-foreground">
                          {result.total ?? "—"} total spots
                        </span>
                      </div>

                      {/* POI summary */}
                      {result.poi_summary && result.poi_summary.total_pois > 0 && (
                        <div className="flex items-center gap-1 mb-1.5">
                          <Building2 className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                          <span className="text-[11px] text-muted-foreground truncate">
                            {result.poi_summary.total_pois} POIs nearby
                          </span>
                        </div>
                      )}

                      {/* Spot type breakdown bars — surface / structured / street */}
                      {(result.surface > 0 || result.structured > 0 || result.street > 0) && (
                        <div className="mt-1.5 flex gap-0.5 h-1.5 rounded-full overflow-hidden">
                          {result.surface > 0 && (
                            <div
                              className="bg-red-500/60"
                              style={{ flex: result.surface }}
                              title={`Surface: ${result.surface}`}
                            />
                          )}
                          {result.structured > 0 && (
                            <div
                              className="bg-purple-500/60"
                              style={{ flex: result.structured }}
                              title={`Structured: ${result.structured}`}
                            />
                          )}
                          {result.street > 0 && (
                            <div
                              className="bg-amber-500/60"
                              style={{ flex: result.street }}
                              title={`Street: ${result.street}`}
                            />
                          )}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {/* ── Footer ── */}
        <div className="px-5 py-3 border-t border-border/40 flex items-center justify-between flex-shrink-0">
          <span className="text-[9px] font-mono text-muted-foreground/50 uppercase tracking-wider">Powered by GrowthFactor</span>
          <span className="text-[9px] font-mono text-muted-foreground/30">v1.0</span>
        </div>
      </motion.aside>
    </div>
  );
}

export default function MapDashboard() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen w-full items-center justify-center bg-background">
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground animate-pulse">Loading analysis...</span>
          </div>
        </div>
      }
    >
      <MapDashboardContent />
    </Suspense>
  );
}

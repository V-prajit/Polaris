"use client";

import { useEffect, useState, useMemo, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import {
  ArrowLeft,
  Car,
  Scan,
  MapPin,
  Maximize,
  Loader2,
  Sparkles,
  AlertTriangle,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ParkingScoreGauge } from "@/components/dashboard/ParkingScoreGauge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { StallBreakdownChart } from "@/components/dashboard/StallBreakdownChart";
import { TrafficAnalysis } from "@/components/dashboard/TrafficAnalysis";
import { MapLayersPanel } from "@/components/dashboard/MapLayersPanel";
import { InsightCard } from "@/components/dashboard/InsightCard";
import {
  fetchEstimate,
  reverseGeocode,
  apiResponseToGeoJSON,
  apiResponseToMetadata,
  buildStallItems,
  computePolarisScore,
  type EstimateResponse,
  type GeocodeResult,
} from "@/lib/api";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

// ─── Dashboard Content ──────────────────────────────────────────────
function MapDashboardContent() {
  const searchParams = useSearchParams();
  const lat = searchParams.get("lat") || "33.8025746";
  const lng = searchParams.get("lng") || "-84.4106416";

  const [mounted, setMounted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiData, setApiData] = useState<EstimateResponse | null>(null);
  const [locationInfo, setLocationInfo] = useState<GeocodeResult>({
    name: "Loading...",
    address: "",
  });
  const [layers, setLayers] = useState<Record<string, boolean>>({
    parking: true,
    roads: true,
    radius: true,
    satellite: true,
    labels: true,
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch live data from backend + geocode on mount
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
        if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
          setError("Cannot reach backend at localhost:8000. Make sure the backend server is running.");
        } else {
          setError(`Backend error: ${msg}`);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, [mounted, lat, lng]);

  // Derived data from API response
  const geojsonData = useMemo(
    () => (apiData ? apiResponseToGeoJSON(apiData) : null),
    [apiData]
  );
  const metadata = useMemo(
    () =>
      apiData
        ? apiResponseToMetadata(apiData)
        : { center: [Number(lat), Number(lng)] as [number, number], radius: 300, total_stalls: 0, features_count: 0 },
    [apiData, lat, lng]
  );
  const stallItems = useMemo(
    () => (apiData ? buildStallItems(apiData) : []),
    [apiData]
  );
  const polarisScore = useMemo(
    () => (apiData ? computePolarisScore(apiData) : 0),
    [apiData]
  );
  const parkingCount = useMemo(
    () =>
      apiData
        ? apiData.surface.features.length + apiData.structured.features.length
        : 0,
    [apiData]
  );

  // Traffic data extracted from GeoJSON (road features)
  const trafficData = useMemo(() => {
    if (!geojsonData) return { avgLanes: 0, maxLanes: 0, speedLimits: [], surfaces: [], litPercentage: 0 };
    const roads = geojsonData.features.filter((f: any) => f.properties?.highway);
    const lanes = roads
      .map((f: any) => parseInt(f.properties?.lanes))
      .filter((n: number) => !isNaN(n));
    const speeds = [
      ...new Set(
        roads
          .map((f: any) => f.properties?.maxspeed)
          .filter(Boolean) as string[]
      ),
    ];
    const surfaces = [
      ...new Set(
        roads
          .map((f: any) => f.properties?.surface)
          .filter(Boolean) as string[]
      ),
    ];
    const litCount = roads.filter(
      (f: any) => f.properties?.lit === "yes"
    ).length;

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

  const insightText = useMemo(() => {
    if (!apiData) return "";
    const totalFeatures = metadata.features_count;
    const densityPerLot = parkingCount > 0 ? (metadata.total_stalls / parkingCount).toFixed(0) : "0";
    const surfaceInfo = apiData.surface.total > 0 ? `${apiData.surface.total} surface spots` : "";
    const structuredInfo = apiData.structured.total > 0 ? `${apiData.structured.total} structured spots` : "";
    const streetInfo = apiData.street.total > 0 ? `${apiData.street.total} street spots` : "";
    const breakdown = [surfaceInfo, structuredInfo, streetInfo].filter(Boolean).join(", ");

    return `Analysis of ${totalFeatures} features within a ${metadata.radius}m radius identifies ${metadata.total_stalls} estimated parking stalls (${breakdown}), across ${parkingCount} parking structure${parkingCount !== 1 ? "s" : ""}, averaging ${densityPerLot} stalls per structure.${trafficData.litPercentage > 0
      ? ` ${trafficData.litPercentage}% of road segments are lit.`
      : ""
      }${trafficData.surfaces.length > 0
        ? ` Road surfaces are predominantly ${trafficData.surfaces.join(" and ")}.`
        : ""
      } Analysis completed in ${apiData.elapsed_seconds}s.`;
  }, [apiData, metadata, parkingCount, trafficData]);

  if (!mounted) return null;

  // Loading state
  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <span className="text-sm text-muted-foreground animate-pulse">
            Analyzing parking infrastructure...
          </span>
          <span className="text-[10px] text-muted-foreground/50 font-mono">
            {Number(lat).toFixed(5)}, {Number(lng).toFixed(5)}
          </span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 max-w-md text-center">
          <AlertTriangle className="h-8 w-8 text-destructive" />
          <span className="text-sm text-muted-foreground">{error}</span>
          <span className="text-[10px] text-muted-foreground/50 font-mono">
            {Number(lat).toFixed(5)}, {Number(lng).toFixed(5)}
          </span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => {
              setError(null);
              setLoading(true);
              fetchEstimate(Number(lat), Number(lng))
                .then((data) => { setApiData(data); setLoading(false); })
                .catch((err) => { setError(err.message); setLoading(false); });
              reverseGeocode(Number(lat), Number(lng)).then(setLocationInfo);
            }}>
              Retry
            </Button>
            <Link href="/">
              <Button variant="outline" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" /> Go Back
              </Button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden relative">
      {/* ─── Map (full bleed) ─── */}
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
        />
      </motion.div>

      {/* ─── Left Floating Panel ─── */}
      <motion.aside
        initial={{ x: -440, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
        className="absolute left-0 top-0 bottom-0 w-[360px] z-20 bg-background border-r border-border flex flex-col overflow-hidden"

      >
        {/* Header */}
        <div className="p-5 pb-4 border-b border-border/30">
          <div className="flex items-center gap-3 mb-3">
            <Link href="/">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-full bg-secondary/40 hover:bg-secondary/60"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
            </Link>
            <div className="flex-1 min-w-0">
              <h1 className="text-lg font-bold tracking-tight truncate">
                {locationInfo.name}
              </h1>
              <p className="text-[11px] text-muted-foreground truncate">
                {locationInfo.address}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <MapPin className="w-3 h-3" />
            <span className="font-mono tabular-nums">
              {Number(lat).toFixed(5)}, {Number(lng).toFixed(5)}
            </span>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 p-4 space-y-4 overflow-y-auto custom-scrollbar">

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              icon={<Car className="w-4 h-4" />}
              value={metadata.total_stalls}
              label="Est. Stalls"
              delay={0.8}
            />
            <MetricCard
              icon={<Scan className="w-4 h-4" />}
              value={metadata.features_count}
              label="Features"
              delay={0.9}
            />
            <MetricCard
              icon={<Maximize className="w-4 h-4" />}
              value={metadata.radius}
              label="Radius"
              suffix="m"
              delay={1.0}
            />
            <MetricCard
              icon={<Sparkles className="w-4 h-4" />}
              value={parkingCount}
              label="Structures"
              delay={1.1}
            />
          </div>

          {/* Stall Breakdown */}
          {stallItems.length > 0 && (
            <StallBreakdownChart items={stallItems} delay={1.2} />
          )}

          {/* Traffic Analysis */}
          {(trafficData.avgLanes > 0 || trafficData.surfaces.length > 0) && (
            <TrafficAnalysis data={trafficData} delay={1.4} />
          )}

          {/* AI Insight */}
          {insightText && <InsightCard text={insightText} delay={1.6} />}
        </div>

        {/* Bottom bar */}
        <div className="px-5 py-3 border-t border-border/30 flex items-center justify-between">
          <span className="text-[9px] font-mono text-muted-foreground/50 uppercase tracking-wider">
            Powered by GrowthFactor
          </span>
          <span className="text-[9px] font-mono text-muted-foreground/30">
            v1.0
          </span>
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
            <span className="text-sm text-muted-foreground animate-pulse">
              Loading analysis...
            </span>
          </div>
        </div>
      }
    >
      <MapDashboardContent />
    </Suspense>
  );
}
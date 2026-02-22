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
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ParkingScoreGauge } from "@/components/dashboard/ParkingScoreGauge";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { StallBreakdownChart } from "@/components/dashboard/StallBreakdownChart";
import { TrafficAnalysis } from "@/components/dashboard/TrafficAnalysis";
import { MapLayersPanel } from "@/components/dashboard/MapLayersPanel";
import { InsightCard } from "@/components/dashboard/InsightCard";
import parkingMetadata from "../../parking_metadata.json";
import parkingData from "../../parking_data.json";

// Cached geocode data
const locationInfo = {
  name: "Walmart Supercenter",
  address: "1801 Howell Mill Rd NW, Atlanta, GA 30318",
};

const MapView = dynamic(() => import("./MapView"), { ssr: false });

// ─── Data extraction helpers ────────────────────────────────────────
function extractStallItems(features: any[]) {
  return features.map((f: any, i: number) => {
    const isParking = f.properties?.amenity === "parking";
    const name = isParking
      ? `Parking ${f.properties?.parking || "Lot"} #${i + 1}`
      : f.properties?.name || f.properties?.short_name || `Segment #${i + 1}`;
    return {
      name,
      count: f.properties?.count || 0,
      type: (isParking ? "parking" : "road") as "parking" | "road",
    };
  });
}

function extractTrafficData(features: any[]) {
  const roads = features.filter((f: any) => f.properties?.highway);
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
}

function computeScore(metadata: typeof parkingMetadata, traffic: ReturnType<typeof extractTrafficData>): number {
  // Composite score factoring density, lighting, infrastructure
  let score = 50;
  // Stall density bonus
  const density = metadata.total_stalls / metadata.features_count;
  if (density > 20) score += 15;
  else if (density > 10) score += 8;
  // Lighting bonus
  score += Math.round(traffic.litPercentage * 0.15);
  // Infrastructure bonus (more lanes = better access)
  if (traffic.avgLanes > 3) score += 10;
  else if (traffic.avgLanes > 2) score += 5;
  // Surface quality bonus
  if (traffic.surfaces.includes("asphalt") || traffic.surfaces.includes("concrete")) score += 5;
  return Math.min(score, 100);
}

// ─── Dashboard Content ──────────────────────────────────────────────
function MapDashboardContent() {
  const searchParams = useSearchParams();
  const lat = searchParams.get("lat") || String(parkingMetadata.center[0]);
  const lng = searchParams.get("lng") || String(parkingMetadata.center[1]);

  const [mounted, setMounted] = useState(false);
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

  const stallItems = useMemo(
    () => extractStallItems(parkingData.features),
    []
  );
  const trafficData = useMemo(
    () => extractTrafficData(parkingData.features),
    []
  );
  const parkIQScore = useMemo(
    () => computeScore(parkingMetadata, trafficData),
    [trafficData]
  );

  const parkingCount = useMemo(
    () =>
      parkingData.features.filter((f: any) => f.properties?.amenity === "parking").length,
    []
  );

  const handleLayerToggle = useCallback((id: string, visible: boolean) => {
    setLayers((prev) => ({ ...prev, [id]: visible }));
  }, []);

  const insightText = useMemo(() => {
    const densityPerLot = (parkingMetadata.total_stalls / parkingCount).toFixed(0);
    return `Analysis of ${parkingMetadata.features_count} features within a ${parkingMetadata.radius}m radius identifies ${parkingMetadata.total_stalls} estimated parking stalls across ${parkingCount} parking structure${parkingCount > 1 ? "s" : ""}, averaging ${densityPerLot} stalls per structure. ${trafficData.litPercentage}% of road segments are lit, with an average lane count of ${trafficData.avgLanes.toFixed(1)}. Road surfaces are predominantly ${trafficData.surfaces.join(" and ")} with speed limits of ${trafficData.speedLimits.join(", ")}.`;
  }, [parkingCount, trafficData]);

  if (!mounted) return null;

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
          radius={parkingMetadata.radius}
          layers={layers}
        />
      </motion.div>

      {/* ─── Left Floating Panel ─── */}
      <motion.aside
        initial={{ x: -440, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
        className="absolute left-4 top-4 bottom-4 w-[360px] z-20 glass-panel-strong rounded-2xl flex flex-col overflow-hidden"
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
          {/* Score Gauge */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="flex justify-center py-2"
          >
            <ParkingScoreGauge score={parkIQScore} delay={0.7} />
          </motion.div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              icon={<Car className="w-4 h-4" />}
              value={parkingMetadata.total_stalls}
              label="Est. Stalls"
              delay={0.8}
              color="hsl(187, 73%, 46%)"
            />
            <MetricCard
              icon={<Scan className="w-4 h-4" />}
              value={parkingMetadata.features_count}
              label="Features"
              delay={0.9}
              color="hsl(213, 74%, 47%)"
            />
            <MetricCard
              icon={<Maximize className="w-4 h-4" />}
              value={parkingMetadata.radius}
              label="Radius"
              suffix="m"
              delay={1.0}
              color="hsl(142, 71%, 45%)"
            />
            <MetricCard
              icon={<Sparkles className="w-4 h-4" />}
              value={parkingCount}
              label="Structures"
              delay={1.1}
              color="hsl(38, 92%, 50%)"
            />
          </div>

          {/* Stall Breakdown */}
          <StallBreakdownChart items={stallItems} delay={1.2} />

          {/* Traffic Analysis */}
          <TrafficAnalysis data={trafficData} delay={1.4} />

          {/* AI Insight */}
          <InsightCard text={insightText} delay={1.6} />
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

      {/* ─── Right Layers Panel ─── */}
      <div className="absolute right-4 top-4 z-20">
        <MapLayersPanel onToggle={handleLayerToggle} delay={1.0} />
      </div>

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
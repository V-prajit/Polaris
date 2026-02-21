"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import { ArrowLeft, Car, Map as MapIcon, Maximize, Loader2 } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import parkingMetadata from "../../parking_metadata.json";

// Dynamic import for the Map component to avoid SSR issues with Leaflet
const MapView = dynamic(() => import("./MapView"), { ssr: false });

function MapDashboardContent() {
  const searchParams = useSearchParams();
  const lat = searchParams.get("lat") || parkingMetadata.center[0];
  const lng = searchParams.get("lng") || parkingMetadata.center[1];

  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      {/* Sidebar Dashboard */}
      <motion.aside
        initial={{ x: -400 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="w-[400px] h-full bg-card border-r shadow-2xl flex flex-col z-20"
      >
        <div className="p-6 border-b bg-background/50 backdrop-blur-sm">
          <div className="flex items-center gap-4 mb-6">
            <Link href="/">
              <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full">
                <ArrowLeft className="h-4 w-4" />
              </Button>
            </Link>
            <h1 className="text-2xl font-bold tracking-tight">ParkIQ Analysis</h1>
          </div>
          <p className="text-sm text-muted-foreground">
            Coordinates: {Number(lat).toFixed(5)}, {Number(lng).toFixed(5)}
          </p>
        </div>

        <div className="flex-1 p-6 space-y-6 overflow-y-auto">
          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-primary/5 border border-primary/20 rounded-xl p-4 flex flex-col items-center justify-center text-center space-y-2"
            >
              <Car className="h-6 w-6 text-primary" />
              <div className="text-3xl font-bold text-primary">
                {parkingMetadata.total_stalls}
              </div>
              <div className="text-xs font-semibold uppercase tracking-wider text-primary/70">
                Est. Stalls
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-secondary/50 border rounded-xl p-4 flex flex-col items-center justify-center text-center space-y-2"
            >
              <MapIcon className="h-6 w-6 text-muted-foreground" />
              <div className="text-3xl font-bold">
                {parkingMetadata.features_count}
              </div>
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Lot Features
              </div>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-card border rounded-xl p-5 space-y-4"
          >
            <h3 className="font-semibold flex items-center gap-2">
              <Maximize className="h-4 w-4" /> Search Radius
            </h3>
            <div className="text-2xl font-bold">{parkingMetadata.radius}m</div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Analysis was performed using the SegFormer-B4 multi-class
              segmentation model to accurately subtract drive aisles and roads
              from the bounding box areas.
            </p>
          </motion.div>
        </div>
      </motion.aside>

      {/* Main Map Area */}
      <main className="flex-1 relative bg-slate-900">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.2 }}
          className="absolute inset-0 z-0"
        >
          <MapView lat={Number(lat)} lng={Number(lng)} radius={parkingMetadata.radius} />
        </motion.div>
      </main>
    </div>
  );
}

export default function MapDashboard() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen w-full items-center justify-center bg-background">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      }
    >
      <MapDashboardContent />
    </Suspense>
  );
}
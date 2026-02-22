"use client";

import { useMemo, useCallback, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Circle,
  Marker,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix Leaflet's default icon path issues
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

// Clean center marker
const centerIcon = L.divIcon({
  className: "",
  html: `
    <div style="position:relative;width:24px;height:24px;">
      <div style="position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#ef4444;border:3px solid #fff;box-shadow:0 1px 6px rgba(0,0,0,0.35);"></div>
    </div>
  `,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

// ─── Color palette by feature type (GrowthFactor-style) ─────────────
const FEATURE_COLORS: Record<string, { fill: string; stroke: string }> = {
  surface: { fill: "rgba(239, 68, 68, 0.25)", stroke: "#ef4444" },   // red
  garage: { fill: "rgba(168, 85, 247, 0.25)", stroke: "#a855f7" },   // purple
  underground: { fill: "rgba(99, 102, 241, 0.22)", stroke: "#6366f1" },   // indigo
  street: { fill: "transparent", stroke: "#f59e0b" },   // amber
  default: { fill: "rgba(239, 68, 68, 0.20)", stroke: "#ef4444" },
};

function getFeatureColor(featureType: string) {
  return FEATURE_COLORS[featureType] || FEATURE_COLORS.default;
}

interface MapViewProps {
  lat: number;
  lng: number;
  radius: number;
  layers: Record<string, boolean>;
  geojsonData: GeoJSON.FeatureCollection | null;
}

// Helper component to recenter map when props change
function MapUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, map.getZoom());
  }, [center, map]);
  return null;
}

// ─── Per-feature styling (colored by type) ──────────────────────────
function featureStyle(feature: any) {
  const fType = feature?.properties?.featureType || feature?.properties?.parking || "default";
  const isStreet = feature?.properties?.highway || fType === "street";
  const colors = getFeatureColor(isStreet ? "street" : fType);

  if (isStreet) {
    return {
      color: colors.stroke,
      weight: 4,
      fillOpacity: 0,
      dashArray: undefined as string | undefined,
      opacity: 0.8,
    };
  }

  return {
    color: colors.stroke,
    weight: 2,
    fillColor: colors.fill,
    fillOpacity: 0.25,
    dashArray: undefined as string | undefined,
  };
}

function buildTooltipHTML(props: any, isParking: boolean, layers: Record<string, boolean>): string {
  const fType = props.featureType || props.parking || "surface";
  const typeLabel = fType.charAt(0).toUpperCase() + fType.slice(1);
  const accentColor = getFeatureColor(fType).stroke;

  // Determine which count to show based on active toggle
  let displayCount = props.count; // Default fallback
  let displayMethod = "Est. Stalls";

  if (isParking && fType === "surface") {
    if (layers.model_segformer && props.count_segformer !== undefined) {
      displayCount = props.count_segformer;
      displayMethod = "Segformer Est.";
    } else if (layers.model_yolo && props.count_yolo !== undefined) {
      displayCount = props.count_yolo;
      displayMethod = "YOLO V11 Est.";
    } else if (props.count_area !== undefined) {
      displayCount = props.count_area;
      displayMethod = "Math Heuristics";
    }
  }

  if (isParking) {
    return `
      <div class="tooltip-title">🅿️ ${props.name || "Parking Area"}</div>
      <div class="tooltip-row">
        <span class="tooltip-label">Type</span>
        <span class="tooltip-value">${typeLabel}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">${displayMethod}</span>
        <span class="tooltip-value" style="color:${accentColor};font-size:15px;font-weight:700;">${displayCount || "—"}</span>
      </div>
      ${props.spots_low && props.spots_high ? `<div class="tooltip-row"><span class="tooltip-label">Range</span><span class="tooltip-value">${props.spots_low}–${props.spots_high}</span></div>` : ""}
      ${props.cars > 0 ? `<div class="tooltip-row"><span class="tooltip-label">Cars Detected</span><span class="tooltip-value">${props.cars}</span></div>` : ""}
      ${props.levels ? `<div class="tooltip-row"><span class="tooltip-label">Levels</span><span class="tooltip-value">${props.levels}</span></div>` : ""}
      ${props.building ? `<div class="tooltip-row"><span class="tooltip-label">Building</span><span class="tooltip-value">${props.building}</span></div>` : ""}
    `;
  } else {
    return `
      <div class="tooltip-title">🛣️ ${props.name || props.short_name || "Street Segment"}</div>
      <div class="tooltip-row">
        <span class="tooltip-label">Est. Spots</span>
        <span class="tooltip-value" style="color:${accentColor};font-size:15px;font-weight:700;">${props.count || "—"}</span>
      </div>
      ${props.length_m ? `<div class="tooltip-row"><span class="tooltip-label">Length</span><span class="tooltip-value">${Math.round(props.length_m)}m</span></div>` : ""}
      ${props.sides ? `<div class="tooltip-row"><span class="tooltip-label">Sides</span><span class="tooltip-value">${props.sides}</span></div>` : ""}
      ${props.lanes ? `<div class="tooltip-row"><span class="tooltip-label">Lanes</span><span class="tooltip-value">${props.lanes}</span></div>` : ""}
      ${props.maxspeed ? `<div class="tooltip-row"><span class="tooltip-label">Speed</span><span class="tooltip-value">${props.maxspeed}</span></div>` : ""}
      ${props.surface ? `<div class="tooltip-row"><span class="tooltip-label">Surface</span><span class="tooltip-value">${props.surface}</span></div>` : ""}
      ${props.lit === "yes" ? `<div class="tooltip-row"><span class="tooltip-label">Lighting</span><span class="tooltip-value">✓ Lit</span></div>` : ""}
    `;
  }
}

export default function MapView({ lat, lng, radius, layers, geojsonData }: MapViewProps) {
  const data = geojsonData;

  const center = useMemo<[number, number]>(() => [lat, lng], [lat, lng]);

  // Separate parking and road features
  const parkingData = useMemo(() => {
    if (!data) return null;
    return {
      ...data,
      features: data.features.filter(
        (f: any) => f.properties?.amenity === "parking"
      ),
    };
  }, [data]);

  const roadData = useMemo(() => {
    if (!data) return null;
    return {
      ...data,
      features: data.features.filter(
        (f: any) => f.properties?.highway
      ),
    };
  }, [data]);

  const onEachParking = useCallback((feature: any, layer: any) => {
    if (feature.properties) {
      layer.bindTooltip(buildTooltipHTML(feature.properties, true, layers), {
        sticky: true,
        className: "parking-tooltip",
        direction: "top",
        offset: [0, -10],
      });

      const fType = feature.properties.featureType || "surface";
      const colors = getFeatureColor(fType);

      layer.on("mouseover", () => {
        layer.setStyle({
          weight: 3.5,
          fillOpacity: 0.4,
          color: colors.stroke,
        });
      });
      layer.on("mouseout", () => {
        layer.setStyle(featureStyle(feature));
      });
    }
  }, [layers]);

  const onEachRoad = useCallback((feature: any, layer: any) => {
    if (feature.properties) {
      layer.bindTooltip(buildTooltipHTML(feature.properties, false, layers), {
        sticky: true,
        className: "parking-tooltip",
        direction: "top",
        offset: [0, -10],
      });

      layer.on("mouseover", () => {
        layer.setStyle({
          weight: 6,
          opacity: 1,
          color: FEATURE_COLORS.street.stroke,
        });
      });
      layer.on("mouseout", () => {
        layer.setStyle(featureStyle(feature));
      });
    }
  }, [layers]);

  return (
    <div className="h-full w-full relative">
      <style>{`
        .leaflet-container {
          background: #f8f8f8 !important;
          filter: saturate(1.25) contrast(1.075) brightness(1.025);
          -webkit-filter: saturate(1.25) contrast(1.075) brightness(1.025);
        }
        .leaflet-tile-pane {
          image-rendering: -webkit-optimize-contrast;
          image-rendering: crisp-edges;
        }
        .leaflet-tile-container {
          transition: opacity 0.25s ease;
        }
        .leaflet-tile {
          filter: contrast(1.01) brightness(1.005);
          image-rendering: -webkit-optimize-contrast;
        }
      `}</style>

      <MapContainer
        center={center}
        zoom={17}
        style={{ height: "100%", width: "100%" }}
        zoomControl={false}
        attributionControl={false}
        scrollWheelZoom={true}
        zoomSnap={0}
        zoomDelta={0.25}
        wheelDebounceTime={40}
        wheelPxPerZoomLevel={180}
        zoomAnimation={true}
      >
        <MapUpdater center={center} />

        {/* Satellite base */}
        {layers.satellite && (
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="Esri World Imagery"
            maxZoom={20}
            keepBuffer={8}
            updateWhenZooming={false}
            updateWhenIdle={true}
            crossOrigin="anonymous"
          />
        )}

        {/* Street labels overlay */}
        {layers.labels && (
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png"
            attribution="&copy; CartoDB"
            maxZoom={20}
            opacity={0.9}
            keepBuffer={8}
            updateWhenZooming={false}
            updateWhenIdle={true}
            crossOrigin="anonymous"
          />
        )}

        {/* Search Radius — subtle pink fill like GrowthFactor trade area */}
        {layers.radius && (
          <Circle
            center={center}
            radius={radius}
            pathOptions={{
              color: "#ef4444",
              fillColor: "rgba(239, 68, 68, 0.06)",
              fillOpacity: 0.06,
              weight: 1.5,
              dashArray: "8, 5",
              opacity: 0.4,
            }}
          />
        )}

        {/* Center Marker — red dot */}
        <Marker position={center} icon={centerIcon} />

        {/* Parking Zones — colored fills */}
        {layers.parking && parkingData && parkingData.features.length > 0 && (
          <GeoJSON
            key={"parking-" + parkingData.features.length}
            data={parkingData}
            style={featureStyle}
            onEachFeature={onEachParking}
          />
        )}

        {/* Road / Street Segments — amber lines */}
        {layers.roads && roadData && roadData.features.length > 0 && (
          <GeoJSON
            key={"roads-" + roadData.features.length}
            data={roadData}
            style={featureStyle}
            onEachFeature={onEachRoad}
          />
        )}
      </MapContainer>
    </div>
  );
}
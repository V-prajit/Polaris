"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
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
import geojsonData from "../../parking_data.json";

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

// Custom pulsing marker using divIcon
const pulseIcon = L.divIcon({
  className: "",
  html: `
    <div style="position:relative;width:24px;height:24px;">
      <div style="position:absolute;inset:0;border-radius:50%;background:hsl(187,73%,46%);opacity:0.3;animation:pulse-glow 2s ease-in-out infinite;"></div>
      <div style="position:absolute;top:4px;left:4px;width:16px;height:16px;border-radius:50%;background:hsl(187,73%,46%);border:3px solid hsl(222,47%,6%);box-shadow:0 0 12px hsla(187,73%,46%,0.6);"></div>
    </div>
  `,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

interface MapViewProps {
  lat: number;
  lng: number;
  radius: number;
  layers: Record<string, boolean>;
}

// Helper component to recenter map when props change
function MapUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, map.getZoom());
  }, [center, map]);
  return null;
}

// Style functions
function getParkingStyle() {
  return {
    color: "hsl(187, 73%, 64%)",
    weight: 2.5,
    fillColor: "hsla(187, 73%, 46%, 0.25)",
    fillOpacity: 0.25,
    dashArray: undefined as string | undefined,
    className: "parking-zone-glow",
  };
}

function getRoadStyle() {
  return {
    color: "hsl(38, 92%, 55%)",
    weight: 3,
    fillOpacity: 0,
    dashArray: "8, 6",
    opacity: 0.7,
  };
}

function buildTooltipHTML(props: any, isParking: boolean): string {
  if (isParking) {
    return `
      <div class="tooltip-title">🅿️ Parking Structure</div>
      <div class="tooltip-row">
        <span class="tooltip-label">Type</span>
        <span class="tooltip-value">${props.parking || "Surface"}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Est. Stalls</span>
        <span class="tooltip-value" style="color:hsl(187,73%,60%);font-size:15px;">${props.count || "—"}</span>
      </div>
      ${props.building ? `<div class="tooltip-row"><span class="tooltip-label">Building</span><span class="tooltip-value">${props.building}</span></div>` : ""}
    `;
  } else {
    return `
      <div class="tooltip-title">🛣️ ${props.name || props.short_name || "Road Segment"}</div>
      <div class="tooltip-row">
        <span class="tooltip-label">Lanes</span>
        <span class="tooltip-value">${props.lanes || "—"}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Speed</span>
        <span class="tooltip-value">${props.maxspeed || "—"}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Surface</span>
        <span class="tooltip-value">${props.surface || "—"}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Vehicles</span>
        <span class="tooltip-value" style="color:hsl(38,92%,55%);">${props.count || "—"}</span>
      </div>
      ${props.lit === "yes" ? `<div class="tooltip-row"><span class="tooltip-label">Lighting</span><span class="tooltip-value" style="color:hsl(45,93%,58%);">✓ Lit</span></div>` : ""}
    `;
  }
}

export default function MapView({ lat, lng, radius, layers }: MapViewProps) {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    setData(geojsonData);
  }, []);

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
      layer.bindTooltip(buildTooltipHTML(feature.properties, true), {
        sticky: true,
        className: "parking-tooltip",
        direction: "top",
        offset: [0, -10],
      });

      // Hover glow effect
      layer.on("mouseover", () => {
        layer.setStyle({
          weight: 4,
          fillOpacity: 0.4,
          color: "hsl(187, 73%, 75%)",
        });
      });
      layer.on("mouseout", () => {
        layer.setStyle(getParkingStyle());
      });
    }
  }, []);

  const onEachRoad = useCallback((feature: any, layer: any) => {
    if (feature.properties) {
      layer.bindTooltip(buildTooltipHTML(feature.properties, false), {
        sticky: true,
        className: "parking-tooltip",
        direction: "top",
        offset: [0, -10],
      });

      layer.on("mouseover", () => {
        layer.setStyle({
          weight: 5,
          opacity: 1,
          color: "hsl(38, 92%, 65%)",
        });
      });
      layer.on("mouseout", () => {
        layer.setStyle(getRoadStyle());
      });
    }
  }, []);

  return (
    <div className="h-full w-full relative">
      {/* Gaussian blur glow overlay for parking zones */}
      <style>{`
        .parking-zone-glow {
          filter: drop-shadow(0 0 8px hsla(187, 73%, 46%, 0.5));
        }
        .leaflet-container {
          background: hsl(222, 47%, 6%) !important;
        }
      `}</style>

      <MapContainer
        center={center}
        zoom={17}
        style={{ height: "100%", width: "100%" }}
        zoomControl={false}
      >
        <MapUpdater center={center} />

        {/* Dark satellite base */}
        {layers.satellite && (
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="Esri World Imagery"
            maxZoom={20}
          />
        )}

        {/* Dark street labels overlay */}
        {layers.labels && (
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png"
            attribution="&copy; CartoDB"
            maxZoom={20}
            opacity={0.8}
          />
        )}

        {/* Search Radius */}
        {layers.radius && (
          <Circle
            center={center}
            radius={radius}
            pathOptions={{
              color: "hsl(187, 73%, 46%)",
              fillColor: "hsla(187, 73%, 46%, 0.06)",
              fillOpacity: 0.06,
              weight: 1.5,
              dashArray: "6, 4",
              opacity: 0.5,
            }}
          />
        )}

        {/* Center Marker — pulsing cyan dot */}
        <Marker position={center} icon={pulseIcon} />

        {/* Parking Zones */}
        {layers.parking && parkingData && parkingData.features.length > 0 && (
          <GeoJSON
            key="parking"
            data={parkingData}
            style={getParkingStyle}
            onEachFeature={onEachParking}
          />
        )}

        {/* Road Segments */}
        {layers.roads && roadData && roadData.features.length > 0 && (
          <GeoJSON
            key="roads"
            data={roadData}
            style={getRoadStyle}
            onEachFeature={onEachRoad}
          />
        )}
      </MapContainer>
    </div>
  );
}
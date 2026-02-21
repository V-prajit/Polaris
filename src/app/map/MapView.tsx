"use client";

import { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, Circle, Marker, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import geojsonData from "../../parking_data.json";

// Fix Leaflet's default icon path issues
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

// Create a custom red icon for the center marker
const redIcon = new L.Icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

interface MapViewProps {
  lat: number;
  lng: number;
  radius: number;
}

// Helper component to recenter map when props change
function MapUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, map.getZoom());
  }, [center, map]);
  return null;
}

export default function MapView({ lat, lng, radius }: MapViewProps) {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    // In a real app we'd fetch this, but for the hackathon we're importing the generated JSON directly
    setData(geojsonData);
  }, []);

  const center: [number, number] = [lat, lng];

  return (
    <div className="h-full w-full relative">
      <MapContainer
        center={center}
        zoom={17}
        style={{ height: "100%", width: "100%" }}
        zoomControl={false}
      >
        <MapUpdater center={center} />
        
        {/* Esri World Imagery Satellite Tiles */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="Esri World Imagery"
          maxZoom={20}
        />

        {/* Optional: OSM Streets overlay with opacity for context */}
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
          opacity={0.3}
          maxZoom={20}
        />

        {/* Search Radius Circle */}
        <Circle
          center={center}
          radius={radius}
          pathOptions={{
            color: "#1976d2",
            fillColor: "#1976d2",
            fillOpacity: 0.1,
            weight: 2,
          }}
        />

        {/* Center Marker */}
        <Marker position={center} icon={redIcon}>
          <Tooltip>Search Center</Tooltip>
        </Marker>

        {/* Parking Geometries */}
        {data && (
          <GeoJSON
            data={data}
            style={() => ({
              color: "#1976d2",
              weight: 2,
              fillOpacity: 0.25,
            })}
            onEachFeature={(feature, layer) => {
              if (feature.properties && feature.properties.count) {
                layer.bindTooltip(`Est. stalls: ${feature.properties.count}`, {
                  sticky: true,
                  className: "font-semibold",
                });
              }
            }}
          />
        )}
      </MapContainer>
    </div>
  );
}
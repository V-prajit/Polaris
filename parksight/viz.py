"""
Interactive map visualisation with Folium.

Creates satellite-overlay maps showing parking features,
stall counts, and search radius.  Returns ``folium.Map``
objects that render inline in Jupyter notebooks.
"""

from __future__ import annotations

import folium
import geopandas as gpd


# ── Esri tile URL (no API key needed) ──────────────────────────────

ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def create_parking_map(
    lat: float,
    lon: float,
    gdf: gpd.GeoDataFrame,
    *,
    radius: int | None = None,
    total_count: int | None = None,
    zoom: int = 17,
) -> folium.Map:
    """
    Build an interactive Folium map centred at *(lat, lon)*.

    Parameters
    ----------
    lat, lon : float
        Centre of the map (WGS-84).
    gdf : GeoDataFrame
        Parking features in EPSG:4326.  If a ``"count"`` column exists its
        values are shown in tooltips.
    radius : int, optional
        If provided, draw a circle showing the search radius in metres.
    total_count : int, optional
        If provided, display a legend with the total estimated stall count.
    zoom : int
        Initial zoom level (default 17).

    Returns
    -------
    folium.Map
        Interactive map (renders inline in Jupyter).
    """
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        control_scale=True,
        tiles=None,
    )

    # Satellite base layer
    folium.TileLayer(
        tiles=ESRI_TILES,
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
        max_zoom=20,
        detect_retina=True,
    ).add_to(m)

    # OSM streets toggle
    folium.TileLayer(
        "OpenStreetMap",
        name="OSM Streets",
        control=True,
    ).add_to(m)

    # Search radius circle
    if radius is not None:
        folium.Circle(
            location=(lat, lon),
            radius=radius,
            color="#1976d2",
            fill=True,
            fill_opacity=0.10,
            weight=2,
        ).add_to(m)

    # Centre marker
    folium.Marker(
        location=(lat, lon),
        tooltip="Search centre",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(m)

    # Parking features
    if not gdf.empty:
        folium.GeoJson(
            data=gdf.__geo_interface__,
            name="Parking Features",
            style_function=lambda _: {
                "color": "#1976d2",
                "weight": 2,
                "fillOpacity": 0.25,
            },
            highlight_function=lambda _: {"weight": 3, "color": "#ff5722"},
        ).add_to(m)

        # Per-feature markers with stall counts
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            stalls = row.get("count", "n/a")
            folium.CircleMarker(
                location=(centroid.y, centroid.x),
                radius=4,
                color="#e53935",
                fill=True,
                fill_opacity=0.8,
                tooltip=f"Est. stalls: {stalls}",
            ).add_to(m)

    # Legend
    if total_count is not None:
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 20px; left: 20px; z-index: 9999;
            background: white; padding: 8px 12px;
            border: 2px solid grey; font-size: 14px;
        ">
            <b>Estimated total stalls:</b> {total_count}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(position="topright").add_to(m)
    return m

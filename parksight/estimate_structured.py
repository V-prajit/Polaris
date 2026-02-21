"""
Estimate parking capacity for structures and streets that satellites cannot see.

Handles three parking types invisible to overhead imagery:
  1. Multi-storey garages  (parking=multi-storey, building=garage)
  2. Underground parking   (parking=underground)
  3. Street parking         (parking:lane, parking:left/right/both tags)

Uses OSM geometry + tags to compute estimates without any ML model.
"""

import logging

from parksight import config, corrected_line_length

logger = logging.getLogger(__name__)


def estimate_structured_parking(gdf):
    """
    Estimate parking spots for garage and underground structures.

    Looks at OSM tags to determine structure type and level count,
    then uses floor area geometry to compute capacity per floor.

    Returns a list of dicts with spot estimates and metadata per feature.
    """
    stall_area = config["STALL_AREA_M2"]
    usable_fraction = config["USABLE_FRACTION_GARAGE"]
    default_garage_levels = config["DEFAULT_GARAGE_LEVELS"]
    default_underground_levels = config["DEFAULT_UNDERGROUND_LEVELS"]

    results = []
    gdf_3857 = gdf.to_crs(epsg=3857) if gdf.crs != "EPSG:3857" else gdf

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()

        parking_type = str(tags.get("parking", "")).lower()
        building_type = str(tags.get("building", "")).lower()

        is_garage = (
            parking_type in ("multi-storey", "multi_storey")
            or building_type in ("garage", "garages")
        )
        is_underground = parking_type == "underground"

        if not is_garage and not is_underground:
            continue

        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue

        floor_area_m2 = geom.area

        # determine number of levels
        levels = None
        for level_tag in ("building:levels", "parking:levels", "levels"):
            val = tags.get(level_tag)
            if val is not None:
                try:
                    levels = int(float(val))
                    break
                except (ValueError, TypeError):
                    continue

        if levels is None:
            levels = default_underground_levels if is_underground else default_garage_levels

        spots_per_floor = int(floor_area_m2 * usable_fraction / stall_area)
        total_spots = spots_per_floor * levels

        # if osm has a capacity tag, blend with our estimate
        capacity_tag = tags.get("capacity")
        if capacity_tag is not None:
            try:
                osm_capacity = int(float(capacity_tag))
                total_spots = int(0.6 * osm_capacity + 0.4 * total_spots)
            except (ValueError, TypeError):
                pass

        structure_type = "underground" if is_underground else "garage"
        name = tags.get("name", f"Structure #{idx}")

        results.append({
            "index": idx,
            "name": name,
            "type": structure_type,
            "floor_area_m2": round(floor_area_m2, 1),
            "levels": levels,
            "spots_per_floor": spots_per_floor,
            "total_spots": total_spots,
            "has_capacity_tag": capacity_tag is not None,
        })

        logger.info(
            "  %s [%s]: %.0f m2 x %d levels = %d spots",
            name, structure_type, floor_area_m2, levels, total_spots,
        )

    return results


def estimate_street_parking(gdf):
    """
    Estimate on-street parking from road segments with parking lane tags.

    Uses curb length and average car spacing to compute parallel spots.
    """
    avg_car_length = config["AVG_CAR_LENGTH_M"]
    results = []

    gdf_3857 = gdf.to_crs(epsg=3857) if gdf.crs != "EPSG:3857" else gdf

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()

        if geom.geom_type not in ("LineString", "MultiLineString"):
            continue

        # check for parking lane tags
        has_left = str(tags.get("parking:left", "")).lower() not in ("", "no", "nan", "none")
        has_right = str(tags.get("parking:right", "")).lower() not in ("", "no", "nan", "none")
        has_both = str(tags.get("parking:both", "")).lower() not in ("", "no", "nan", "none")
        has_lane = str(tags.get("parking:lane", "")).lower() not in ("", "no", "nan", "none")

        if not any([has_left, has_right, has_both, has_lane]):
            continue

        length_m = corrected_line_length(geom)

        # determine how many sides have parking
        sides = 0
        if has_both or has_lane:
            sides = 2
        else:
            if has_left:
                sides += 1
            if has_right:
                sides += 1

        # subtract ~20% for driveways, intersections, fire hydrants
        usable_length = length_m * 0.80
        spots = int(usable_length / avg_car_length) * sides

        name = tags.get("name", f"Street #{idx}")

        results.append({
            "index": idx,
            "name": name,
            "type": "street",
            "length_m": round(length_m, 1),
            "sides": sides,
            "total_spots": max(spots, 0),
        })

        logger.info(
            "  %s [street]: %.0f m x %d sides = %d spots",
            name, length_m, sides, max(spots, 0),
        )

    return results

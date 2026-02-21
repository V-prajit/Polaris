import argparse
import logging

from parksight.count import get_line_count
from parksight.fetch import get_parking_data, get_satellite_tile
from parksight.viz import create_parking_map
from yolo.detect import YOLOParkingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--radius", default=300, type=int)
    args = parser.parse_args()

    detector = YOLOParkingDetector(args.weights)
    
    logger.info(f"Fetching data for {args.address}")
    gdf, (lat, lon) = get_parking_data(args.address, dist=args.radius)
    
    if gdf.empty:
        logger.warning("No parking data found.")
        return

    gdf_3857 = gdf.to_crs(epsg=3857)
    counts = []

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)
            spots = detector.count_spots(img, geom)
            counts.append(spots)
        elif geom.geom_type in ("LineString", "MultiLineString"):
            spots = get_line_count(geom)
            counts.append(spots)
        else:
            counts.append(0)

    gdf["count"] = counts
    total_count = int(sum(counts))
    
    logger.info("Per-lot counts:")
    for idx, count in zip(gdf.index, counts):
        logger.info(f"  Feature {idx}: {count} spots")
        
    logger.info(f"Total count: {total_count}")

    m = create_parking_map(lat, lon, gdf, radius=args.radius, total_count=total_count)
    m.save("parking_map_yolo.html")
    logger.info("Saved map to parking_map_yolo.html")

if __name__ == "__main__":
    main()

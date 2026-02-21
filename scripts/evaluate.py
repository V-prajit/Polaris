import pandas as pd
import numpy as np
from parksight.count import count_edges, get_line_count
from parksight.detect import ParkingDetector
from parksight.fetch import get_parking_data, get_satellite_tile
from yolo.detect import YOLOParkingDetector

def evaluate():
    locations = [
        {"name": "Georgia Tech Library parking deck, Atlanta", "truth": 150},
        {"name": "Lenox Square Mall, Atlanta", "truth": 2000},
        {"name": "Atlantic Station, Atlanta", "truth": 1000},
        {"name": "Midtown residential area, Atlanta, GA", "truth": 50},
    ]

    try:
        yolo_det = YOLOParkingDetector("runs/parksight_yolo/weights/best.pt")
    except Exception as e:
        print(f"Skipping YOLO due to error loading weights: {e}")
        yolo_det = None

    try:
        dino_det = ParkingDetector()
    except Exception as e:
        print(f"Skipping DINO due to error: {e}")
        dino_det = None

    results = []

    for loc in locations:
        address = loc["name"]
        truth = loc["truth"]
        
        print(f"Evaluating {address}...")
        gdf, _ = get_parking_data(address, dist=300)
        
        if gdf.empty:
            print(f"No data for {address}")
            continue
            
        gdf_3857 = gdf.to_crs(epsg=3857)
        
        yolo_count = 0
        cv_count = 0
        dino_count = 0
        
        for idx, row in gdf_3857.iterrows():
            geom = row.geometry
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                img = get_satellite_tile(geom)
                cv_count += count_edges(geom)
                if yolo_det:
                    yolo_count += yolo_det.count_spots(img, geom)
                if dino_det:
                    dino_count += dino_det.count_spots(img, geom)
            elif geom.geom_type in ("LineString", "MultiLineString"):
                val = get_line_count(geom)
                cv_count += val
                if yolo_det: yolo_count += val
                if dino_det: dino_count += val
                
        results.append({
            "Location": address,
            "Truth": truth,
            "YOLO": yolo_count if yolo_det else np.nan,
            "CV": cv_count,
            "DINO": dino_count if dino_det else np.nan
        })

    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)
    
    def calc_metrics(pred, truth):
        mask = ~np.isnan(pred)
        if sum(mask) == 0:
            return np.nan, np.nan, np.nan
        err = pred[mask] - truth[mask]
        mae = np.mean(np.abs(err))
        mape = np.mean(np.abs(err) / truth[mask]) * 100
        rmse = np.sqrt(np.mean(err**2))
        return mae, mape, rmse

    print("\nResults:")
    print(df.to_string(index=False))
    
    print("\nMetrics:")
    for model in ["YOLO", "CV", "DINO"]:
        if model not in df or df[model].isna().all(): continue
        mae, mape, rmse = calc_metrics(df[model].values, df["Truth"].values)
        print(f"[{model}] MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    evaluate()

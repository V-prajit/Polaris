## Inspiration

We are Polaris and we built a north star that guides retailers to their perfect location, based on insights about parking.

Our project solves a real problem that retailers face: locations that look perfect on paper but might lead to millions/years in potential loss of foot traffic due to hidden parking dynamics.

## What it does

Polaris is a scalable ML platform that quantifies parking inventory from space. Drop a pin anywhere, and in under 2 seconds, we parse satellite imagery to deliver stall counts, confidence intervals, and a 0–100 Polaris Score for accessibility.

## How we built it

We tried 3 approaches, all 3 have their pros and cons:

1. Segmentation Transformer model: creates highly accurate masks of parking lots, but struggles to differentiate between individual stalls (computationally expensive)

2. YOLO model: creates bounding boxes around each stall and is able to detect cars, but is not very precise (fast)

3. Geometric Heuristics: uses OSM + SegFormer boundaries to estimate stall counts based on lot geometry and optimal parking angles, but struggles with irregular shaped lots (near instant, highly accurate)

## Challenges we ran into

1. Training the SegFormer model was both time consuming and expensive
2. Handling edge cases such as multi-story parking lots and street parking
3. Adapting real world insights such as optimal parking angles to our model

## What we learned

Raj taught us a lot about how to think about how this project fits into the entire ecosystem of growth factor.
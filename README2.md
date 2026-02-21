# ParkSight: Our Approach

We use a fine-tuned **YOLOv8 Nano (yolo26)** for satellite parking detection as our fast baseline, with a **SegFormer-b5** model currently training on the **ParkSeg12k** dataset for pixel-level parking stall segmentation to replace the bounding-box area estimation. OSM metadata (underground levels, garage floors) and additional APIs supplement the vision models to accurately count multi-level and underground parking that no satellite model can see from above.

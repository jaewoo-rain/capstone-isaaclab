# OMY Vision + YOLO order

## 1) Generate dataset
```bash
./isaaclab.sh -p source/omy/vision/scripts/generate_yolo_dataset.py --headless --out_dir datasets/omy_yolo --num_samples 4000
```

## 2) Train YOLO
```bash
pip install ultralytics
yolo detect train data=datasets/omy_yolo/dataset.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 device=0
```
Copy best weight to `checkpoints/yolo/best.pt`.

## 3) Play vision env
```bash
./isaaclab.sh -p source/omy/vision/scripts/play_grasp_vision.py --checkpoint checkpoints/rl/grasp.zip
./isaaclab.sh -p source/omy/vision/scripts/play_lift_vision.py --checkpoint checkpoints/rl/lift.zip
./isaaclab.sh -p source/omy/vision/scripts/play_place_vision.py --checkpoint checkpoints/rl/place.zip
```

## yolo 학습 데이터 뽑기
## datasets/omy_yolo_v4 저장
./isaaclab.sh -p source/omy/vision/scripts/generate_yolo_dataset.py --out_dir datasets/omy_yolo_v4 --num_samples 500 --enable_cameras

## yolo 학습하기
yolo detect train data=datasets/omy_yolo_v4/dataset.yaml model=yolov8n.pt imgsz=640 epochs=20 batch=16 device=0

## 모델 옮기기
cp runs/detect/train2/weights/best.pt checkpoints/yolo/best.pt

## 모델 단독 추론 확인
yolo detect predict model=checkpoints/yolo/best.pt source=datasets/omy_yolo_v4/images/val save=True
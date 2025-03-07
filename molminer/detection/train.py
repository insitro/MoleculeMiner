# Changed for core
from ultralytics import YOLO

if __name__ == "__main__":
    # pretrained on COCO https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco.yaml
    model = YOLO("yolov8n.yaml")
    model.train(data="data.yml", epochs=100, imgsz=896)

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m on the RPC dataset.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    args = parser.parse_args()

    model = YOLO(args.weights)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=3e-4,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.005,
        warmup_epochs=5,
        hsv_h=0.08,
        hsv_s=0.8,
        hsv_v=0.7,
        degrees=10,
        translate=0.2,
        scale=0.5,
        shear=2,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        erasing=0.4,
        close_mosaic=10,
        freeze=10,
        project=args.project,
    )


if __name__ == "__main__":
    main()

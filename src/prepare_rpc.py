import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)
    return images, annotations


def convert_split(img_ids, images, annotations, rpc_root, output_root, split):
    image_out = output_root / "images" / split
    label_out = output_root / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for image_id in img_ids:
        img = images[image_id]
        filename = img["file_name"]
        width, height = img["width"], img["height"]

        src = rpc_root / "val2019" / filename if split != "test" else rpc_root / "test2019" / filename
        if not src.exists():
            continue

        shutil.copy2(src, image_out / filename)

        label_file = label_out / filename.replace(".jpg", ".txt")
        with open(label_file, "w", encoding="utf-8") as f:
            for ann in annotations[image_id]:
                x, y, bw, bh = ann["bbox"]
                xc = (x + bw / 2) / width
                yc = (y + bh / 2) / height
                bw /= width
                bh /= height
                cls = int(ann["category_id"]) - 1
                f.write(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert RPC annotations to YOLO format.")
    parser.add_argument("--rpc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_images, val_annotations = load_annotations(args.rpc_root / "instances_val2019.json")
    ids = list(val_images)
    random.shuffle(ids)

    split_point = int(0.5 * len(ids))
    train_ids = ids[:split_point]
    val_ids = ids[split_point:]

    convert_split(train_ids, val_images, val_annotations, args.rpc_root, args.output_dir, "train")
    convert_split(val_ids, val_images, val_annotations, args.rpc_root, args.output_dir, "val")

    test_images, test_annotations = load_annotations(args.rpc_root / "instances_test2019.json")
    test_ids = list(test_images)
    if args.test_samples < len(test_ids):
        test_ids = random.sample(test_ids, args.test_samples)

    convert_split(test_ids, test_images, test_annotations, args.rpc_root, args.output_dir, "test")

    yaml_path = args.output_dir / "rpc.yaml"
    names = [str(i) for i in range(200)]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {args.output_dir.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "nc: 200\n"
            f"names: {names}\n"
        )

    print(f"Prepared YOLO dataset at: {args.output_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from .vae_model import VAE


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)

    inter = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    area_a = max(0, xa2 - xa1 + 1) * max(0, ya2 - ya1 + 1)
    area_b = max(0, xb2 - xb1 + 1) * max(0, yb2 - yb1 + 1)

    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_encoder(kind, checkpoint, device):
    if kind == "dino":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["student"])
        model.eval()

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def encode(crop):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(tensor).cpu().numpy()[0]
            return embedding

    else:
        model = VAE(latent_dim=256).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])

        def encode(crop):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = model.encode(tensor)
            return mu.cpu().numpy()[0]

    return encode


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS product embedding database from RPC validation images."
    )
    parser.add_argument("--rpc-root", required=True, type=Path)
    parser.add_argument("--yolo-checkpoint", required=True, type=Path)
    parser.add_argument("--encoder", choices=["dino", "vae"], default="dino")
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(args.yolo_checkpoint))
    encode = load_encoder(args.encoder, args.encoder_checkpoint, device)

    json_path = args.rpc_root / "instances_val2019.json"
    image_root = args.rpc_root / "val2019"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    if args.max_images is not None:
        images = images[: args.max_images]

    embeddings = []
    labels = []

    for item in tqdm(images, desc="Building embedding database"):
        path = image_root / item["file_name"]
        image = cv2.imread(str(path))
        if image is None:
            continue

        detections = yolo(image, verbose=False)

        for result in detections:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                best_iou = 0.0
                best_label = -1

                for ann in annotations[item["id"]]:
                    gx, gy, gw, gh = ann["bbox"]
                    gt_box = [gx, gy, gx + gw, gy + gh]
                    score = iou([x1, y1, x2, y2], gt_box)

                    if score > best_iou:
                        best_iou = score
                        best_label = int(ann["category_id"])

                if best_iou >= args.iou_threshold:
                    embedding = encode(crop).astype("float32")
                    embedding /= max(np.linalg.norm(embedding), 1e-12)
                    embeddings.append(embedding)
                    labels.append(best_label)

    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check dataset/model paths.")

    matrix = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output_dir / "product_embeddings.index"))

    with open(args.output_dir / "embedding_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"Stored {len(labels)} embeddings in {args.output_dir}")


if __name__ == "__main__":
    main()

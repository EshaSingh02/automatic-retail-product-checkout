import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

from .vae_model import VAE


def normalize(vector):
    vector = np.asarray(vector, dtype="float32")
    return vector / max(np.linalg.norm(vector), 1e-12)


def load_dino(checkpoint_path, device):
    student = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["student"])
    student.eval()
    return student


def main():
    parser = argparse.ArgumentParser(description="Run retail product retrieval on an image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--yolo-checkpoint", required=True, type=Path)
    parser.add_argument("--encoder", choices=["dino", "vae"], default="dino")
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(args.yolo_checkpoint))
    index = faiss.read_index(str(args.index))
    labels = json.loads(args.labels.read_text())

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)

    if args.encoder == "dino":
        encoder = load_dino(args.encoder_checkpoint, device)
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def get_embedding(crop):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(tensor).cpu().numpy()[0]
            return normalize(emb).astype("float32")

    else:
        encoder = VAE(latent_dim=256).to(device)
        encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
        encoder.eval()
        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])

        def get_embedding(crop):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = encoder.encode(tensor)
            return normalize(mu.cpu().numpy()[0]).astype("float32")

    results = yolo(image, verbose=False)

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = get_embedding(crop).reshape(1, -1)
            distances, indices = index.search(embedding, args.k)

            neighbor_labels = [
                labels[idx] for idx in indices[0] if idx != -1
            ]
            prediction = (
                Counter(neighbor_labels).most_common(1)[0][0]
                if neighbor_labels else None
            )

            print(
                {
                    "bbox": [x1, y1, x2, y2],
                    "predicted_category": prediction,
                    "similarity_scores": distances[0].tolist(),
                }
            )


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def normalize_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def build_index(embeddings, labels, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = normalize_embeddings(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "product_embeddings.index"))
    with open(output_dir / "embedding_labels.json", "w", encoding="utf-8") as f:
        json.dump([int(x) for x in labels], f)

    return index


def search(index, query_embedding, labels, k=5):
    query = normalize_embeddings(np.asarray(query_embedding).reshape(1, -1))
    distances, indices = index.search(query, k)

    neighbor_labels = [
        labels[idx] for idx in indices[0] if idx != -1
    ]

    if not neighbor_labels:
        return None, distances[0].tolist()

    counts = {}
    for label in neighbor_labels:
        counts[label] = counts.get(label, 0) + 1

    prediction = max(counts, key=counts.get)
    return prediction, distances[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Create a FAISS index from embeddings.")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    labels = np.load(args.labels)

    build_index(embeddings, labels, args.output_dir)
    print(f"Index written to {args.output_dir}")


if __name__ == "__main__":
    main()

# Methodology

## Pipeline

The project follows a two-stage computer-vision pipeline:

1. Detect products using YOLOv8.
2. Crop the detected product regions.
3. Encode each crop using either a VAE or DINOv2.
4. Store normalized embeddings in a FAISS index.
5. Retrieve the nearest embeddings using cosine similarity.
6. Use the top five neighbors and majority voting for product identification.

## YOLOv8

The supplied training notebook fine-tunes a pretrained YOLOv8m detector on 200 RPC product categories. Its training configuration uses 22 epochs, 800-pixel images, batch size 16, SGD, learning rate `3e-4`, five warm-up epochs and ten frozen backbone layers, together with checkout-oriented augmentations.

## VAE

The VAE receives 128×128 RGB product crops. Its encoder contains four convolutional stages and maps the resulting feature representation to a 256-dimensional mean vector and log-variance vector. The latent sample is obtained through the reparameterization trick. The training objective combines reconstruction/KL loss with a product-similarity loss.

## DINOv2

The DINO pipeline uses ViT-S/14. The supplied notebook creates two 224×224 global crops and four 98×98 local crops. The student fine-tunes the last two transformer blocks, while the teacher is updated using momentum. A 256-dimensional projection head is used for the training objective.

## Retrieval

Embeddings are L2-normalized and stored in a FAISS `IndexFlatIP` index. Because normalized inner product equals cosine similarity, nearest-neighbor search can be used for cosine-based product retrieval. The system retrieves five neighbors and applies majority voting.

## Evaluation

The supplied project evaluates Precision, Recall, mAP@0.5, mAP@0.5:0.95 and FPS along with RPC-specific ACD, mCCD and mCIoU metrics.

## Embedding database construction

`src/build_database.py` follows the supplied notebooks by running YOLOv8 on RPC validation images, matching each detection to its highest-IoU ground-truth annotation, retaining detections above the configured IoU threshold, encoding the crop, normalizing the embedding, and writing a FAISS inner-product index plus label map.

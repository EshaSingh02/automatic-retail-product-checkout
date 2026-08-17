import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .vae_model import VAE, contrastive_loss, vae_loss


class RPCVAEDataset(Dataset):
    """Loads cropped RPC images named using image_<label>.jpg."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.jpg"))
        self.transform = T.Compose(
            [
                T.Resize((128, 128)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.1, 0.1, 0.1, 0.02),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        label = int(path.stem.split("_")[1])
        return self.transform(image), label


def main():
    parser = argparse.ArgumentParser(description="Train the RPC VAE embedding model.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", default="checkpoints/best_vae.pth", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RPCVAEDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = VAE(latent_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            recon, mu, logvar = model(imgs)
            loss_reconstruction = vae_loss(recon, imgs, mu, logvar)
            loss_similarity = contrastive_loss(mu, labels)
            loss = loss_reconstruction + 0.5 * loss_similarity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output)

    print(f"Best VAE checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()

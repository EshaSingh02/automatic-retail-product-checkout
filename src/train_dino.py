import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dino_model import DINOHead, DINOLoss, product_similarity_loss, update_teacher


class DINOTransform:
    def __init__(self):
        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.GaussianBlur(3),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(98, scale=(0.2, 0.5)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image):
        crops = [self.global_transform(image), self.global_transform(image)]
        crops.extend(self.local_transform(image) for _ in range(4))
        return crops


class RPCDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.jpg"))
        self.transform = DINOTransform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        label = int(path.stem.split("_")[1])
        return self.transform(image), label


def collate_fn(batch):
    crops = list(zip(*[item[0] for item in batch]))
    crops = [torch.stack(group) for group in crops]
    labels = torch.tensor([item[1] for item in batch])
    return crops, labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for RPC product embeddings.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", default="checkpoints/dino_rpc_model.pth", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RPCDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
    )

    student = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)

    for name, param in student.named_parameters():
        param.requires_grad = "blocks.10" in name or "blocks.11" in name

    for param in teacher.parameters():
        param.requires_grad = False

    student_head = DINOHead().to(device)
    teacher_head = DINOHead().to(device)
    for param in teacher_head.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(student_head.parameters()), lr=args.lr
    )
    criterion = DINOLoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        student.train()
        student_head.train()

        for crops, labels in loader:
            crops = [crop.to(device) for crop in crops]
            labels = labels.to(device)

            student_outputs = [student_head(student(crop)) for crop in crops]

            with torch.no_grad():
                teacher_outputs = [
                    teacher_head(teacher(crop)) for crop in crops[:2]
                ]

            loss_dino = criterion(student_outputs, teacher_outputs)
            loss_product = product_similarity_loss(student_outputs[0], labels)
            loss = loss_dino + 0.5 * loss_product

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            update_teacher(student, teacher, student_head, teacher_head)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"total={loss.item():.4f}, dino={loss_dino.item():.4f}, "
            f"product={loss_product.item():.4f}"
        )

    torch.save(
        {
            "student": student.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher": teacher.state_dict(),
            "teacher_head": teacher_head.state_dict(),
        },
        args.output,
    )
    print(f"DINO checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()

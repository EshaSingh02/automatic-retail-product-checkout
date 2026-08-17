import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim=384, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DINOLoss(nn.Module):
    def __init__(self, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(self, student_outputs, teacher_outputs):
        total = 0.0
        count = 0

        for teacher_out in teacher_outputs:
            teacher_prob = F.softmax(
                teacher_out.detach() / self.teacher_temp, dim=-1
            )

            for student_out in student_outputs:
                student_log_prob = F.log_softmax(
                    student_out / self.student_temp, dim=-1
                )
                total += torch.mean(
                    torch.sum(-teacher_prob * student_log_prob, dim=-1)
                )
                count += 1

        return total / max(count, 1)


def product_similarity_loss(embeddings, labels, margin=0.3):
    embeddings = F.normalize(embeddings, dim=1)
    total = 0.0
    count = 0

    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i == j:
                continue

            sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0),
            )

            if labels[i] == labels[j]:
                total += 1 - sim
            else:
                total += F.relu(sim - margin)

            count += 1

    return total / max(count, 1)


@torch.no_grad()
def update_teacher(student, teacher, student_head, teacher_head, momentum=0.996):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_(
            student_param.data, alpha=1 - momentum
        )

    for student_param, teacher_param in zip(
        student_head.parameters(), teacher_head.parameters()
    ):
        teacher_param.data.mul_(momentum).add_(
            student_param.data, alpha=1 - momentum
        )

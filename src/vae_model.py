import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Convolutional VAE used for 128x128 RGB product crops."""

    def __init__(self, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h = self.decoder_input(z).view(-1, 256, 8, 8)
        recon = self.decoder(h)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight=0.001):
    reconstruction = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction + kl_weight * kl


def contrastive_loss(embeddings, labels, margin=0.3):
    embeddings = F.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.T

    labels = labels.unsqueeze(1)
    positive = (labels == labels.T).float()
    negative = (labels != labels.T).float()

    positive_loss = (1 - similarity) * positive
    negative_loss = F.relu(similarity - margin) * negative

    denom = positive.sum() + negative.sum()
    return (positive_loss.sum() + negative_loss.sum()) / denom.clamp_min(1.0)

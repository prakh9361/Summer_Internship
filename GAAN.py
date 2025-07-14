import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

# ----------------------
# Data Preparation
# ----------------------
class XYZPointCloudDataset(Dataset):
    def __init__(self, folder_path, num_points=2048):
        self.files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('.xyz'):
                    self.files.append(os.path.join(root, filename))
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = []
        with open(self.files[idx], 'r') as f:
            for line in f:
                if line.strip() and (line[0].isdigit() or line[0] == '-'):
                    parts = line.split()
                    if len(parts) == 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        data = np.array(data)
        if data.shape[0] > self.num_points:
            indices = np.random.choice(data.shape[0], self.num_points, replace=False)
            data = data[indices]
        elif data.shape[0] < self.num_points:
            pad = np.zeros((self.num_points - data.shape[0], 3))
            data = np.vstack([data, pad])
        return torch.tensor(data, dtype=torch.float32)

# ----------------------
# Model Components
# ----------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, latent_dim, 1)
        )

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = torch.max(x, 2)[0]
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_points=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
        self.num_points = num_points

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, self.num_points, 3)

class Discriminator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, z):
        return self.fc(z)

# ----------------------
# Training Setup
# ----------------------
def chamfer_distance(x, y):
    x = x.unsqueeze(2)  # [B, N, 1, 3]
    y = y.unsqueeze(1)  # [B, 1, M, 3]
    dist = ((x - y) ** 2).sum(-1)
    return dist.min(2)[0].mean(1) + dist.min(1)[0].mean(1)


def train():
    latent_dim = 128
    num_points = 4320
    dataset = XYZPointCloudDataset("./xyz_files", num_points)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    encoder = Encoder(latent_dim).cuda()
    generator = Generator(latent_dim, num_points).cuda()
    discriminator = Discriminator(latent_dim).cuda()

    opt_EG = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=1e-4)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(20):
        encoder.train()
        generator.train()
        discriminator.train()
        for batch in dataloader:
            batch = batch.cuda()

            z_fake = encoder(batch)
            x_recon = generator(z_fake)

            # Reconstruction Loss
            recon_loss = chamfer_distance(batch, x_recon).mean()

            # Adversarial Loss for D
            z_real = torch.randn_like(z_fake)
            d_real = discriminator(z_real)
            d_fake = discriminator(z_fake.detach())
            loss_D = -(d_real.mean() - d_fake.mean())

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Adversarial Loss for EG
            d_fake = discriminator(z_fake)
            adv_loss = -d_fake.mean()
            loss_EG = recon_loss + 0.05 * adv_loss

            opt_EG.zero_grad()
            loss_EG.backward()
            opt_EG.step()

            print(f"Epoch {epoch}: Recon {recon_loss.item():.4f} Adv {adv_loss.item():.4f}")

        torch.save(encoder.state_dict(), f'encoder_epoch_{epoch}.pth')
        torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    train()

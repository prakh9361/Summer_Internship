import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

# === Chamfer Distance ===
def chamfer_distance(p1, p2):
    x, y = p1, p2
    x_size, y_size = x.size(1), y.size(1)
    x = x.unsqueeze(2).expand(-1, -1, y_size, -1)
    y = y.unsqueeze(1).expand(-1, x_size, -1, -1)
    dist = torch.norm(x - y, dim=3)
    dist1 = dist.min(2)[0].mean(1)
    dist2 = dist.min(1)[0].mean(1)
    return (dist1 + dist2).mean()

# === Data Loading ===
class XYZDataset(Dataset):
    def __init__(self, directory, max_points=4320):
        self.files = sorted(glob.glob(f"{directory}/*.xyz"))
        self.max_points = max_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            lines = f.readlines()
            # Skip header lines
            atom_lines = lines[2:]  # First 2 lines are metadata
            coords = []
            for line in atom_lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            coords = np.array(coords)
            if coords.shape[0] > self.max_points:
                coords = coords[:self.max_points]
            elif coords.shape[0] < self.max_points:
                coords = np.pad(coords, ((0, self.max_points - coords.shape[0]), (0, 0)))
            return torch.tensor(coords, dtype=torch.float32)


# === Models ===
class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, z_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 3)
        x = self.mlp(x).mean(dim=1)  # Global average pool
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim=128, out_points=4320):
        super().__init__()
        self.out_points = out_points
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 512), nn.ReLU(),
            nn.Linear(512, out_points * 3)
        )

    def forward(self, z):
        x = self.mlp(z).view(-1, self.out_points, 3)
        return x

class Discriminator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.net(z)

# === Training Loop ===
def train(directory, epochs=100, batch_size=32, z_dim=128, lr=1e-4):
    dataset = XYZDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    E = Encoder(z_dim).cuda()
    G = Decoder(z_dim).cuda()
    D = Discriminator(z_dim).cuda()

    opt_EG = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)

    for epoch in range(epochs):
        for pc in dataloader:
            pc = pc.cuda()

            # === Forward ===
            z = E(pc)
            rec_pc = G(z)

            # === Loss ===
            loss_rec = chamfer_distance(pc, rec_pc)

            # === Discriminator update ===
            z_real = torch.randn_like(z)
            D_loss = -torch.mean(D(z_real)) + torch.mean(D(z.detach()))
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # === Encoder-Decoder update ===
            D_adv = -torch.mean(D(z))
            loss = loss_rec + 0.01 * D_adv
            opt_EG.zero_grad()
            loss.backward()
            opt_EG.step()

        print(f"Epoch {epoch+1} | Rec Loss: {loss_rec.item():.4f} | D Loss: {D_loss.item():.4f}")

# === Run training ===
if __name__ == "__main__":
    train("./xyz_files")

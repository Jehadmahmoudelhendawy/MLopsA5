import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import shutil


base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

os.makedirs("./data/MNIST/raw", exist_ok=True)

for f in files:
    gz_path = f"./data/MNIST/raw/{f}"
    if not os.path.exists(gz_path):
        print(f"Downloading {f}...")
        urllib.request.urlretrieve(base_url + f, gz_path)
        print(f"Extracting {f}...")
        with gzip.open(gz_path, 'rb') as f_in, open(gz_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
lr = 0.0002
epochs = 30
latent_dim = 100

from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=False 
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1, 28, 28)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)
for epoch in range(epochs):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)

        optimizer_D.zero_grad()
        outputs = D(real_imgs)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_imgs = G(z)
        outputs = D(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_imgs = G(z)
        outputs = D(fake_imgs)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

z = torch.randn(16, latent_dim).to(device)
generated = G(z).detach().cpu()

grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1,2,0))
plt.axis("off")
plt.show()
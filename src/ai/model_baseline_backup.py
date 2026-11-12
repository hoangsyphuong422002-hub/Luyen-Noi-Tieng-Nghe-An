import torch
import torch.nn as nn

class ProsodyAutoEncoder(nn.Module):
    """
    Hybrid 1D-Convolutional & Bottleneck Auto-Encoder for Nghe An Prosody (F0 Contour).
    - Conv1D layers capture local syllable/tone transitions (rising, falling, dipping tones).
    - Compact Linear bottleneck enforces global Nghe An sentence cadence constraints.
    Input/Output shape: (batch_size, 200)
    """
    def __init__(self, input_dim: int = 200, latent_dim: int = 32):
        super(ProsodyAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Convolutional feature extractor: (B, 1, 200) -> (B, 32, 25)
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),  # -> (B, 16, 100)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),  # -> (B, 32, 50)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),  # -> (B, 32, 25)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck projection: 32 * 25 = 800 -> latent_dim -> 800
        self.fc_encode = nn.Sequential(
            nn.Linear(32 * 25, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32 * 25),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional decoder: (B, 32, 25) -> (B, 1, 200)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 32, 50)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # -> (B, 16, 100)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1)    # -> (B, 1, 200)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)  # (B, 1, 200)
        h = self.conv_encoder(h)
        h = h.view(h.size(0), -1)
        return self.fc_encode(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), 32, 25)
        out = self.conv_decoder(h)
        return out.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


if __name__ == "__main__":
    model = ProsodyAutoEncoder(input_dim=200, latent_dim=32)
    x = torch.randn(4, 200)
    y = model(x)
    print(f"Conv1D AutoEncoder test: {x.shape} -> {y.shape}")

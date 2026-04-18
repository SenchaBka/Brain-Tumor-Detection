# Unsupervised — Autoencoder / Feature Extraction

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from main import train_loader

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Training Function

import torch.optim as optim

def train_autoencoder(model, train_loader, epochs=10, lr=0.001):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model, losses

if __name__ == "__main__":
    # Autoencoder Training
    autoencoder = Autoencoder()

    autoencoder, ae_losses = train_autoencoder(
        autoencoder,
        train_loader,
        epochs=20, # Increased from 10
        lr=0.001
    )

    torch.save(autoencoder.encoder.state_dict(), os.path.join(output_dir, "encoder.pth"))

    # Save training metrics to txt file
    final_loss = ae_losses[-1] if ae_losses else 0
    with open(os.path.join(output_dir, "accuracy_comparison.txt"), "a") as f:
        f.write("="*60 + "\n")
        f.write("EXPERIMENT 2.1: Unsupervised Autoencoder\n")
        f.write("="*60 + "\n\n")
        f.write("Note: This is unsupervised learning, so no classification accuracy.\n")
        f.write(f"Final Autoencoder Reconstruction Loss: {final_loss:.6f}\n")
        f.write(f"Initial Reconstruction Loss: {ae_losses[0]:.6f}\n")
        f.write(f"Loss Improvement: {ae_losses[0] - final_loss:.6f}\n")
        f.write(f"Epochs Trained: {len(ae_losses)}\n\n")

    print("✓ Metrics saved to output/accuracy_comparison.txt")

    # Result visualization
    plt.figure(figsize=(8, 5))
    plt.plot(ae_losses)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "autoencoder_training_loss.png"))
    plt.close()



    images, _ = next(iter(train_loader))
    images = images.to(device)

    with torch.no_grad():
        reconstructed = autoencoder(images)

    # Move to CPU for plotting
    images = images.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i in range(5):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")

    plt.savefig(os.path.join(output_dir, "autoencoder_reconstructed_images.png"))
    plt.close()
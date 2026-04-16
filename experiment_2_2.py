# Unsupervised — Autoencoder / Feature Extraction

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from experiment_2_1 import Autoencoder
from main import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder Model for Feature Extraction
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Autoencoder().encoder

    def forward(self, x):
        return self.encoder(x)

encoder = Encoder()
encoder.encoder.load_state_dict(torch.load("encoder.pth"))
encoder.to(device)

# Classifier using Frozen Encoder and Fine-tuning
class Classifier(nn.Module):
    def __init__(self, encoder, freeze_encoder=True):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output

model_frozen = Classifier(encoder, freeze_encoder=True).to(device)

encoder_ft = Encoder()
encoder_ft.encoder.load_state_dict(torch.load("encoder.pth"))

model_finetune = Classifier(encoder_ft, freeze_encoder=False).to(device)


# Training Function
def train_classifier(model, train_loader, val_loader, epochs=10, lr=0.001):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels_f = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels_f).item()
                preds = (outputs > 0.5).squeeze(1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct / total)

    return train_losses, val_losses, val_accs


# Train both models and visualize results
train_loss_frozen, val_loss_frozen, val_acc_frozen = train_classifier(model_frozen, train_loader, val_loader)
train_loss_finetune, val_loss_finetune, val_acc_finetune = train_classifier(model_finetune, train_loader, val_loader)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training Loss
axes[0].plot(train_loss_frozen, label="Frozen")
axes[0].plot(train_loss_finetune, label="Fine-tuned")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Validation Loss
axes[1].plot(val_loss_frozen, label="Frozen")
axes[1].plot(val_loss_finetune, label="Fine-tuned")
axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

# Validation Accuracy
axes[2].plot(val_acc_frozen, label="Frozen")
axes[2].plot(val_acc_finetune, label="Fine-tuned")
axes[2].set_title("Validation Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].legend()

plt.suptitle("Frozen Encoder vs Fine-tuned Encoder", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# Prediction Visualization
def show_predictions(model, loader):

    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)

    outputs = model(images)
    preds = (outputs > 0.5).squeeze(1)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {int(preds[i].item())}\nTrue: {labels[i]}", fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

print("Frozen Model Predictions")
show_predictions(model_frozen, val_loader)

print("Fine-tuned Model Predictions")
show_predictions(model_finetune, val_loader)
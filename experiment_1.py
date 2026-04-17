# Experiment 1 (Supervised CNN)

import os
import torch
import torch.nn as nn
import torch.optim as optim

from main import train_loader, test_loader, val_loader, train_loader_no_aug

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # normalization
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # regularization
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# Training Loop
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=1e-4):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        train_acc = correct / total
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies

# Augmentation Comparison
#With Augmentation
print("\nTraining WITH augmentation")

model_aug = SimpleCNN(dropout_rate=0.3)

model_aug, _, _, _, val_acc_aug = train_model(
    model_aug,
    train_loader,
    val_loader,
    epochs=20,
    lr=0.0005
)

#Without Augmentation
print("\nTraining WITHOUT augmentation")

model_no_aug = SimpleCNN(dropout_rate=0.3)

model_no_aug, _, _, _, val_acc_no_aug = train_model(
    model_no_aug,
    train_loader_no_aug,
    val_loader,
    epochs=20,
    lr=0.0005
)

# Hyperparameters
# Experiment with different hyperparameters
configs = [
    {"lr": 0.001, "dropout": 0.5},
    {"lr": 0.0005, "dropout": 0.3},
    {"lr": 0.0001, "dropout": 0.5},
    {"lr": 0.001, "dropout": 0.3},
]

results = []

best_model = None
best_acc = 0

for config in configs:
    print("\nTesting config:", config)

    model = SimpleCNN(dropout_rate=config["dropout"])

    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=config["lr"]
    )

    final_acc = val_acc[-1]

    results.append({
        "config": config,
        "final_val_loss": val_loss[-1],
        "final_val_acc": val_acc[-1]
    })

    if final_acc > best_acc:
        best_acc = final_acc
        best_model = trained_model

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

print("\nTesting the Best Model:")
evaluate(trained_model, test_loader)

print("\nAugmentation Comparison:")
print(f"Validation Accuracy WITH Augmentation: {val_acc_aug[-1]:.4f}")
print(f"Validation Accuracy WITHOUT Augmentation: {val_acc_no_aug[-1]:.4f}")

print("\nFinal Comparison:")
for r in results:
    print(f"Config: {r['config']}, "
          f"Val Loss: {r['final_val_loss']:.4f}, "
          f"Val Acc: {r['final_val_acc']:.4f}")

# Save accuracy results to txt file
with open(os.path.join(output_dir, "accuracy_comparison.txt"), "w") as f:
    f.write("="*60 + "\n")
    f.write("EXPERIMENT 1: Supervised CNN\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Best Model Config: {results[-1]['config']}\n")
    f.write(f"Best Validation Accuracy: {results[-1]['final_val_acc']:.4f}\n")
    f.write(f"Best Validation Loss: {results[-1]['final_val_loss']:.4f}\n\n")
    
    f.write(f"Augmentation Impact:\n")
    f.write(f"  WITH Augmentation: {val_acc_aug[-1]:.4f}\n")
    f.write(f"  WITHOUT Augmentation: {val_acc_no_aug[-1]:.4f}\n\n")
    
    f.write(f"All Configurations:\n")
    for r in results:
        f.write(f"  Config {r['config']}: Val Acc = {r['final_val_acc']:.4f}, Val Loss = {r['final_val_loss']:.4f}\n")
    f.write("\n")

print("\n✓ Accuracies saved to output/accuracy_comparison.txt")
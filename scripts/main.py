import os
import kagglehub
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# LOAD DATASET
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

all_files, all_labels = [], []
for label, class_name in enumerate(("no", "yes")):
    folder = os.path.join(path, class_name)
    for fname in os.listdir(folder):
        if fname.lower().endswith(VALID_EXTS):
            all_files.append(os.path.join(folder, fname))
            all_labels.append(label)

# DATA CLEANING
clean_files = []
clean_labels = []

for file_path, label in zip(all_files, all_labels):
    try:
        img = Image.open(file_path)
        img.verify()
        clean_files.append(file_path)
        clean_labels.append(label)
    except:
        print(f"Corrupted image removed: {file_path}")

all_files = clean_files
all_labels = clean_labels

# DATA ANALYSIS
print("\nDataset Analysis:")

total_images = len(all_files)
tumor_count = sum(all_labels)
healthy_count = total_images - tumor_count

print(f"Total Images: {total_images}")
print(f"Tumor Images: {tumor_count}")
print(f"Healthy Images: {healthy_count}")

print(f"Tumor Percentage: {100 * tumor_count / total_images:.2f}%")
print(f"Healthy Percentage: {100 * healthy_count / total_images:.2f}%")

# PRE PROCESSING IMAGES
IMG_SIZE = 224

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare_image(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    return transform(img)


FIRST_N = 3
sample_images, sample_titles = [], []

for label in ("yes", "no"):
    folder = os.path.join(path, label)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))[:FIRST_N]
    for fname in files:
        fpath = os.path.join(folder, fname)
        sample_images.append(Image.open(fpath).convert("RGB"))
        sample_titles.append(f"{'Tumor' if label == 'yes' else 'Healthy'}\n{fname}")
        tensor = prepare_image(fpath, eval_transform)

fig, axes = plt.subplots(2, FIRST_N, figsize=(FIRST_N * 4, 8))
for ax, img, title in zip(axes.flat, sample_images, sample_titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.suptitle("First 3 images per class (original)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "original_images.png"))

# SPLIT DATASET

train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.30, stratify=all_labels, random_state=42)

val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42)

print(f"\nDataset split:")
print(f"  Train      : {len(train_files)} images  (tumor={sum(train_labels)}, healthy={train_labels.count(0)})")
print(f"  Validation : {len(val_files)} images  (tumor={sum(val_labels)}, healthy={val_labels.count(0)})")
print(f"  Test       : {len(test_files)} images  (tumor={sum(test_labels)}, healthy={test_labels.count(0)})")

# DATA LOADER
BATCH_SIZE = 32
NUM_WORKERS = 0


class BrainTumorDataset(Dataset):
    """Simple dataset that loads images on-the-fly and applies a transform."""

    def __init__(self, file_paths, labels, transform):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


train_dataset = BrainTumorDataset(train_files, train_labels, train_transform)
val_dataset = BrainTumorDataset(val_files, val_labels, eval_transform)
test_dataset = BrainTumorDataset(test_files, test_labels, eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


# NO-AUGMENTATION VERSION
train_transform_no_aug = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset_no_aug = BrainTumorDataset(train_files, train_labels, train_transform_no_aug)

train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"\nDataLoaders:")
print(f"  Train batches      : {len(train_loader)}  (batch size={BATCH_SIZE}, shuffle=True)")
print(f"  Train batches (no aug): {len(train_loader_no_aug)}  (batch size={BATCH_SIZE}, shuffle=True)")

print(f"  Validation batches : {len(val_loader)}  (batch size={BATCH_SIZE}, shuffle=False)")
print(f"  Test batches       : {len(test_loader)}  (batch size={BATCH_SIZE}, shuffle=False)")

imgs, lbls = next(iter(train_loader))
print(f"Train batch images: {imgs.shape}")
print(f"Train batch labels: {lbls.shape}")

imgs, lbls = next(iter(train_loader_no_aug))
print(f"Train batch images (no aug): {imgs.shape}")
print(f"Train batch labels (no aug): {lbls.shape}")

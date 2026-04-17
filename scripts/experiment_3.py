import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
import os
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



# We use the Keras Sequential Layer to apply Random Horizontal Flips, Random Rotations, and Random Contrast
# We do the augemnetation on the fly using the map function using the GPU
# We use batch size of 16 


# 1. SETTINGS
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# 2. DATA PIPELINE
from scripts.main import train_files, train_labels, val_files, val_labels, test_files, test_labels

def process_path(file_path, label):
    # Read and decode the image (automatically converts to 3 channels/RGB like PyTorch)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    # Resize to match the PyTorch pipeline
    img = tf.image.resize(img, IMG_SIZE)
    # Explicitly set the shape to avoid batching issues
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img, label

# Convert the splits generated in main.py into TensorFlow Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_ds = train_ds.shuffle(len(train_files)).map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Augmentation Layer (Essential for small datasets)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomContrast(0.1),
])


# Preprocessing
# EfficientNetB0 has a built in Normalization Layer at the very beginning of its architecture
## Data Augmentation to Training Dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


# 1. Check the number of batches
num_batches = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Number of batches: {num_batches}")

# 2. Check the total number of images (Records)
total_images = 0
for images, labels in train_ds:
    total_images += images.shape[0]

print(f"Total number of images in train_ds: {total_images}")




# Display Augmented Dataset to see what transformed dataset looks like

# 1. Grab one batch (16 images) from your training dataset
for images, labels in train_ds.take(1):
    plt.figure(figsize=(15, 10))

    # 2. Loop through the first 6 images in that batch
    for i in range(6):
        # Original Image
        ax = plt.subplot(2, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("Original")
        plt.axis("off")

        # Augmented Version of THAT SAME Image
        ax = plt.subplot(2, 6, i + 7)
        # FORCE training=True to see the transformation
        aug_img = data_augmentation(tf.expand_dims(images[i], 0), training=True)

        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("Augmented")
        plt.axis("off")

    plt.suptitle("On-the-Fly Data Augmentation: Original vs. GPU-Transformed", fontsize=16)
    plt.show()
    break



# Saving augmented samples
output_dir = 'augmented_samples'
os.makedirs(output_dir, exist_ok=True)

for images, _ in train_ds.take(1):
    for i in range(5):
        aug_img = data_augmentation(tf.expand_dims(images[i], 0), training=True)
        # Convert back to standard image format and save
        img_to_save = tf.keras.utils.array_to_img(aug_img[0])
        img_to_save.save(f'{output_dir}/aug_sample_{i}.png')
print(f"Saved 5 augmented images to {output_dir} folder!")


# --- EXPERIMENT B: Train from Scratch ---
def build_model_scratch():
    base_model = applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # No pretrained weights
    )
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- EXPERIMENT C: Transfer Learning (Pretrained) ---
def build_model_transfer():
    base_model = applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet' # Use ImageNet weights
    )
    base_model.trainable = False # Freeze the backbone

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# --- RUNNING THE EXPERIMENTS ---
print("\n--- Training Experiment B: Scratch ---")
model_scratch = build_model_scratch()
history_scratch = model_scratch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("\n--- Training Experiment C: Transfer Learning ---")
model_transfer = build_model_transfer()
history_transfer = model_transfer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save the transfer learning model weights
os.makedirs("output", exist_ok=True)
model_transfer.save_weights(os.path.join("output", "model_transfer_weights.h5"))
print("\n✓ Saved Transfer Learning model weights to output/model_transfer_weights.h5")

model_transfer.save('brain_tumor_tf_model_v1')


# All the plots

# Accuracy Plots

# 1. Figure for Scratch Experiment
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_scratch.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history_scratch.history['val_accuracy'], label='Val Acc', color='red')
plt.title('Experiment B (Scratch): Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_scratch.history['loss'], label='Train Loss', color='blue')
plt.plot(history_scratch.history['val_loss'], label='Val Loss', color='red')
plt.title('Experiment B (Scratch): Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Figure for Transfer Learning Experiment
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_transfer.history['accuracy'], label='Train Acc', color='green')
plt.plot(history_transfer.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Experiment C (Transfer Learning): Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transfer.history['loss'], label='Train Loss', color='green')
plt.plot(history_transfer.history['val_loss'], label='Val Loss', color='orange')
plt.title('Experiment C (Transfer Learning): Loss')
plt.legend()
plt.tight_layout()
plt.show()



# Confusion Matrix

def plot_confusion_matrix(model, dataset, class_names):
    # 1. Get all true labels and predictions
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        # changed from 0.5 to 0.3
        y_pred.extend((preds > 0.5).astype(int).flatten())

    # 2. Compute Matrix
    cm = confusion_matrix(y_true, y_pred)

    # 3. Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: Brain Tumor Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Run it for your Transfer Learning model (evaluate on unseen test data!)
plot_confusion_matrix(model_transfer, test_ds, ['No Tumor', 'Tumor'])
# Brain Tumor Detection Using Deep Learning

## Project Overview

This project implements three different deep learning approaches for automated brain tumor detection from MRI images:

1. **Experiment 1**: Supervised CNN
2. **Experiment 2.1**: Unsupervised Autoencoder (Feature Learning)
3. **Experiment 2.2**: Transfer Learning with Pre-trained Encoder
4. **Experiment 3**: Transfer Learning with EfficientNetB0 

The project compares these approaches to identify the most effective method for brain tumor classification with limited labeled data.

---

## Group Members

Davi Zevallos, Erwin Julian, Tian Li, Michael Asfeha, Arseni Buriak, Anmol Shrestha

---

## Dataset Information

### Dataset Source
- **Name**: Brain MRI Images for Brain Tumor Detection
- **Source**: Kaggle Hub (`navoneel/brain-mri-images-for-brain-tumor-detection`)
- **Download**: Automatic via `kagglehub` Python library
- **Images**: Binary classification (tumor/healthy)


## Getting Started

### 1. Prerequisites
* Python 3.9+
* Virtual Environment (recommended)
* Kaggle API token (for dataset download)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/SenchaBka/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

# Create and activate virtual environment
python3 -m venv venv

# For Windows
venv\Scripts\activate
# For MacOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---

## Running Research Experiments

Navigate to the `research/` directory and run experiments in order:

```bash
cd research

# Step 1: Download and process dataset
python main.py

# Step 2: Run all experiments
python experiment_1.py       # Supervised CNN
python experiment_2_1.py     # Unsupervised Autoencoder
python experiment_2_2.py     # Transfer Learning (Pre-trained Encoder)
python experiment_3.py       # Transfer Learning with EfficientNetB0
```

All results will be saved to `research/output/` directory with metrics and visualizations.

---

## Running the Application (NeuroScan AI)

### 1. Download the Pre-trained Model
```bash
cd app
python3 download_model.py
```

### 2. Run the Backend API
From the root directory:
```bash
cd app
python3 app.py
```
The API will run at http://127.0.0.1:5000

### 3. Run the Frontend UI
From the root directory:
```bash
streamlit run frontend/ui.py
```

## Experiment Details

### Dataset Retrieval & Processing

The dataset is automatically downloaded and processed by `main.py`:

1. **Automatic Download**
   - First run automatically downloads dataset from Kaggle Hub 

2. **Data Cleaning**
   - Removes corrupted/invalid images
   - Verifies image integrity
   - Filters for valid formats (.jpg, .jpeg, .png, .bmp)

3. **Data Splitting**
   - **Train**: 70% (with augmentation)
   - **Validation**: 15%
   - **Test**: 15%

4. **Preprocessing**
   - Resize to 224×224 pixels
   - Normalize with ImageNet statistics
   - Apply conservative augmentation to training data only

---

### Experiment 1: Supervised CNN
**Goal**: Train CNN directly on labeled data

**Architecture**: 
- Input: 224×224 RGB images
- Conv layers: 32 → 64 → 128 filters
- Classifier: FC 256 → 2 classes
- Dropout: 0.3-0.5

**Training**:
- Hyperparameter tuning (4 configurations)
- Data augmentation comparison (with/without)
- Test on unseen data

**Output**: Classification report, confusion matrix, accuracy metrics

---

### Experiment 2.1: Unsupervised Autoencoder
**Goal**: Learn feature representations WITHOUT labels

**Architecture**:
- Encoder: Conv layers learning compressed features
- Decoder: Reconstructs images from compressed features
- Objective: Minimize reconstruction loss (MSE)

**Training**:
- 20 epochs on unlabeled training data
- Learns generic image patterns
- Saves encoder weights to `encoder.pth`

**Output**: Reconstruction loss curves, encoder weights, image reconstructions

---

### Experiment 2.2: Transfer Learning
**Goal**: Use pre-trained encoder for classification

**Architecture**:
- Feature Extractor: Encoder from Exp 2.1
- Classifier: Small network on top (128 → 64 → 1)
- Two variants: Frozen encoder vs Fine-tuned encoder

**Training**:
- Frozen: Only train classifier head
- Fine-tuned: Train all parameters
- Compare effectiveness of each approach

**Output**: Accuracy comparison, prediction visualizations

---

### Experiment 3: Transfer Learning with EfficientNetB0 (TensorFlow/Keras)
**Goal**: Leverage pre-trained ImageNet weights for brain tumor detection

**Architecture**:
- Base Model: EfficientNetB0 (pre-trained on ImageNet)
- Classification Head: GlobalAveragePooling2D → Dropout(0.2) → Dense(1, sigmoid)
- Two variants: 
  - From Scratch: Random initialization (baseline)
  - Transfer Learning: Frozen backbone + trained classifier

**Training**:
- Framework: TensorFlow/Keras
- Epochs: 20
- Batch Size: 16
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-entropy
- Data Augmentation: Random flip, rotation, contrast (GPU-accelerated on-the-fly)

**Results**:
- **From Scratch**: 63.16% validation accuracy (severe overfitting - 64% train acc visible)
- **Transfer Learning**: 89.47% validation accuracy
- **Test Set Performance**: 87% accuracy, **91% tumor recall** (after threshold optimization)
- **AUC**: High AUC score via ROC curve analysis

---

# 🧠 Application : NeuroScan AI: Brain Tumor Detection Portal

**NeuroScan AI** is a medical imaging service that utilizes Deep Learning (**EfficientNetB0**) to detect tumors in axial brain MRI scans. The project is architected using the **Service-Layer, Repository, and Adapter patterns** to ensure scalability, maintainability, and clear separation of concerns.

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit (Python-based UI) |
| **Backend API** | Flask (Python) |
| **AI Model** | TensorFlow / Keras (EfficientNetB0 Transfer Learning) |
| **Image Processing** | Pillow (PIL), NumPy |
| **Communication** | REST API (JSON via HTTP) |


## References

- PyTorch Documentation: https://pytorch.org/
- Dataset Source: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
- CNN Fundamentals: LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
- Transfer Learning: Yosinski et al., "How transferable are features in deep neural networks?"

---

**Last Updated**: April 18, 2026  
**Status**: Active Development

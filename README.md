# Brain Tumor Detection Using Deep Learning

## Project Overview

This project implements three different deep learning approaches for automated brain tumor detection from MRI images:

1. **Experiment 1**: Supervised CNN (Direct Learning)
2. **Experiment 2.1**: Unsupervised Autoencoder (Feature Learning)
3. **Experiment 2.2**: Transfer Learning with Pre-trained Encoder
4. **Experiment 3**:

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

## Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (optional, recommended for faster training)
- Activate Virtual Enviroment before (optional, but recommended)

## Running the Code

```bash
# Step 1: Install dependencies
pip install -r requirement.txt

# Step 2: Download and process dataset
python main.py

# Run all experiments sequentially
python experiment_1.py       
python experiment_2_1.py    
python experiment_2_2.py  

## Results & Output Files

All results saved to `output/` folder:

- **accuracy_comparison.txt** - Detailed accuracy metrics for all experiments
- **original_images.png** - Sample MRI images (tumor & healthy)
- **autoencoder_training_loss.png** - Autoencoder loss curves
- **autoencoder_reconstructed_images.png** - Reconstruction examples
- **encoder.pth** - Pre-trained encoder weights (from Exp 2.1)
- **$(experiment)_training.png** - Training/validation curves


## Experiment Details

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

### Experiment 3: 

---

## References

- PyTorch Documentation: https://pytorch.org/
- Dataset Source: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
- CNN Fundamentals: LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
- Transfer Learning: Yosinski et al., "How transferable are features in deep neural networks?"

---

**Last Updated**: April 16, 2026  
**Status**: Active Development

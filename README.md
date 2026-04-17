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
```
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

## 📚 Core Libraries
* **TensorFlow:** For model inference and loading the pre-trained weights.
* **Flask & Flask-CORS:** To host the RESTful API and handle cross-origin requests.
* **Streamlit:** To provide a responsive, user-friendly diagnostic interface.
* **Pillow:** For consistent image resizing and RGB conversion.
* **Requests:** For communication between the UI and the Backend.

---

## 🏗️ Architecture Design Patterns

This project follows a decoupled architecture to separate business logic from data providers and the web interface.



### 1. Adapter Pattern (`image_adapter.py`)
The **Adapter** acts as a "translator." It takes raw bytes from the HTTP request and transforms them into a normalized NumPy tensor. This ensures that the model always receives data in the exact format it was trained on (e.g., 224x224 RGB), regardless of the original file's dimensions.

### 2. Repository Pattern (`model_repository.py`)

The **Repository** manages the persistence and lifecycle of the AI model. It handles the heavy lifting of loading the `.keras` file from the disk into memory. By isolating this, the rest of the application doesn't need to know *how* the model is stored or loaded.

### 3. Service Layer (`prediction_service.py`)
The **Service** layer is the "Brain" of the application. It coordinates the logic:
1. Receiving the processed image from the **Adapter**.
2. Requesting the model from the **Repository**.
3. Executing the prediction and applying business rules (e.g., applying the 0.5 probability threshold for "Tumor" vs. "Healthy").

### 4. Presentation Layer (`ui.py` & `app.py`)
The **Flask API** acts as the delivery mechanism for the services, while the **Streamlit UI** provides a human-readable interface for doctors and researchers to interact with the system.

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.9+
* Virtual Environment (recommended)

### 2. Virtual Environment & Packages
```bash
# Clone the repository
git clone https://github.com/SenchaBka/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

# Create and activate virtual environment
python3 -m venv venv

# for MacOS
source venv/bin/activate

# Install dependencies for Application (not research)
pip install -r tensorflow Pillow Flask numpy
```
### Download the model from Huggin face
```bash
cd app
python3 script.py

```
### 2. Run the Backend
current-directory: Brain-Tumor-Detection
```bash
cd app
python3 app.py
```
The API will run at http://127.0.0.1:5000

### 3. Run the Frontend
Go back to root directory : Brain-Tumor-Detection
```bash
streamlit run frontend/ui.py
```

### Project Structure

```bash
BRAIN-TUMOR-DETECTION/
├── app/                      # Production Backend
│   ├── models/               # Pre-trained .keras files
│   ├── app.py                # Flask Entry Point
│   ├── image_adapter.py      # Data Transformation (Adapter)
│   ├── model_repository.py   # Model Loading (Repository)
│   └── prediction_service.py # Business Logic (Service)
├── frontend/                 # Streamlit UI
│   └── ui.py
├── research/                 # Experimental Scripts, Model Weights
├── dataset/                  # MRI Scans in train_test_val split ()
├── requirements.txt          # Global Dependencies
└── README.md
```

## References

- PyTorch Documentation: https://pytorch.org/
- Dataset Source: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
- CNN Fundamentals: LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
- Transfer Learning: Yosinski et al., "How transferable are features in deep neural networks?"

---

**Last Updated**: April 16, 2026  
**Status**: Active Development

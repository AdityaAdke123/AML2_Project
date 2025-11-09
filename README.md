# 🩺 Skin Lesion Classification with Explainable AI (LIME + SHAP)

This project implements an **end-to-end Explainable AI (XAI)** pipeline for **skin lesion classification** using a **Convolutional Neural Network (CNN)** model combined with **LIME** and **SHAP** interpretability methods. The system not only predicts lesion types but also visualises the reasoning behind its predictions, promoting **trust and transparency in AI-assisted dermatology**.

---

## 📘 Overview

Skin cancer detection through image-based machine learning offers a scalable, accessible way to support early diagnosis. This project uses deep learning on the **PAD-UFES-20 dataset** to classify lesions into six categories while providing interpretable explanations for clinicians.

The model was trained, validated, and deployed with an integrated **Gradio interface**, allowing users to upload images, view predictions, and examine interpretability visualisations.

---

## 🧩 Dataset

**Dataset:** [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)  
**Images:** 2,298 PNG images from 1,373 patients  
**Metadata:** `metadata.csv`

### Lesion Classes

| Code | Medical Name             |
|------|--------------------------|
| ACK  | Actinic Keratoses        |
| BCC  | Basal Cell Carcinoma     |
| BKL  | Benign Keratosis         |
| DFB  | Dermatofibroma           |
| MEL  | Melanoma                 |
| NEV  | Nevus                    |

### Google Drive Dataset Links
- [imgs_part_1](https://drive.google.com/drive/folders/1jc0aqK7NXcvcPZ9mTsDXV5um9cdAn5gG?usp=sharing)
- [imgs_part_2](https://drive.google.com/drive/folders/1_1JznLOqsZ8ujQWIiP3SbViFQDdPKDZf?usp=drive_link)
- [imgs_part_3](https://drive.google.com/drive/folders/1X1XbENwMkjVtsp74QRP82FTyWI2tEIUt?usp=drive_link)

---

## 📁 Folder Structure



### Folder Structure
```
data/
├── metadata.csv
└── images/
    ├── imgs_part_1/
    ├── imgs_part_2/
    └── imgs_part_3/
```

---

## 📂 Project Structure
```
AML2_Project/
├── data/                # Raw data and images
│   ├── metadata.csv
│   └── images/
│       ├── imgs_part_1/
│       ├── imgs_part_2/
│       └── imgs_part_3/
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── src/                 # Model scripts and helper functions
├── ui/                  # Streamlit app interface
├── results/             # Model results, plots, and metrics
├── docs/                # Technical blueprint and design documents
└── README.md            # Project overview
```

---

## ⚙️ Methodology

**Model Architecture:** Convolutional Neural Network (CNN)  
**Frameworks:** TensorFlow, Keras

### Approach
1. Preprocess and augment dataset (resize, normalize, balance classes)  
2. Train CNN using **transfer learning** (ResNet50 / VGG16)  
3. Evaluate using **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
4. Integrate trained model into Streamlit UI for real-time prediction

---

## 🧭 Installation

### Step 1 — Clone this repository
```bash
git clone https://github.com/AdityaAdke123/AML2_Project.git
cd AML2_Project
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download dataset
Download the **PAD-UFES-20 dataset** and extract under:
```
data/images/
```
## ⚙️ Model Details

**Frameworks:** TensorFlow, Keras
**Architecture:** Custom CNN (Conv2D → MaxPool → Dropout → Dense)
**Training Epochs:** 20
**Image Size:** 224 × 224 pixels
**Explainability Tools:** LIME, SHAP
**Interface:** Gradio app with real-time visualization
---

## 🚀 Usage

### 🔹 Train the CNN model
```bash
jupyter notebook notebooks/train_model.ipynb
```

### 🔹 Run Streamlit app
```bash
streamlit run ui/app.py
```

---

## 💻 Streamlit App Features

- Upload a dermoscopic image  
- Classify lesion as **Benign** or **Malignant**  
- Display prediction confidence  
- Visualize **Grad-CAM heatmap** for explainability  

### UI Layout 
```

---

## 📊 Results

Test Accuracy: 51.3%
Best Validation Accuracy: 47%
Metrics:

- Precision (weighted): 45.9%

- Recall (weighted): 48.7%

- F1-Score: 44.0%

Visual Outputs:

- Confusion Matrix

- Sample Prediction Plots

- LIME and SHAP Explanations

These early results confirm that the model can identify lesion patterns but still requires fine-tuning and larger, more balanced data for higher accuracy.
> *Results may vary depending on preprocessing and hyperparameter tuning.*

---

## ⚠️ Known Issues

- Model performance limited by dataset imbalance.

- LIME visualization sometimes fails to highlight clear regions.

- Requires GPU for smooth training and explanation generation.

- Gradio visualization may not render yellow LIME contours for all samples.

---

## 🤖 Responsible AI

- **Fairness:** Model trained on diverse skin tones to minimize bias.  
- **Transparency:** Predictions include probability and Grad-CAM visualization.  
- **Privacy:** Dataset is de-identified and used only for research purposes.

---

## 👨‍💻 Author

**Name:** Aditya Adke  
**Role:** Graduate Student — Machine Learning 2  
**University:** University of Florida  
**Email:** a.adke@ufl.edu 
**GitHub:** [https://github.com/AdityaAdke123](https://github.com/AdityaAdke123)

---

## 🙏 Acknowledgments

- **PAD-UFES-20 Dataset Authors:** Pereira et al. (2020)  
- **University of Florida — Machine Learning 2 Course Project**  
- **TensorFlow and Streamlit open-source communities**

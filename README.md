# 🩺 Skin Lesion Classification with Explainable AI (LIME + Integrated Gradients)

<img width="1184" height="757" alt="Screenshot 2025-11-23 175742" src="https://github.com/user-attachments/assets/eab50192-79a8-443e-b371-48e53e121a4a" />


This project implements an **end-to-end Explainable AI (XAI)** pipeline for automated skin lesion classification using **EfficientNetB0**, paired with **LIME** and **Integrated Gradients** for transparent interpretability.  
The final system includes a modern **Gradio interface** with probability bars, dual-explanation modes, preprocessing previews, and an elegant UI designed for clinical-style decision support.

---

## 📘 Overview

Skin cancer remains one of the most common cancers globally, and **early detection** dramatically improves survival outcomes.  
This project leverages dermatoscopic images from the **PAD-UFES-20 dataset** to classify lesions into:

- Basal Cell Carcinoma (BCC)  
- Squamous Cell Carcinoma (SCC)  
- Melanoma (MEL)  
- Nevus (NEV)  
- Actinic Keratoses (ACK)  
- Seborrheic Keratosis (SEK)  

Unlike traditional black-box models, this system integrates **visual explanation layers** (LIME + IG) to ensure transparency and support trust in AI-assisted dermatology.

---

## 🧩 Dataset

**Dataset:** PAD-UFES-20  
**Total Images:** 2,298  
**Patients:** 1,373  
**Metadata File:** `metadata.csv`

### Lesion Classes

| Code | Label                       |
|------|------------------------------|
| ACK  | Actinic Keratoses            |
| BCC  | Basal Cell Carcinoma         |
| BKL  | Benign Keratosis             |
| DFB  | Dermatofibroma               |
| MEL  | Melanoma                     |
| NEV  | Nevus                        |
| SEK  | Seborrheic Keratosis         |

### Dataset Download Links (Google Drive Mirrors)
- imgs_part_1 — https://drive.google.com/drive/folders/1jc0aqK7NXcvcPZ9mTsDXV5um9cdAn5gG  
- imgs_part_2 — https://drive.google.com/drive/folders/1_1JznLOqsZ8ujQWIiP3SbViFQDdPKDZf  
- imgs_part_3 — https://drive.google.com/drive/folders/1X1XbENwMkjVtsp74QRP82FTyWI2tEIUt  

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


---

## ⚙️ Methodology

### 🔸 Model
- **Architecture:** EfficientNetB0  
- **Input Size:** 224 × 224  
- **Loss:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Class Weights:** Enabled (to address imbalance)  

### 🔸 Explainability
| Method | Description |
|--------|-------------|
| **LIME** | Superpixel-based local interpretability |
| **Integrated Gradients** | Pixel attribution using gradient accumulation |

### 🔸 UI (Gradio)
- Modern card-based layout  
- Tabs: **Classifier** and **About & Disclaimer**
- Toggles for:
  - LIME  
  - Integrated Gradients  
  - LIME + IG  
  - None (fast prediction)
- Light/Dark theme switch  
- Confidence probability bar chart  
- Preprocessing visualisation  
- Explanation interpretation text  

---

## 📊 Performance (Phase 3 Final Model)

### EfficientNetB0 Results
| Metric               | Score   |
| -------------------- | ------- |
| **Test Accuracy**    | **73%** |
| Validation Accuracy  | ~65%    |
| Precision (weighted) | 64%     |
| Recall (weighted)    | 67%     |
| F1-Score (weighted)  | 65%     |


### Why the accuracy is not higher?
- Dataset is **highly imbalanced**
- Some lesion categories are visually **very similar**
- Dermatology requires **fine-grained texture clues**  
- Dataset size is small compared to ISIC standards  
- Lighting variations and noisy backgrounds  

Even with these constraints, the model shows strong baseline performance with transparent decision pathways.

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
```
**Frameworks:** TensorFlow, Keras
**Architecture:** Custom CNN (Conv2D → MaxPool → Dropout → Dense)
**Training Epochs:** 20
**Image Size:** 224 × 224 pixels
**Explainability Tools:** LIME, SHAP
**Interface:** Gradio app with real-time visualization

```
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

### UI Layout 
<img width="1205" height="620" alt="Screenshot 2025-11-08 222341" src="https://github.com/user-attachments/assets/f27938ca-7fb2-48b2-bb53-aa7a7b932616" />

---

## ⚠️ Known Issues

- Integrated Gradients heatmaps may appear low-contrast depending on lesion texture

- LIME is stochastic → results vary slightly per run

- GPU is strongly recommended for faster explanation generation

- Minor rendering differences possible in Gradio Cloud

---

## 🤖 Responsible AI

- Explanations accompany every prediction

- Uncertainty visualised via probability bars

- Only de-identified medical images used

- Model is presented for research and educational purposes only

- Ethically aligned with transparent AI guidelines

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

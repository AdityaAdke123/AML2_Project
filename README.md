# 🧠 Automated Skin Cancer Detection Using CNNs

This project builds a **Convolutional Neural Network (CNN)** model that classifies skin lesion images as **benign or malignant** to enable early detection of skin cancer using deep learning.

---

## 📘 Overview

Skin cancer is among the most common cancers globally, and early detection significantly improves treatment outcomes. However, access to dermatological screening is limited in many regions.

This project applies **deep learning (CNNs)** to classify skin lesion images from the **PAD-UFES-20 dataset** into benign and malignant categories, offering a potential step toward **automated, accessible skin cancer screening**.

---

## 🧩 Dataset

**Name:** [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)  
**Total Images:** 2,298  
**Patients:** 1,373  
**Format:** PNG images with accompanying `metadata.csv`

### Lesion Classes

| Category   | Classes                                                                 |
|-------------|--------------------------------------------------------------------------|
| Malignant   | BCC (Basal Cell Carcinoma), MEL (Melanoma), SCC (Squamous Cell Carcinoma) |
| Benign      | NEV (Nevus), ACK (Actinic Keratosis), SEK (Seborrheic Keratosis)         |

imgs_part_1: https://drive.google.com/drive/folders/1jc0aqK7NXcvcPZ9mTsDXV5um9cdAn5gG?usp=sharing
imgs_part_2: https://drive.google.com/drive/folders/1_1JznLOqsZ8ujQWIiP3SbViFQDdPKDZf?usp=drive_link
imgs_part_3: https://drive.google.com/drive/folders/1X1XbENwMkjVtsp74QRP82FTyWI2tEIUt?usp=drive_link

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

### UI Layout (Example)
```
----------------------------------------------
| Upload Image: [Choose File]                |
|                                            |
| [Predict]                                  |
| Prediction: Malignant                      |
| Confidence: 94%                            |
| [Show Heatmap]                             |
----------------------------------------------
```

---

## 📊 Results

| Metric        | Expected Value |
|---------------|----------------|
| Accuracy      | 90–92%         |
| Precision     | 88%            |
| Recall        | 91%            |
| F1-Score      | 89%            |
| ROC-AUC       | 0.93           |

> *Results may vary depending on preprocessing and hyperparameter tuning.*

---

## 🤖 Responsible AI

- **Fairness:** Model trained on diverse skin tones to minimize bias.  
- **Transparency:** Predictions include probability and Grad-CAM visualization.  
- **Privacy:** Dataset is de-identified and used only for research purposes.

---

## 🔮 Future Work

- Expand dataset for improved generalization  
- Add multi-class classification for all six lesion types  
- Integrate Grad-CAM visualization in UI  
- Deploy Streamlit app on Streamlit Cloud or Hugging Face Spaces  

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

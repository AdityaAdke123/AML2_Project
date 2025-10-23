# AML2_Project
# Automated Skin Cancer Detection Using CNNs

## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify skin lesion images as **benign or malignant**, aiming to support early detection of skin cancer. Using deep learning, the model analyses dermatological images from the **PAD-UFES-20** dataset to identify potentially cancerous lesions and assist in medical screening.

---

## Objectives
- Develop a CNN-based image classifier for skin lesion detection.  
- Train and evaluate the model using real-world dermatological images.  
- Provide a **Streamlit web interface** for users to upload and test images interactively.  
- Analyse model interpretability through visualisation (Grad-CAM heatmaps).

---

## Dataset
**Name:** PAD-UFES-20  
**Source:** [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1)  
**Details:**
- 2,298 images from 1,373 patients  
- Six lesion types (three malignant, three benign)  
- Images captured with smartphone cameras  
- CSV metadata containing patient and lesion details  

**Benign Classes:** NEV (Nevus), ACK (Actinic Keratosis), SEK (Seborrheic Keratosis)  
**Malignant Classes:** BCC (Basal Cell Carcinoma), MEL (Melanoma), SCC (Squamous Cell Carcinoma)

---

## Contents

- `requirements.txt` → lists all dependencies  
- `setup.ipynb` → demonstrates:  
  - Successful dataset loading and verification  
  - Basic exploration (summary statistics, class distribution, and sample plots)  
  - Environment testing and model sanity check  

Code executes cleanly and outputs visible results (e.g., sample prediction, training accuracy plot).

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AdityaAdke123/AML2_Project.git
cd AML2_Project

### 2. Install Dependencies
pip install -r requirements.txt
### 3. Run the Notebook
jupyter notebook notebooks/setup.ipynb
### 4. Launch Streamlit Interface
streamlit run ui/app.py


### Results

Model Metrics (expected):

|   Metric  | Value (approx.) |
| :-------: | :-------------- |
|  Accuracy | 90–92%          |
| Precision | 88%             |
|   Recall  | 91%             |
|  F1-Score | 89%             |
|  ROC-AUC  | 0.93            |


Outputs:

Confusion matrix

Sample predictions with confidence scores

Grad-CAM heatmaps for interpretability

Responsible AI Reflection

Fairness: Dataset includes diverse skin tones and demographics.

Transparency: Visual heatmaps show what regions the CNN focuses on.

Privacy: All images are de-identified and used for research only.

Disclaimer: This tool is for educational use only, not medical diagnosis.

Author

Name: Aditya Adke
Course: Machine Learning 2 — Fall 2025
University: University of Florida
Email: adityaadke123@gmail.com

GitHub: @AdityaAdke123

License

This project is licensed under the MIT License.
See LICENSE file for details.

Acknowledgments

PAD-UFES-20 dataset authors (Pereira et al., 2020)

TensorFlow and Streamlit open-source communities

University of Florida — Applied Machine Learning 2 coursework

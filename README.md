# AML2_Project

project:
  title: "Automated Skin Cancer Detection Using CNNs"
  description: >
    This project builds a Convolutional Neural Network (CNN) model that classifies skin lesion images
    as benign or malignant to enable early detection of skin cancer using deep learning.

overview:
  summary: >
    Skin cancer is among the most common cancers globally, and early detection significantly improves
    treatment outcomes. However, access to dermatological screening is limited in many regions.
    This project uses deep learning (CNNs) to classify skin lesion images from the PAD-UFES-20 dataset
    into benign and malignant categories, offering a potential step toward automated, accessible skin cancer screening.

dataset:
  name: "PAD-UFES-20"
  source: "https://data.mendeley.com/datasets/zr7vgbcyr2/1"
  details:
    total_images: 2298
    patients: 1373
    lesion_classes:
      malignant: ["BCC (Basal Cell Carcinoma)", "MEL (Melanoma)", "SCC (Squamous Cell Carcinoma)"]
      benign: ["NEV (Nevus)", "ACK (Actinic Keratosis)", "SEK (Seborrheic Keratosis)"]
    format: "PNG images with accompanying metadata.csv"
    folder_structure: |
      data/
      ├── metadata.csv
      └── images/
          ├── imgs_part_1/
          ├── imgs_part_2/
          └── imgs_part_3/

project_structure: |
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

methodology:
  model_architecture: "Convolutional Neural Network (CNN)"
  frameworks: ["TensorFlow", "Keras"]
  approach:
    - Preprocess and augment dataset (resize, normalize, balance classes)
    - Train CNN using transfer learning (ResNet50 / VGG16)
    - Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC
    - Integrate trained model into Streamlit UI for real-time prediction

installation:
  steps:
    - step: "Clone this repository"
      command: |
        git clone https://github.com/AdityaAdke123/AML2_Project.git
        cd AML2_Project
    - step: "Install dependencies"
      command: |
        pip install -r requirements.txt
    - step: "Download and extract dataset"
      note: "Download PAD-UFES-20 dataset and extract under data/images/"

usage:
  training:
    description: "Train the CNN model using Jupyter Notebook"
    command: |
      jupyter notebook notebooks/train_model.ipynb
  streamlit_ui:
    description: "Run Streamlit app for real-time predictions"
    command: |
      streamlit run ui/app.py

streamlit_app:
  features:
    - Upload a dermoscopic image
    - Classify lesion as Benign or Malignant
    - Display prediction confidence
    - Visualize Grad-CAM heatmap for explainability
  ui_layout: |
    ----------------------------------------------
    | Upload Image: [Choose File]                |
    |                                             |
    | [Predict]                                  |
    | Prediction: Malignant                      |
    | Confidence: 94%                            |
    | [Show Heatmap]                             |
    ----------------------------------------------

results:
  expected_metrics:
    accuracy: "90–92%"
    precision: "88%"
    recall: "91%"
    f1_score: "89%"
    roc_auc: "0.93"
  note: "Results may vary depending on preprocessing and hyperparameter tuning."

responsible_ai:
  fairness: "Model trained on diverse skin tones to minimize bias."
  transparency: "Predictions include probability and Grad-CAM visualization."
  privacy: "Dataset is de-identified and used only for research purposes."
  disclaimer: >
    This project is for educational and research purposes only.
    It is not intended for clinical use or medical diagnosis.

future_work:
  - "Expand dataset for improved generalization"
  - "Add multi-class classification for all 6 lesion types"
  - "Integrate Grad-CAM visualization in UI"
  - "Deploy Streamlit app on Streamlit Cloud or Hugging Face Spaces"

author:
  name: "Aditya Adke"
  role: "Graduate Student — Machine Learning 2"
  university: "University of Florida"
  email: "adityaadke123@gmail.com"
  github: "https://github.com/AdityaAdke123"

license:
  type: "MIT License"
  file: "LICENSE"

acknowledgments:
  - "PAD-UFES-20 Dataset Authors: Pereira et al. (2020)"
  - "University of Florida – Machine Learning 2 Course Project"
  - "TensorFlow and Streamlit open-source communities"

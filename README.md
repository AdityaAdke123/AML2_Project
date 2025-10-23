project:
  title: "AML2_Project"
  subtitle: "Automated Skin Cancer Detection Using CNNs"

overview:
  description: >
    This project implements a Convolutional Neural Network (CNN) to classify
    skin lesion images as benign or malignant, aiming to support early detection
    of skin cancer. Using deep learning, the model analyses dermatological images
    from the PAD-UFES-20 dataset to identify potentially cancerous lesions and assist
    in medical screening.

objectives:
  - Develop a CNN-based image classifier for skin lesion detection.
  - Train and evaluate the model using real-world dermatological images.
  - Provide a Streamlit web interface for users to upload and test images interactively.
  - Analyse model interpretability through visualisation (Grad-CAM heatmaps).

dataset:
  name: "PAD-UFES-20"
  source: "https://data.mendeley.com/datasets/zr7vgbcyr2/1"
  details:
    total_images: 2298
    patients: 1373
    description: "Smartphone images with corresponding CSV metadata for each lesion."
    classes:
      benign:
        - "NEV (Nevus)"
        - "ACK (Actinic Keratosis)"
        - "SEK (Seborrheic Keratosis)"
      malignant:
        - "BCC (Basal Cell Carcinoma)"
        - "MEL (Melanoma)"
        - "SCC (Squamous Cell Carcinoma)"

structure: |
  AML2_Project/
  │
  ├── data/                # Raw or sample data
  │   ├── metadata.csv
  │   └── images/          # Image dataset (LFS-tracked)
  │       ├── imgs_part_1/
  │       ├── imgs_part_2/
  │       └── imgs_part_3/
  │
  ├── notebooks/           # Jupyter notebooks
  │   └── setup.ipynb      # Data exploration, preprocessing, training
  │
  ├── src/                 # Helper scripts, model architecture, data pipeline
  │   ├── cnn_model.py
  │   └── preprocess.py
  │
  ├── ui/                  # Streamlit interface
  │   └── app.py
  │
  ├── results/             # Visual outputs, metrics, plots
  │
  ├── docs/                # Diagrams, blueprints, reports
  │
  ├── README.md            # Project documentation
  └── requirements.txt     # List of dependencies

contents:
  - file: "requirements.txt"
    purpose: "Lists all dependencies required to run the project."
  - file: "notebooks/setup.ipynb"
    purpose: >
      Demonstrates dataset loading, early exploration, and model training.
      Includes summary statistics, class distribution, and initial results.
  - note: "Code executes cleanly and produces visible output, such as sample predictions and accuracy plots."

installation_and_setup:
  steps:
    - step: "Clone the repository"
      command: |
        git clone https://github.com/AdityaAdke123/AML2_Project.git
        cd AML2_Project
    - step: "Install dependencies"
      command: |
        pip install -r requirements.txt
    - step: "Run the notebook"
      command: |
        jupyter notebook notebooks/setup.ipynb
    - step: "Launch Streamlit interface"
      command: |
        streamlit run ui/app.py

results:
  model_metrics_expected:
    - metric: "Accuracy"
      value: "90–92%"
    - metric: "Precision"
      value: "88%"
    - metric: "Recall"
      value: "91%"
    - metric: "F1-Score"
      value: "89%"
    - metric: "ROC-AUC"
      value: "0.93"
  outputs:
    - "Confusion matrix"
    - "Sample predictions with confidence scores"
    - "Grad-CAM heatmaps for interpretability"

responsible_ai_reflection:
  fairness: "Dataset includes diverse skin tones and demographics."
  transparency: "Visual heatmaps illustrate model focus regions."
  privacy: "All patient data is de-identified and used solely for research."
  disclaimer: "This project is for educational and research use only, not medical diagnosis."

author:
  name: "Aditya Adke"
  course: "Machine Learning 2 — Fall 2025"
  university: "University of Florida"
  email: "adityaadke123@gmail.com"
  github: "https://github.com/AdityaAdke123"

license:
  type: "MIT License"
  file: "LICENSE"
  description: "Permission is granted to use, copy, modify, and distribute this software for educational purposes."

acknowledgments:
  - "PAD-UFES-20 dataset authors (Pereira et al., 2020)"
  - "TensorFlow and Streamlit open-source communities"
  - "University of Florida — Applied Machine Learning 2 coursework"

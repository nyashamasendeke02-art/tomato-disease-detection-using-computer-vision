# tomato-disease-detection-using-computer-vision
This repository contains dataset and model for tomato disease detection on raspberry pi using transfer learning.  
The dataset for this project was taken from Kaggle tomato disease dataset.  
A pretrained model was used as the base model for transfer learning (MobileNet) optimised for embedded harware.  
A transfer layer was added to the base model for disease detection.  
Model transfer learning on raspberry pi was evaluated using software for resource monitoring during training.  
The model was tested on real tomato leaves in an experimental tomato garden with infected tomato plants.  
The models confidence score/ predictions were not that far off.  
The major diseases which were being investigated were early blight and late blight.  
Model inference was also evaluated noting parameters such as latency.  
Hardware considerations were taken into account to reduce latencies during inference

project-root/
│
├── data/
│   ├── raw/                # Original, unchanged data
│   ├── processed/          # Cleaned, resized, or augmented data
│   │   ├── train/
│   │   │   ├── class_a/    # Images/files of class A
│   │   │   └── class_b/    # Images/files of class B
│   │   ├── val/
│   │   │   ├── class_a/
│   │   │   └── class_b/
│   │   └── test/           # Optional: For final evaluation
│   │       ├── class_a/
│   │       └── class_b/
│
├── notebooks/              # Jupyter notebooks for EDA and prototyping
│
├── src/                    # Source code for the project
│   ├── __init__.py         # Make src a Python module
│   ├── data/               # Data handling utilities and preprocessing
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   └── model.py
│   ├── utils/              # Utility functions (logging, metrics, etc.)
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── train.py            # Training pipeline
│   └── predict.py          # Inference script
│
├── models/                 # Saved model weights (.pth, .h5, etc.)
│
├── tests/                  # Unit and integration tests
│   ├── test_data.py
│   └── test_models.py
│
├── docs/                   # Documentation files (Markdown, configuration for Sphinx, etc.)
│   └── index.md
│
├── scripts/                # Utility scripts (e.g., data download, environment setup)
│   ├── download_data.sh
│   └── setup_environment.sh
│
├── configs/                # Configuration files (YAML, JSON, etc.)
│   └── default.yaml
│
├── .gitignore              # Ignore files not to be committed
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment (optional)
├── Dockerfile              # Dockerfile for containerization
├── setup.py                # Setup script for packaging (if needed)
├── LICENSE                 # License file
└── README.md               # Project description

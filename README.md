# Autonomous Driving Object Detection

## Project Overview
This project focuses on **object detection for autonomous driving** using **YOLOv8**. The goal is to detect vehicles, pedestrians, traffic signs, and other relevant objects in real-time to facilitate autonomous navigation. The project includes **data preprocessing, model training and fine-tuning, visualization, and deployment** via a Flask web application.  

All notebooks are designed to run on **Kaggle Notebooks**, which provides GPU support for faster training and inference.

---

## Folder Structure

Real-Time-Object-Detection-for-Autonomous-Vehicles/
│
├── Data/
│ ├── bdd-in-yolo-fromat.ipynb # Notebook to convert BDD dataset annotations to YOLO format (run on Kaggle)
│ ├── dataset_utils.py # Utility functions used in data preprocessing
│
├── Deployment/
│ ├── YOLOV8s_best.onnx # Trained YOLOv8 model in ONNX format for deployment
│ ├── app.py # Flask web app for real-time object detection
│
├── Model/
│ ├── Visualization output/ # Folder containing visualization outputs
│ │ ├── image.png # Sample predicted image
│ │ └── Video.mp4 # Sample predicted video
│ │
│ ├── train.py # Training helper functions
│ ├── training experiment.ipynb # Notebook for model training and hyperparameter tuning using Optuna (run on Kaggle)
│ ├── visualization.ipynb # Notebook for predicting and visualizing objects on images and videos (run on Kaggle)
│ ├── YOLOV8s_best.pt # Trained YOLOv8 PyTorch model
│
├── requirements.txt # Python dependencies


---

## Key Components

### 1. Data Preprocessing
- **bdd-in-yolo-fromat.ipynb**: Converts BDD dataset labels into YOLO format. Designed for Kaggle notebooks for easy GPU usage.
- **dataset_utils.py**: Contains helper functions for data conversion and preprocessing.

### 2. Model Training & Visualization
- **training experiment.ipynb**: Notebook for training YOLOv8 with Optuna for hyperparameter optimization. Run on Kaggle for GPU acceleration.
- **train.py**: Helper functions used in the training notebook.
- **visualization.ipynb**: Runs inference on images and videos and visualizes bounding boxes. Run on Kaggle for GPU acceleration.
- **Visualization output/**: Contains sample outputs for quick reference.

### 3. Deployment
- **app.py**: Flask-based web application for deploying the trained YOLOv8 model locally or on a server.
- **YOLOV8s_best.onnx**: Exported model in ONNX format for fast inference during deployment.

---

## Installation

1. Clone the repository:
    git clone <repository_url>
    cd my_repo

2. Create a virtual environment (optional but recommended):
    python -m venv venv
    source venv/bin/activate      # On Windows use `venv\Scripts\activate`

3. Install dependencies:
    pip install -r requirements.txt
    Note: Notebooks are designed to run on Kaggle, which comes with pre-installed packages and GPU support. Use a local environment mainly for deployment.
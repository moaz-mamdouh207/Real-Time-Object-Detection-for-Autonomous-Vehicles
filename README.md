# Autonomous Driving Object Detection with YOLOv8

This project implements a real-time object detection system for autonomous driving applications using YOLOv8s trained on the BDD100K dataset. The system can detect various objects commonly encountered in driving scenarios and provides a web interface for easy interaction.

## 🚀 Project Overview

This object detection system is specifically designed for autonomous driving scenarios, capable of identifying vehicles, pedestrians, traffic signs, and other relevant objects in diverse environmental conditions.

### Key Features
- **Real-time object detection** for autonomous driving
- **Web-based interface** for easy testing and demonstration
- **Diverse weather and time condition handling**
- **Optimized for performance** with YOLOv8s architecture
- **Containerized deployment** with Docker

## 📊 Model & Dataset

### Model: YOLOv8s
- **Why YOLOv8s?** - Excellent balance between speed and accuracy, making it ideal for real-time autonomous driving applications
- **Architecture** - Optimized small version of YOLOv8 with high efficiency
- **Performance** - Suitable for deployment on edge devices with limited computational resources

### Dataset: BDD100K
- **Why BDD100K?** - Contains diverse driving scenarios across different:
  - **Time zones** (day, night, dawn, dusk)
  - **Weather conditions** (sunny, rainy, snowy, foggy)
  - **Geographic locations**
  - **Driving scenarios** (highway, urban, residential)

## 📈 Model Performance

training metrics:
Box Loss: 1.32682
Classification Loss: 0.80327
Distribution Focal Loss: 1.03117
Precision: 0.70327
Recall: 0.49147
mAP@0.5: 0.5508
mAP@0.5:0.95: 0.31659

## 🗂️ repo Structure
├── train/ # Training scripts and utilities
│ ├── stage1.ipynb # Initial training stage
│ ├── stage2.ipynb # Fine-tuning stage
│ └── train_utils.py # Training utilities and functions
├── web_page/ # Web application
│ ├── model/
│ │ └── final.pt # Trained YOLOv8s model
│ ├── templates/
│ │ └── index.html # Web interface
│ ├── uploads/ # Directory for uploaded images
│ ├── app.py # Flask application
│ ├── Dockerfile # Container configuration
│ └── requirements.txt # Python dependencies
├── data/ # Data processing
│ ├── bdd-in-yolo-format.ipynb # Dataset conversion notebook
│ ├── data_utils/ # Data processing utilities
│ └── data_config.py # Dataset configuration and class names
└── requirements.txt # Main project dependencies


## Installation & Setup

### Prerequisites
- Python 3.8+
- Kaggle environment (for training notebooks)
- Docker (for deployment)

### Local Installation

1. **Clone the repository**
    git clone https://github.com/moaz-mamdouh207/Real-Time-Object-Detection-for-Autonomous-Vehicles
    cd autonomous-driving-object-detection

2. **Install dependencies**
    pip install -r requirements.txt

3. **Run the web application**
    cd web_page
    python app.py

## Docker Deployment
1. **Build the Docker image**
    cd web_page
    docker build -t autonomous-driving-detection .

2. **Run the container**
    docker run -p 5000:5000 autonomous-driving-detection

3. **Access the application**
    Open your browser and navigate to http://localhost:5000
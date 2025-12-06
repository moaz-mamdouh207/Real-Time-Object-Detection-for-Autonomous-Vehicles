# Autonomous Driving Object Detection System

An object detection system specifically designed for autonomous driving scenarios, capable of identifying vehicles, pedestrians, traffic signs, and other relevant objects in diverse environmental conditions. The system uses **YOLOv8s** for its small size and high speed, making it suitable for real-time applications and edge devices.

---

## 🗂 Repository Structure

![Error](Repo_structure.png)



---

## 📦 Dataset Preparation

The **BDD100K** dataset is used for training. Since the original dataset is not YOLO-ready, custom functions were implemented:

1. **Dataset Reorganization**  
   - Scan raw files and identify images and labels  
   - Create a clean folder structure  
   - Copy files to correct locations  
   - Separate images by time of day (day, night, dawn & dusk)  

2. **Train/Validation/Test Split**  
   - Shuffle the dataset to avoid bias  
   - Split into train, validation, and test sets  
   - Ensure images and matching labels remain paired  

3. **BDD100K JSON → YOLO Conversion**  
   - Load JSON annotations  
   - Map categories to YOLO class IDs  
   - Convert bounding boxes to YOLO format `(class_id x_center y_center width height)`  
   - Save `.txt` labels matching image filenames  

**Kaggle Notebook:** [View on Kaggle](https://www.kaggle.com/code/moazmamdouh205/depi-cleaning/edit)

---

## 🏋️ Training

Training is split into **two stages** to handle dataset size and class imbalance:

### Stage 1
- Train YOLOv8s on a **balanced subset** of the dataset to learn unbiased features.  
- The best checkpoint from this stage is used for Stage 2.  

### Stage 2
- Fine-tune the Stage 1 model on the **full dataset**, applying augmentations to improve robustness and fix class imbalance.  

**Training Utilities:** `train_utils.py` – contains helper functions for dataset sampling and preprocessing.  

---

## 🌐 Web Application

A web interface allows users to upload a video, process it through the trained model, and view the output with detected objects highlighted.  

- **Docker Port:** 5000  
- **Setup:**  
```bash
git clone <repo_url>
cd web_app
docker build -t autonomous-detection .
docker run -p 5000:5000 autonomous-detection




# Dataset Preparation for BDD100K – YOLOv8

This directory contains utilities to prepare the **BDD100K autonomous driving dataset** for YOLOv8 training. The original dataset is not YOLO-ready [View on Kaggle](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k), so custom functions were implemented for:

**1. Dataset Reorganization**  
- Scan raw files, identify images and labels  
- Create clean folder structure  
- Copy files to correct locations  
- Separate images by time of day (day, night, dawn&dusk)  

**2. Train/Val/Test Split**  
- Shuffle dataset to avoid bias  
- Split into train/validation/test sets  
- Move images and matching labels while keeping pairs consistent  

**3. BDD100K JSON → YOLO Conversion**  
- Load JSON annotations  
- Map categories to YOLO class IDs  
- Convert bounding boxes to YOLO format `(class_id x_center y_center width height)`  
- Save `.txt` labels matching image filenames  

**Kaggle Notebook:**  [View on Kaggle](https://www.kaggle.com/code/moazmamdouh205/depi-cleaning/edit)

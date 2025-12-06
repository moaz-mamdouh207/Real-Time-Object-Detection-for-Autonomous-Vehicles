## 📘 Overview
The training is split into two stages to handle dataset size and class imbalance:

### **Stage 1 (stage1.ipynb)**  
Train YOLOv8s on a **balanced subset** of the dataset to learn clean, unbiased features.  
The best checkpoint from this stage is used for Stage 2.

### **Stage 2 (stage2.ipynb)**  
Fine-tune the Stage 1 model on the **full dataset**, applying augmentations to improve robustness and fix class imbalance.

## 🧰 train_utils.py
Contains all helper functions used across both stages:
- dataset sampling utilities  

## ▶️ Kaggle Notebook
A ready-to-run Kaggle version of the full pipeline:  
**[https://www.kaggle.com/code/moazmamdouh205/depi-training-1?scriptVersionId=284213465]**


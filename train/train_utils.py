import os
import random
import shutil
import yaml

import data_config

def create_yaml(path):
    dataset_yaml = {
        "path": path,
        "train": f"{path}/train/images",
        "val": f"{path}/val/images",
        "nc": 10,
        "names": classes
    }
    
    with open("dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)

def sample(output_dir, image_folders, label_folders, val_image_folder, val_label_folder, size = 512):

    paths = {
        "train_images" : f"{output_dir}/train/images", 
        "train_labels" : f"{output_dir}/train/labels",
    } 
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    n = len(image_folders)
    samples_per_folder = [(size // n) + (i < size % n) for i in range(n)]


    for img_folder, lbl_folder, n_samples in zip(image_folders, label_folders, samples_per_folder):
        all_images = [f for f in os.listdir(img_folder) if f.endswith(".jpg")]
        sampled_images = random.sample(all_images, n_samples)

        for img_file in sampled_images:
            
            shutil.copy2(os.path.join(img_folder, img_file),
                         os.path.join(paths["train_images"], img_file))
            
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            shutil.copy2(os.path.join(lbl_folder, lbl_file),
                         os.path.join(paths["train_labels"], lbl_file))
            
    shutil.copytree(val_image_folder, paths["val_images"], dirs_exist_ok=True)
    shutil.copytree(val_label_folder, paths["val_labels"], dirs_exist_ok=True)

    create_yaml(output_dir)




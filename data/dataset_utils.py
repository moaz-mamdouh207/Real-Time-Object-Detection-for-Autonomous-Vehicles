import os
import json
from pathlib import Path
import random
from tqdm import tqdm
import cv2
import shutil

def Create_output_directories(output_folder):
    """
    Create the directory structure for YOLO dataset splits.
        output_folder/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/

    Args:
        output_folder (str): Path to the base output folder where the directories will be created.

    Returns:
        None
    """
    splits=["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_folder, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labels", split), exist_ok=True)


def convert_bbox(bbox, img_width, img_height):
    """
    Convert BDD100K bbox to YOLO format
    BDD100K bbox format: [x1, y1, x2, y2]
    YOLO format: class x_center y_center width height (normalized)
    """
    x, y, w, h = bbox
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


def split_dataset(json_file, train_ratio=0.7, val_ratio=0.2):
    """
    Shuffle and split the dataset into train, val, and test.
    
    Args:
        json_file (Path or str): Path to the BDD100K JSON file.
        train_ratio (float): Fraction of items for training set.
        val_ratio (float): Fraction of items for validation set.
        
    Returns:
        dict: Dictionary with keys "train", "val", "test" containing split items.
    """
    with open(json_file) as f:
        data = json.load(f)

    random.shuffle(data)
    total = len(data)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


def process_split(items, img_dir, output_folder, split, class_map):
    """
    Process a dataset split: copy images, convert annotations to YOLO format, save labels.
    
    Args:
        items (list): List of annotation items for this split.
        img_dir (Path): Path to source images.
        output_folder (str): Path to the base output folder.
        split: split name.
        class_map (dict): Mapping of category names to class IDs.
    """
    img_output_dir = os.path.join(output_folder, "images", split)
    lbl_output_dir = os.path.join(output_folder, "labels", split)

    for item in tqdm(items, desc=f"Processing {img_output_dir.name}"):
        img_name = item['name']
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        
        # Copy image to output folder
        shutil.copy(img_path, img_output_dir / img_name)
        
        # Read image size
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        # Convert labels to YOLO format
        yolo_lines = []
        for label in item.get('labels', []):
            category = label.get('category')
            bbox = label.get('box2d')
            if category not in class_map or not bbox:
                continue
            xmin, ymin = bbox['x1'], bbox['y1']
            xmax, ymax = bbox['x2'], bbox['y2']
            yolo_bbox = convert_bbox([xmin, ymin, xmax - xmin, ymax - ymin], w, h)
            yolo_lines.append(f"{class_map[category]} " + " ".join(map(str, yolo_bbox)))
        
        # Save YOLO label file if there are labels
        if yolo_lines:
            with open(lbl_output_dir / img_name.replace(".jpg", ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))


def create_yaml(output_dir, class_names):
    """
    Create a YOLO data.yaml file.
    
    Args:
        output_dir (Path): Base output directory containing images/train, images/val, images/test.
        class_names (list): List of class names.
    """
    yaml_content = f"""
                    train: {output_dir}/images/train
                    val: {output_dir}/images/val
                    test: {output_dir}/images/test

                    nc: {len(class_names)}  
                    names: {class_names}
                    """
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)


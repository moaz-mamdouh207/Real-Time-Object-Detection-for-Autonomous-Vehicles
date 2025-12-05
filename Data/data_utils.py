import json
import os
import shutil
from pathlib import Path
import random

categories = {
    "bike": 0,
    "bus": 1,
    "car": 2,
    "motor": 3,
    "person": 4,
    "rider": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "train": 8,
    "truck": 9
}

def organize_train_data(json_path, images_root, output_folder):
    """
    Organize training images into subfolders by time of day and generate corresponding JSON files.

    Args:
        json_path (Path or str): Path to the JSON file containing image metadata.
        images_root (Path): Root folder containing all images.
        output_folder (Path or str): Folder where organized images and JSON files will be saved.

    Process:
        1. Collect all image filenames from the JSON.
        2. Copy images from images_root to a temporary folder.
        3. Move images into subfolders based on 'timeofday' attribute.
        4. Create a reduced JSON per time of day with image names and labels.
        5. Remove the temporary folder after organizing.
    """

    temp_output_folder = output_folder / "all_images"
    os.makedirs(temp_output_folder, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    target_names = {item["name"] for item in data}

    for path in images_root.rglob("*"):
        if path.name in target_names:
            shutil.copy(path, temp_output_folder / path.name)

    missing_images = len(target_names) - len(os.listdir(temp_output_folder))
    print(f"{missing_images} images missing")

    grouped_json = {}

    for entry in data:
        name = entry.get("name")
        attributes = entry.get("attributes", {})
        timeofday = attributes.get("timeofday", "unknown")
        labels = entry.get("labels", [])

        if timeofday == "undefined":
            timeofday = "night"     
        elif timeofday == "dawn/dusk":
            timeofday = "dawn&dusk"
            
        sub_folder_path = os.path.join(output_folder, timeofday)
        os.makedirs(sub_folder_path, exist_ok=True)

        src_path = Path(temp_output_folder) / name
        dest_path = Path(sub_folder_path) / name

        if src_path.exists():
            shutil.move(str(src_path), str(dest_path))

        reduced_entry = {
            "name": name,
            "labels": labels
        }
        grouped_json.setdefault(timeofday, []).append(reduced_entry)

    for timeofday, items in grouped_json.items():
        output_json_path = os.path.join(output_folder, f"{timeofday} labels.json")
        with open(output_json_path, "w") as f:
            json.dump(items, f, indent=4)

    shutil.rmtree(temp_output_folder)


def val_test_split(json_path, images_root, output_folder, split_ratio=0.8): 
    """
    Split a dataset into validation and test sets, copy images, and generate corresponding JSON files.

    Args:
        json_path (Path or str): Path to the JSON file containing image metadata.
        images_root (Path): Root directory where the original images are stored.
        output_folder (Path or str): Directory where the validation and test folders and JSON files will be saved.
        split_ratio (float, optional): Proportion of images to assign to the test set. Default is 0.8.

    Process:
        1. Load image metadata from the JSON file.
        2. Shuffle the images and split into test and validation sets according to split_ratio.
        3. Copy the images to their respective folders ('test' or 'val').
        4. Generate reduced JSON files containing image names and labels for each split.
    """

    test_folder = output_folder / "test"
    val_folder = output_folder / "val"
    
    test_folder.mkdir(parents=True, exist_ok=True)
    val_folder.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_dict = {item["name"]: item for item in data}
    target_names = list(data_dict.keys())
    
    random.shuffle(target_names)
    
    split_index = int(len(target_names) * split_ratio)
    test_images = target_names[:split_index]
    val_images = target_names[split_index:]
    
    def copy_and_build_json(images_list, folder, output_file):
        split_json = []
        for name in images_list:
            src_path = images_root / name
            dest_path = folder / name
            if src_path.exists():
                shutil.copy(src_path, dest_path)
            
            entry = data_dict[name]
            reduced_entry = {
                "name": name,
                "labels": entry.get("labels", [])
            }
            split_json.append(reduced_entry)
        
        with open(output_file, 'w') as f:
            json.dump(split_json, f, indent=4)
    
    copy_and_build_json(test_images, test_folder, output_folder / "test labels.json")
    copy_and_build_json(val_images, val_folder, output_folder / "val labels.json")

def bdd_to_yolo(bdd_json_path, output_folder, categories):
    """
    Convert BDD100K format to YOLO format.
    
    Params:
        bdd_json_path (str)  : path to BDD100K JSON label file
        output_folder (str)  : folder to save YOLO .txt files
        categories (dict)    : {"car": 0, "person": 1, ...}
    """

    os.makedirs(output_folder, exist_ok=True)

    with open(bdd_json_path, "r") as f:
        data = json.load(f)

    for item in data:
        img_name = item["name"]
        labels = item.get("labels", [])

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_name)

        yolo_lines = []

        img_w = item["width"] if "width" in item else 1280
        img_h = item["height"] if "height" in item else 720

        for obj in labels:
            if "box2d" not in obj:
                continue

            cls = obj["category"]
            if cls not in categories:
                continue

            cls_id = categories[cls]

            x1 = obj["box2d"]["x1"]
            y1 = obj["box2d"]["y1"]
            x2 = obj["box2d"]["x2"]
            y2 = obj["box2d"]["y2"]

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
    os.remove(bdd_json_path)

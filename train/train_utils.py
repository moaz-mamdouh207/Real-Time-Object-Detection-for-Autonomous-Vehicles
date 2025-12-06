import os
import random
import shutil
import yaml

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

def create_yaml(path):
    dataset_yaml = {
        "path": path,
        "train": f"{path}/train/images",
        "val": f"{path}/val/images",
        "nc": len(categories),
        "names": categories
    }
    
    with open(f"{path}/dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)


def sample(data_set_dir, image_folders, label_folders, val_image_folder, val_label_folder, size=512, mode="sample"):
    """
    Create a YOLO-style dataset by moving or sampling images and labels
    from multiple source folders.

    This function copies training images and labels into a unified dataset 
    structure:
        data_set_dir/
            train/images/
            train/labels/
            val/images/
            val/labels/

    Two modes are supported:

    1. mode="all"
       - Copies all images from each folder in `image_folders` and all labels from `label_folders`.

    2. mode="sample"
       - Evenly samples a total of `size` images across all folders.
       - For example, if size=300 and there are 3 folders, each folder will 
         contribute 100 images (±1 when not divisible evenly).

    Parameters
    ----------
    data_set_dir : str
        Output directory where the YOLO dataset structure will be created.

    image_folders : list of str
        List of paths to folders containing images.

    label_folders : list of str
        List of paths to folders containing matching labels (same order as 
        `image_folders`).

    val_image_folder : str
        Path to the validation images folder.

    val_label_folder : str
        Path to the validation labels folder.

    size : int, optional (default=512)
        Total number of images to sample when mode="sample".

    mode : str, optional (default="sample")
        Either:
        - "sample" : evenly divide `size` across folders
        - "all"    : copy all images and labels from all folders
    """

    paths = {
        "train_images": f"{data_set_dir}/train/images",
        "train_labels": f"{data_set_dir}/train/labels",
        "val_images": f"{data_set_dir}/val/images",
        "val_labels": f"{data_set_dir}/val/labels",
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    # -------------- MOVE ALL ------------------
    if mode == "all":
        for img_folder, lbl_folder in zip(image_folders, label_folders):

            all_images = os.listdir(img_folder)

            for img_file in all_images:
                shutil.copy2(
                    os.path.join(img_folder, img_file),
                    os.path.join(paths["train_images"], img_file)
                )

                lbl_file = img_file.rsplit(".", 1)[0] + ".txt"
                shutil.copy2(
                    os.path.join(lbl_folder, lbl_file),
                    os.path.join(paths["train_labels"], lbl_file)
                )

    # ------------- EQUALLY DIVIDED SAMPLE ---------------
    else:
        n = len(image_folders)
        samples_per_folder = [(size // n) + (i < size % n) for i in range(n)]

        for img_folder, lbl_folder, n_samples in zip(image_folders, label_folders, samples_per_folder):

            all_images = os.listdir(img_folder)
            sampled = random.sample(all_images, n_samples)

            for img_file in sampled:
                shutil.copy2(
                    os.path.join(img_folder, img_file),
                    os.path.join(paths["train_images"], img_file)
                )

                lbl_file = img_file.rsplit(".", 1)[0] + ".txt"
                shutil.copy2(
                    os.path.join(lbl_folder, lbl_file),
                    os.path.join(paths["train_labels"], lbl_file)
                )

    shutil.copytree(val_image_folder, paths["val_images"], dirs_exist_ok=True)
    shutil.copytree(val_label_folder, paths["val_labels"], dirs_exist_ok=True)

    create_yaml(data_set_dir)


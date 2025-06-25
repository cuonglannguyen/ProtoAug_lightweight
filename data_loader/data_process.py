#/cm/shared/cuongnl8/APT/caltech-101/101_ObjectCategories
#/cm/shared/cuongnl8/Cuong-thesis/data/caltech_16shots_seed0
#/cm/shared/cuongnl8/Cuong-thesis/data/split_caltech.csv"
import os
import shutil
import random
import csv

# Set the random seed for reproducibility
random.seed(0)

# Paths
root_dir = "/cm/shared/cuongnl8/APT/caltech-101/101_ObjectCategories"
output_dir = "/cm/shared/cuongnl8/Cuong-thesis/data/caltech_16shots_seed0"
train_image_dir = os.path.join(output_dir, "images", "train")
val_image_dir = os.path.join(output_dir, "images", "val")
split_file = "/cm/shared/cuongnl8/Cuong-thesis/data/split_caltech.csv"

# Ignore Faces_easy folder
ignore_folder = "Faces_easy"

# Ensure the output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

# Step 1: Process the dataset and sample 16 images from each folder (excluding Faces_easy)
subfolders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f != ignore_folder])
print(len(subfolders))
# Write metadata for the classes in meta.txt
with open(os.path.join(output_dir, "meta.txt"), "w") as meta_file:
    for idx, folder in enumerate(subfolders):
        meta_file.write(f"{folder}\n")

# Dictionary to hold the class label mapping
label_dict = {folder: idx for idx, folder in enumerate(subfolders)}

# Prepare to store image mappings for training and testing
train_images = {}
test_images = {}

for folder in subfolders:
    folder_path = os.path.join(root_dir, folder)
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Step 2: Sample 16 images for training
    sampled_train = random.sample(images, 16)
    
    # Store train images in dictionary
    for img in sampled_train:
        full_path = os.path.join(folder, img)  # Store both folder and image name
        train_images[full_path] = label_dict[folder]
    
    # The remaining images go to the validation/test set
    remaining_images = set(images) - set(sampled_train)
    for img in remaining_images:
        full_path = os.path.join(folder, img)  # Store both folder and image name
        test_images[full_path] = label_dict[folder]

# Step 3: Copy images to the respective directories and create train.txt and test.txt
train_txt = open(os.path.join(output_dir, "train.txt"), "w")
test_txt = open(os.path.join(output_dir, "test.txt"), "w")

# Copy training images
for idx, (img_path, label) in enumerate(train_images.items()):
    img_filename = f"train_{idx}.jpg"
    src_img_path = os.path.join(root_dir, img_path)  # Rebuild the full path
    shutil.copy(src_img_path, os.path.join(train_image_dir, img_filename))
    train_txt.write(f"{img_filename}----{label}\n")

# Copy test/validation images
for idx, (img_path, label) in enumerate(test_images.items()):
    img_filename = f"val_{idx}.jpg"
    src_img_path = os.path.join(root_dir, img_path)  # Rebuild the full path
    shutil.copy(src_img_path, os.path.join(val_image_dir, img_filename))
    test_txt.write(f"{img_filename}----{label}\n")

train_txt.close()
test_txt.close()

print(f"Processing complete. Data saved to: {output_dir}")

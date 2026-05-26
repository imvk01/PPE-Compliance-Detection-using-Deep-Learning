import os
import random
import shutil

images_dir = "dataset/images"
labels_dir = "dataset/labels"

train_img_dir = "dataset/train/images"
train_lbl_dir = "dataset/train/labels"
valid_img_dir = "dataset/valid/images"
valid_lbl_dir = "dataset/valid/labels"

image_files = [f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

random.shuffle(image_files)

split_index = int(len(image_files) * 0.8)

train_files = image_files[:split_index]
valid_files = image_files[split_index:]

def copy_files(files, img_dest, lbl_dest):
    for img_file in files:
        name = os.path.splitext(img_file)[0]
        label_file = name + ".txt"

        shutil.copy(os.path.join(images_dir, img_file), os.path.join(img_dest, img_file))

        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(lbl_dest, label_file))

copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(valid_files, valid_img_dir, valid_lbl_dir)

print("Dataset split completed.")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(valid_files)}")
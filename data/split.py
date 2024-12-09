import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

base_path = '/Users/sosen/UniProjects/eng-thesis/data/datasets-unsplit/AC8-rejected/CROPPED/NON-STED'
output_base_path = '/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-rejected/NON-STED'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(output_base_path)

classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]  # Filter to include directories only

train_ratio = 0.85
test_ratio = 0.15

for class_name in classes:
    class_dir = os.path.join(base_path, class_name)
    train_dir = os.path.join(output_base_path, 'train_ds', class_name)
    test_dir = os.path.join(output_base_path, 'test_ds', class_name)
    
    ensure_dir(train_dir)
    ensure_dir(test_dir)
    
    # Filtering the images to exclude files like .DS_Store
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if not img.startswith('.')]
    
    train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=123)
    
    def copy_files(files, target_dir):
        for file in files:
            shutil.copy(file, target_dir)
    
    copy_files(train_images, train_dir)
    copy_files(test_images, test_dir)
    
    print(f"Processed class {class_name}:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
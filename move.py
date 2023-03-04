import os
import shutil
import pandas as pd

# Define the paths to the CSV files and image directories
train_csv_file = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/train_data.csv'
test_csv_file = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/test_data.csv'
source_dir_h = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/TBX11K/imgs/health'
source_dir_tb = "/home/yufei.zhang/Documents/HC/assignment_2/task_2/TBX11K/imgs/tb"
target_dir = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/dataset'

# Define the names of the two subdirectories
train_dir = os.path.join(target_dir, 'train')
test_dir = os.path.join(target_dir, 'test')

# Create the subdirectories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load the filenames and labels from the CSV files
train_data = pd.read_csv(train_csv_file)
test_data = pd.read_csv(test_csv_file)

# Copy the training images to the training directory
for i, row in train_data.iterrows():
    filename = row['filename']
    label = row['label']
    if filename[0] == "h":
        src_path = os.path.join(source_dir_h, filename)
        dst_path = os.path.join(train_dir, filename)
        shutil.copyfile(src_path, dst_path)
    else:
        src_path = os.path.join(source_dir_tb, filename)
        dst_path = os.path.join(train_dir, filename)
        shutil.copyfile(src_path, dst_path)

# Copy the testing images to the testing directory
for i, row in test_data.iterrows():
    filename = row['filename']
    label = row['label']
    if filename[0] == "h":
        src_path = os.path.join(source_dir_h, filename)
        dst_path = os.path.join(test_dir, filename)
        shutil.copyfile(src_path, dst_path)
    else:
        src_path = os.path.join(source_dir_tb, filename)
        dst_path = os.path.join(test_dir, filename)
        shutil.copyfile(src_path, dst_path)

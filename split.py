import os
import pandas as pd

# Set paths for the image directories
healthy_dir = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/TBX11K/imgs/health'
tb_dir = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/TBX11K/imgs/tb'

# Get a list of filenames for each class
healthy_filenames = sorted(os.listdir(healthy_dir))
tb_filenames = sorted(os.listdir(tb_dir))

# Calculate the number of images for testing and training sets
num_test_healthy = int(len(healthy_filenames) * 0.2)
num_train_healthy = len(healthy_filenames) - num_test_healthy
num_test_tb = int(len(tb_filenames) * 0.2)
num_train_tb = len(tb_filenames) - num_test_tb

# Split the filenames into testing and training sets
test_healthy = healthy_filenames[:num_test_healthy]
train_healthy = healthy_filenames[num_test_healthy:]
test_tb = tb_filenames[:num_test_tb]
train_tb = tb_filenames[num_test_tb:]

# Combine the filenames and class labels into a single dataframe
test_data = pd.DataFrame({
    'filename': test_healthy + test_tb,
    'label': ['Healthy'] * num_test_healthy + ['TB'] * num_test_tb
})
train_data = pd.DataFrame({
    'filename': train_healthy + train_tb,
    'label': ['Healthy'] * num_train_healthy + ['TB'] * num_train_tb
})

# Save the dataframes as CSV files
test_data.to_csv('/home/yufei.zhang/Documents/HC/assignment_2/task_2/test_data.csv', index=False)
train_data.to_csv('/home/yufei.zhang/Documents/HC/assignment_2/task_2/train_data.csv', index=False)





# Report the number and range of filenames for each class in the training and testing sets
print(f'Number of Healthy images in training set: {num_train_healthy}')
print(f'Range of filenames for Healthy images in training set: {train_healthy[0]} to {train_healthy[-1]}')
print(f'Number of TB images in training set: {num_train_tb}')
print(f'Range of filenames for TB images in training set: {train_tb[0]} to {train_tb[-1]}')
print(f'Number of Healthy images in testing set: {num_test_healthy}')
print(f'Range of filenames for Healthy images in testing set: {test_healthy[0]} to {test_healthy[-1]}')
print(f'Number of TB images in testing set: {num_test_tb}')
print(f'Range of filenames for TB images in testing set: {test_tb[0]} to {test_tb[-1]}')

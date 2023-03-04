# X-ray-classification-resnet50
Using ResNet50 to do classification on X-ray images

dataset : X-ray Tuberculosis dataset   (https://www.kaggle.com/datasets/usmanshams/tbx-11) 

Task : It contains many X-ray images from multiple classes, but we will only work with two classes (Healthy and TB) for binary classification. There are 3800 healthy x-ray images and 800 TB x-rays.

1. Use split.py to split data for training and testing. 
I use the first (sorted in an ascending order by ID) 20% of images per class for testing and the remaining 80% for training, create two CSV files named test_data.csv and train_data.csv and report the number and the range of filenames for each class in training and testing sets.

2. According to the test_data.csv and train_data.csv, I use move.py to extract the data that will be exploited (Healthy and TB) to a new folder "dataset" (due to the memory, not loaded on the github).

3. Based on the dataloader.py, ResNet50.py and ResNet50dropout.py, I do 5 experiments - train_1.py, train_contrast.py, train_dropout.py, train_regu.py, train_sharp.py. After training, the parameters of models in every epoch are saved to the subfolders of the folder "model" (due to the memory, not loaded on the github). Also, acc_test.txt, acc_train.txt, f_1_test.txt, loss_test.txt, loss_train.txt are saved to the subfolders in the folder "results" to visualize and compare the results later.

4. Use test.py to get test accuracy, f1 score and confusion matrix on specific epoch with epochXX.pt in subfolders of the folder "model".

5. Use visualization.py to get the process of the training/validation accuracy/loss, saved in the folder "visualize".

6. Use flops.py to get the FLOPS of the network.


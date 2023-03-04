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

## Data augmentation strategies and network architecture

I select **ResNet50** as the main network architecture. 

ResNet50 is a deep convolutional neural network architecture that was first proposed in 2015. It consists of 50 layers and is capable of achieving high accuracy in image classification tasks, including the classification of X-ray images. The network achieves this by analyzing patterns and features within the images, using a series of convolutional layers that extract increasingly complex information from the input data.

One of the key advantages of ResNet50 is its use of skip connections, which allow information to bypass certain layers and be passed directly to later layers. This helps to mitigate the problem of vanishing gradients, which can occur when gradients become too small to propagate through the network during training. By enabling the network to retain more information throughout the training process, skip connections can help to improve the accuracy and efficiency of ResNet50 for X-ray classification and other image recognition tasks.

For data augmentation strategies, I use **ColorJitter(*contrast = 0.2*)** and **RandomAdjustSharpness(*sharpness_factor=1.2*, *p=0.2*).**

**ColorJitter(*brightness = 0(default)*, *contrast = 0(default)*, *saturation = 0(default)*, *hue = 0(default)*)** can randomly change the brightness, contrast, saturation and hue of an image. X-ray images often have low contrast, making it difficult to distinguish between different anatomical structures or identify certain pathologies. For the task of X-ray image classification, the property of contrast is the core feature which can make the model learn to be more robust to variations and better able to identify relevant features in the images, so I only change the parameters of contrast and keep others zero.

**RandomAdjustSharpness(*sharpness_factor*, *p=0.5(default)*)** can adjust the sharpness of the image randomly with a given probability. X-ray images can often be of low quality, with poor sharpness. By using RandomAdjustSharpness() during training, the robustness of the model  to these variations can be improved and the classification will be more effective with X-ray images that may have different levels of sharpness. 

Both data augmentation methods above can help to prevent overfitting, which occurs when a model becomes too specialized to the training data and does not perform well on new, unseen data. By introducing variations in the training data, the model is forced to learn more generalizable features, which can help it to perform better on new, unseen X-ray images.

## Details of the 5 experiments

**experiment 1 baseline**
```python
lr = 0.0001
epochs=20
batch_size = 4
model = resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
```
**experiment 2  L2 regularization penalty**
```python
lr = 0.0001
epochs=20
batch_size = 4
model = resnet50()
**criterion = RegularizedCrossEntropyLoss(lambda_reg=0.001)**
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
```
**experiment 3  dropout**
```python
lr = 0.0001
epochs=20
batch_size = 4
**model = resnet50_dropout()**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
```

**experiment 4 data augmentation 1  sharpness**
```python
lr = 0.0001
epochs=20
batch_size = 4
model = resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
transform = transforms.Compose([**transforms.RandomAdjustSharpness(sharpness_factor = 1.2, p = 0.2),**
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
```
**experiment 5  data augmentation 2 contrast**
```python
lr = 0.0001
epochs=20
batch_size = 4
model = resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
transform = transforms.Compose([**transforms.ColorJitter(contrast=0.2)**,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
```

**Reasons :**
The learning rate determines how much the model's parameters are updated during training. If the learning rate is too high, the model may diverge and fail to converge to a good solution. If the learning rate is too low, the model may take a long time to converge. Considering the model complexity and the dataset scale, I set **lr = 0.0001.**

The number of epochs determines how many times the model sees each example in the training data. If the number of epochs is too low, the model may underfit and fail to learn the complex patterns in the data. If the number of epochs is too high, the model may overfit and memorize the training data without generalizing well to new data. The number of epochs should be chosen based on the complexity of the problem and the size of the dataset. For this problem setting, I set **epochs = 20** for sure that the results will converge. To avoid overfitting, I will select the epoch with best performance in evaluation, not only the last epoch.

The batch size determines how many examples are processed in each update step of the model during training. If the batch size is too small, the updates may be noisy and slow. If the batch size is too large, the model may not be able to fit in memory or may not generalize well to new data. The batch size should be chosen based on the available memory and the size of the dataset. To get a balance with the convergence speed and accuracy, I set **batch_size = 4.**

For the selection of network, I have mentioned in the introduction of ResNet50.

In experiment 3, I add **dropout** with a probability of **p=0.5** to the fully connected layer of resnet50. Dropout is a commonly used regularization technique that can help prevent overfitting in deep neural networks and improve the generalization performance of the model.

For the criterion, I try **CrossEntropyLoss** and **RegularizedCrossEntropyLoss with $\lambda$=0.001.**

**CrossEntropyLoss** is useful when training a classification problem with C classes. This is particularly useful when we have an unbalanced training set, as our x-ray training set (3040 healthy & 640 TB).

**RegularizedCrossEntropyLoss** adds a penalty term to the loss function during training, which encourages the model to learn smaller weight values and help prevent overfitting and improve generalization performance. I set  **$\lambda$=0.001** to balance the trade-off between fitting the training data well and avoiding overfitting.

**The Adam optimizer** is a popular choice for training ResNet50 models for image classification tasks due to its adaptive learning rates, momentum, regularization, efficient memory usage, and ease of use. It can help to improve the convergence speed and accuracy of the optimizer, prevent overfitting, and save time and resources during training.

For the selection of data augmentation strategies, I have mentioned in the introduction above.

## Results and Analysis

(epoch 0 - 19, for test convenience when using epoch.pt)

at epoch with highest f1 score

| experiment | epoch | test accuracy | f1 score | confusion matrix |
| --- | --- | --- | --- | --- |
| experiment 1 baseline | 18 | 98.9130 | 0.9686 | [[756 4][ 6 154]] |
| experiment 2 regu_penalty | 16 | 99.2391 | 0.9781 | [[757 3][ 4 156]] |
| experiment 3 dropout | 17 | 99.2391 | 0.9783 | [[755 5][ 2 158]] |
| experiment 4 sharpness | 12 | 99.0217 | 0.9718 | [[756 4][ 5 155]] |
| experiment 5 contrast | 19 | 99.1304 | 0.9750 | [[756 4][ 4 156]] |

at epoch 19 : 

| experiment | test accuracy | f1 score | confusion matrix |
| --- | --- | --- | --- |
| experiment 1 baseline | 97.8261 | 0.9390 | [[746 14][ 6 154]] |
| experiment 2 regu_penalty | 98.8043 | 0.9653 | [[756 4][ 7 153]] |
| experiment 3 dropout | 99.1304 | 0.9755 | [[753 7][ 1 159]] |
| experiment 4 sharpness | 97.8261 | 0.9408 | [[741 19][ 1 159]] |
| experiment 5 contrast | 99.1304 | 0.9750 | [[756 4][ 4 156]] |

**Analysis :**

The results of experiment 2-5 prove the effectiveness of **RegularizedCrossEntropyLoss, dropout, sharpness adjustment** and **contrast adjustment** respectively. Balanced with the performance at the last epoch and the epoch with highest f1 score, I select **experiment 3 dropout and experiment 5 contrast as top 2 experiments.**

## Top 2 experiments Visualization

(Rest figures can be found in folder "visualize")

![train_acc.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/799a3416-a2a3-4ae9-8f17-ce244cddece8/train_acc.png)

## Parameters and FLOPS
| experiment | model | number of parameters | GFLOPs |
| --- | --- | --- | --- |
| experiment 1 baseline  | ResNet50 | 23512130 | 3.88433G |
| experiment 2 regu_penalty | ResNet50 | 23512130 | 3.88433G |
| experiment 3 dropout | ResNet50 with dropout | 23512130 | 3.88433G |
| experiment 4 sharpness | ResNet50 | 23512130 | 3.88433G |
| experiment 5 contrast | ResNet50 | 23512130 | 3.88433G |

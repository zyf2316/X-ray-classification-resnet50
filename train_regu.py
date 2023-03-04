import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from dataloader import XRayDataset
from tqdm import tqdm
from ResNet50 import resnet50
from sklearn.metrics import f1_score
import time
import numpy as np


## Load dataset
train_csv_file = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/train_data.csv'
test_csv_file = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/test_data.csv'
train_dir = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/dataset/train'
test_dir = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/dataset/test'
save_path = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/model/experiment_regu'
results_path = "/home/yufei.zhang/Documents/HC/assignment_2/task_2/results/experiment_regu"


lr = 0.0001
epochs=20
batch_size = 4

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Create the training and testing datasets
train_dataset = XRayDataset(train_csv_file, train_dir, transform=transform)
test_dataset = XRayDataset(test_csv_file, test_dir, transform=transform)

# Create the dataloaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


## device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## define model
model = resnet50()
print(model)
pytorch_total_params = sum(p.numel() for p in  model.parameters())
print('Number of parameters: {0}'.format(pytorch_total_params))

## Creating training loop
def train(model, acc_train, loss_train):
    model.train()
    train_loss = 0
    total = 0
    correct=0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            # send to device
            data = data.to(device)
            target = target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target, model)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        print(' train loss: {:.4f} accuracy: {:.4f}'.format(train_loss/(batch_idx+1), 100.*correct/total))
        acc_train = acc_train.append( 100.*correct/total)
        loss_train = loss_train.append(train_loss/(batch_idx+1))
        #wandb.log({"train_loss": train_loss/(batch_idx+1),"train_accuracy":100.*correct/total})      
        torch.save(model.state_dict(), os.path.join(save_path,f"epoch{epoch}.pt"))
## Creating training loop
def validate(model, acc_test, loss_test,  f_1_score):   
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    pred = []
    true = []
    with tqdm(test_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            # send to device
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target, model)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            target_cpu = target.cpu() # GPU tensor to CPU tensor
            target_np = target_cpu.detach().numpy()
            target = list(target_np)

            predicted_cpu = predicted.cpu() # GPU tensor to CPU tensor
            predicted_np = predicted_cpu.detach().numpy()
            predicted = list(predicted_np)

            pred.append(predicted)
            true.append(target)
        # wandb.log({"test_accuracy": 100.*correct / total,"test_loss":test_loss/(batch_idx+1)})
        pred = np.array(pred).flatten()
        true = np.array(true).flatten()
        f_1 = f1_score(true, pred)
        print(' test loss: {:.4f} accuracy: {:.4f} f1 score: {:.4f}'.format(test_loss/(batch_idx+1), 100.*correct/total, f_1))
        acc_test.append(100.*correct/total)
        loss_test.append(test_loss/(batch_idx+1))
        f_1_score.append(f_1)
        # print(' test loss: {:.4f}'.format(test_loss/(batch_idx+1)))




## Starting training
class RegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, lambda_reg):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, outputs, targets, model):
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.sum(param.pow(2))
        loss = ce_loss + 0.5 * self.lambda_reg * reg_loss
        return loss 
criterion = RegularizedCrossEntropyLoss(lambda_reg=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

## move the model to the device
model.to(device)
next(model.parameters()).is_cuda
start = time.time()
acc_test = []
loss_test = []
acc_train = []
loss_train = []
f_1_score = []
for epoch in range(0, epochs):
    print("epoch number: {0}".format(epoch))
    train(model, acc_train, loss_train)
    validate(model, acc_test, loss_test,  f_1_score)
end = time.time()
Total_time=end-start
print('Total training and inference time is: {0}'.format(Total_time))

acc_test = np.array(acc_test)
np.savetxt(os.path.join(results_path,"acc_test.txt"), acc_test, delimiter=' ',fmt='%.4f') # save to file
loss_test = np.array(loss_test)
np.savetxt(os.path.join(results_path,"loss_test.txt"), loss_test, delimiter=' ',fmt='%.4f') 
acc_train = np.array(acc_train)
np.savetxt(os.path.join(results_path,"acc_train.txt"), acc_train, delimiter=' ',fmt='%.4f') # save to file
loss_train = np.array(loss_train)
np.savetxt(os.path.join(results_path,"loss_train.txt"), loss_test, delimiter=' ',fmt='%.4f') 
f_1_score = np.array(f_1_score)
np.savetxt(os.path.join(results_path,"f_1_test.txt"), f_1_score, delimiter=' ',fmt='%.4f') 


# with open('/home/yufei.zhang/Documents/HC/assignment_2/task_2/results/experiment_1/acc_test.txt', 'w') as f:
#     for item in acc_test:
#         f.write("%s\n" % item)


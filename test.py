import numpy as np
import torch
import os
from dataloader import XRayDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from ResNet50 import resnet50
from sklearn.metrics import f1_score, confusion_matrix

load_model_path = "/home/yufei.zhang/Documents/HC/assignment_2/task_2/model/experiment_contrast"
load_model_name = "epoch19.pt"
test_csv_file = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/test_data.csv'
test_dir = '/home/yufei.zhang/Documents/HC/assignment_2/task_2/dataset/test'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

batch_size = 1
test_dataset = XRayDataset(test_csv_file, test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = resnet50()	
model.load_state_dict(torch.load(os.path.join(load_model_path,load_model_name)))	
   
model.eval()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
total = 0
correct=0
pred = []
true = []
with tqdm(test_loader, unit="batch") as tepoch:
    for batch_idx, (data, target) in enumerate(tepoch):
        # send to device
        #data, target = data.to(device), target.to(device) 
        output = model(data)
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

    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    acc = 100.*correct/total
    f_1 = f1_score(true, pred)
    c_m = confusion_matrix(true, pred)
print("test accuracy is {}, \nf1 score is {}, \n confusion matrix is {}".format(acc, f_1, c_m))
        
            
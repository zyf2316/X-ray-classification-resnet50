import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class XRayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')     
        label = self.data.iloc[idx, 1]
        if label == "Healthy":
            label = 0 # healthy
        else:
            label = 1 # TB
        label = torch.tensor(label) 
        if self.transform:
            image = self.transform(image)
        # Convert the image to a numpy array
        image_array = np.asarray(image) 
        img = torch.tensor(image_array, dtype=torch.float32) 
    
        return img, label


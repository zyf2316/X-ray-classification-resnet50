#!pip install flopth
import torch
from ResNet50dropout import resnet50_dropout
from flopth import flopth

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet50_dropout()
dummy_inputs = torch.rand(1, 3, 224, 224)
model.train()
dummy_inputs.to(device)
flops, params = flopth(model, inputs=(dummy_inputs,))
print(flops)
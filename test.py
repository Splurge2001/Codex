import torch
from model.cnn import sampleCNN

x = torch.randn(32, 1, 224, 224)
model = sampleCNN(num_classes=3)
output = model(x)
print(output.size())
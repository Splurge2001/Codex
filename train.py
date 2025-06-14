import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset.pointcloud_dataset import PointCloudDataset
from tqdm import tqdm
import os
from model.cnn import sampleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = PointCloudDataset("dataset", transform=train_transformer)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
trainset, testset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)

def train(model, train_loader, test_loader, criterion, optimizer, epoch_num, save_path):
    best_acc = 0.0
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"epoch:{epoch+1}/{epoch_num}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"epoch[{epoch+1}/{epoch_num}], train_loss: {epoch_loss:.4f}")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print("model is saved with best acc", best_acc)

def evaluate(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(loader.dataset)
    accuracy = 100.0 * correct / total
    print(f"test_loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")
    return accuracy

def save_model(model,save_path):
    torch.save(model.state_dict(),save_path)

if __name__ =="__main__":
    epoch_num = 10
    learning_rate = 0.001
    num_class = 3
    save_path = r"model_path\best.pth"
    model = sampleCNN(num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, test_loader, criterion, optimizer, epoch_num, save_path)
    evaluate(model, test_loader, criterion)

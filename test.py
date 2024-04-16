import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Path adjustments
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_path, 'data'))
sys.path.append(os.path.join(dir_path, 'model'))

from MLP import MLP
from dataloader import create_dataloaders


def test_model(model, test_loader, criterion = nn.MSELoss()):
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# file_path
dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(dir_path, 'dataset')

train_data = (os.path.join(dataset_dir, 'train_inputs_10000.txt'), os.path.join(dataset_dir, 'train_outputs_10000.txt'))
val_data = (os.path.join(dataset_dir, 'validation_inputs_10000.txt'), os.path.join(dataset_dir, 'validation_outputs_10000.txt'))
test_data = (os.path.join(dataset_dir, 'test_inputs_10000.txt'), os.path.join(dataset_dir, 'test_outputs_10000.txt'))

# create data_loader
_, _, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size=64, scaler_type="minmax")

# load model
trained_model_dir = os.path.join(dir_path, 'trained_model')
model_path = os.path.join(trained_model_dir, 'model_complete.pth')
model = torch.load(model_path)
model.eval()

# test model
loss_list = [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()]
name_list = ['MSE', 'L1', 'Huber']
for i in range(len(loss_list)):
    loss = loss_list[i]
    name = name_list[i]
    test_loss = test_model(model, test_loader,criterion=loss)
    print(f"Average Test Loss of {name}: {test_loss}")

rmse_loss = 0.0
criterion = nn.MSELoss()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = torch.sqrt(criterion(outputs, targets))
        rmse_loss += loss.item()
rmse_loss / len(test_loader)
print(f"Average Test Loss of RMSE: {rmse_loss}")

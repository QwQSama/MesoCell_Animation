import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys
import os
from torch.optim.lr_scheduler import MultiStepLR

# Path adjustments
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_path, 'data'))
sys.path.append(os.path.join(dir_path, 'model'))


from dataloader import create_dataloaders
from MLP import MLP

# Load data
dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(dir_path, 'dataset')
trained_model_dir = os.path.join(dir_path, 'trained_model')

train_data = (os.path.join(dataset_dir, 'train_inputs_10000.txt'), os.path.join(dataset_dir, 'train_outputs_10000.txt'))
val_data = (os.path.join(dataset_dir, 'validation_inputs_10000.txt'), os.path.join(dataset_dir, 'validation_outputs_10000.txt'))
test_data = (os.path.join(dataset_dir, 'test_inputs_10000.txt'), os.path.join(dataset_dir, 'test_outputs_10000.txt'))

# Assuming the data loaders return DataLoader instances
scaler_path = os.path.join(trained_model_dir, 'scaler_X.joblib')
train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size=64, scaler_type="minmax", save_path=scaler_path)

# Model setup
input_features = 8  
output_features = 4  
num_hidden_layers = 2

model = MLP(input_features, num_hidden_layers, output_features)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001) 

scheduler = MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)  # 在第500和800个epoch时将学习率乘以0.1
num_epochs = 1000

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # if (epoch + 1) % 10 ==0:
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # Validation phase
    if (epoch + 1) % 10 ==0:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            print(f'Validation Loss: {val_loss / len(val_loader)}')

    scheduler.step()

model_path = os.path.join(trained_model_dir, 'model_complete.pth')
torch.save(model, model_path)
print('model saved!')
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
from joblib import dump, load

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs, scaler=None):
        if scaler:
            self.inputs = scaler.transform(inputs)
        else:
            self.inputs = inputs
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def load_data(file_path):
    return np.loadtxt(file_path, dtype=np.float32)

def create_scaler(data, scaler_type="minmax", save_path = None):
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type")
    scaler.fit(data)

    if save_path is not None:
        dump(scaler, save_path)

    return scaler

def create_dataloaders(train_data, val_data, test_data, batch_size=32, scaler_type="minmax", save_path = None):
    # Load the actual data from the disk
    train_inputs, train_outputs = load_data(train_data[0]), load_data(train_data[1])
    val_inputs, val_outputs = load_data(val_data[0]), load_data(val_data[1])
    test_inputs, test_outputs = load_data(test_data[0]), load_data(test_data[1])

    # Create a scaler based on the training data
    scaler = create_scaler(train_inputs, scaler_type=scaler_type, save_path=save_path)

    # Create dataset instances
    train_dataset = CustomDataset(train_inputs, train_outputs, scaler=scaler)
    val_dataset = CustomDataset(val_inputs, val_outputs, scaler=scaler)
    test_dataset = CustomDataset(test_inputs, test_outputs, scaler=scaler)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# # test:
# dir_path = os.path.dirname(os.path.abspath(__file__))
# dataset_dir = os.path.join(dir_path, '..', 'dataset')
# train_data = (os.path.join(dataset_dir, 'train_inputs_10000.txt'), os.path.join(dataset_dir, 'train_outputs_10000.txt'))
# val_data = (os.path.join(dataset_dir, 'validation_inputs_10000.txt'), os.path.join(dataset_dir, 'validation_outputs_10000.txt'))
# test_data = (os.path.join(dataset_dir, 'test_inputs_10000.txt'), os.path.join(dataset_dir, 'test_outputs.txt'))

# train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size=64, scaler_type="minmax")

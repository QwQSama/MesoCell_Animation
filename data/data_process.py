import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

def load_and_filter_data(input_file, output_file, lower_thresholds, upper_thresholds, max_samples=None):
    # Load data from text files
    inputs = np.loadtxt(input_file).T
    outputs = np.loadtxt(output_file).T
    
    # Apply filtering based on thresholds
    valid_indices = np.all((outputs >= lower_thresholds) & (outputs <= upper_thresholds), axis=1)
    filtered_inputs = inputs[valid_indices]
    filtered_outputs = outputs[valid_indices]

    filtered_outputs = filtered_outputs[:, :4]
    
    # If max_samples is specified and less than the number of available samples, randomly select samples
    if max_samples is not None and max_samples < len(filtered_inputs):
        indices = np.random.choice(len(filtered_inputs), max_samples, replace=False)
        filtered_inputs = filtered_inputs[indices]
        filtered_outputs = filtered_outputs[indices]

    scaler = MinMaxScaler()
    normalized_outputs = scaler.fit_transform(filtered_outputs)
    
    return filtered_inputs,  normalized_outputs, scaler

def split_data(inputs, outputs, train_size=0.8, val_size=0.1, test_size=0.1):
    total_size = len(inputs)
    train_count = int(total_size * train_size)
    val_count = int(total_size * val_size)
    test_count = total_size - train_count - val_count 

    # testset
    inputs_train_val, inputs_test, outputs_train_val, outputs_test = train_test_split(
        inputs, outputs, test_size=test_count, random_state=42)

    # trainset + valset
    inputs_train, inputs_val, outputs_train, outputs_val = train_test_split(
        inputs_train_val, outputs_train_val, test_size=val_count, random_state=42)

    return inputs_train, outputs_train, inputs_val, outputs_val, inputs_test, outputs_test

# Define thresholds
lower_thresholds = np.array([0, 0, -3e7, np.pi/6, 2, 2])
upper_thresholds = np.array([1.2e8, 0.8e8, 3e7, np.pi/2, 6, 4])

# Get the absolute directory path of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(dir_path, '..', 'dataset')

# Specify file paths
input_file = os.path.join(dir_path, '..', 'dataset', 'raw_input.txt')
output_file = os.path.join(dir_path, '..', 'dataset', 'raw_output.txt')

# Process data for 10k and 100k samples
sizes = [10000, 100000]
for size in sizes:
    inputs, outputs, scaler = load_and_filter_data(input_file, output_file, lower_thresholds, upper_thresholds, max_samples=size)
    inputs_train, outputs_train, inputs_val, outputs_val, inputs_test, outputs_test = split_data(inputs, outputs)

    # Save the split data to files in the dataset folder
    np.savetxt(os.path.join(dataset_dir, f'train_inputs_{size}.txt'), inputs_train)
    np.savetxt(os.path.join(dataset_dir, f'train_outputs_{size}.txt'), outputs_train)
    np.savetxt(os.path.join(dataset_dir, f'validation_inputs_{size}.txt'), inputs_val)
    np.savetxt(os.path.join(dataset_dir, f'validation_outputs_{size}.txt'), outputs_val)
    np.savetxt(os.path.join(dataset_dir, f'test_inputs_{size}.txt'), inputs_test)
    np.savetxt(os.path.join(dataset_dir, f'test_outputs_{size}.txt'), outputs_test)
    dump(scaler, os.path.join(dataset_dir, f'scaler_{size}.joblib'))

    print('Dataset saved for size:', size)

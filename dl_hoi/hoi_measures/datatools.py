import pickle
import os
import torch
import seaborn as sns


def load_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def load_params_data_dir(data_dir):
    data = []
    for data_file in os.listdir(data_dir):
        data.append(load_data(os.path.join(data_dir, data_file))["params"])
    return data

def data_dir_to_tensor(dir_path, params_to_load="all"): # options: "all", "first_layer_weights", "first_layer_biases", "second_layer_weights"
    data = load_params_data_dir(dir_path)
    samplerange = range(len(data))
    timerange = range(len(data[0]))
    if params_to_load == "all":
        tensor_data = torch.stack([torch.vstack([torch.concat((data[i][t][0].flatten(), data[i][t][1], data[i][t][2].flatten())) for i in samplerange]) for t in timerange])
    elif params_to_load == "first_layer_weights":
        tensor_data = torch.stack([torch.vstack([data[i][t][0].flatten() for i in samplerange]) for t in timerange])
    elif params_to_load == "first_layer_biases":
        tensor_data = torch.stack([torch.vstack([data[i][t][1] for i in samplerange]) for t in timerange])
    elif params_to_load == "second_layer_weights":
        tensor_data = torch.stack([torch.vstack([data[i][t][2].flatten() for i in samplerange]) for t in timerange])
    else:
        raise ValueError("Invalid value for params_to_load. Accepted values: 'all', 'first_layer_weights', 'first_layer_biases', 'second_layer_weights'")
    return tensor_data

def plot_correlogram(datatensor, id):
    datatensor_np = datatensor[id, :].numpy()
    return 0
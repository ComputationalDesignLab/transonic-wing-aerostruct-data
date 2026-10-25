# importing packages
import torch
from scipy.io import loadmat, savemat
import numpy as np
from models import PODNIROM
from scimlstudio.utils import Standardize, Normalize
import time
from utils import r2_calculation, nrmse_calculation
from utils import set_seed
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seed for random numbers")
parser.add_argument("--test_mode", help="mode of testing")
arguments = parser.parse_args()

set_seed(int(arguments.seed))
mode = arguments.test_mode

# define function for building a network
def build_network(input_dim, hidden_dims, output_dim, activation):
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation)
    for i in range(1, len(hidden_dims)):
        layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        layers.append(activation)
    layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))    
    network = torch.nn.Sequential(*layers)
    return network

# defining arguments for torch
args = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}

# loading the data
data = loadmat("./data_preprocessing/training_data_struct.mat")
x_train = torch.tensor(data["x"], **args)
y_train = torch.tensor(data["fail"], **args)
x_data = x_train
vector_data = y_train

# loading the validation data
val_data = loadmat("./data_preprocessing/validation_data_struct.mat")
x_val = torch.tensor(val_data["x"], **args)
y_val = torch.tensor(val_data["fail"], **args)

# defining the networks
latent_dim = 64
latent_nn = build_network(input_dim=x_data.shape[-1], hidden_dims=[1024, 1024, 1024], output_dim=latent_dim, 
                          activation=torch.nn.SiLU()).to(**args)

# creating the model
model = PODNIROM(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
            ric=0.9999,latent_nn=latent_nn,
            input_transform=Standardize(x_data), vector_output_transform=Standardize(vector_data),latent_output_transform=Standardize)

# training the model
ti = time.time()
model.fit_POD()
optimizer = torch.optim.Adam(model.parameters, lr=1e-3, weight_decay=1e-8)
loss_func = torch.nn.MSELoss() 
model.fit(seed = int(arguments.seed), optimizer=optimizer, loss_func=loss_func, epochs=3000)
tf = time.time()
print("Time for training:", tf-ti)

# training latent vector
os.makedirs("./results/pod/", exist_ok=True)
train_latents = model.POD_model.encoding(y_train.T)
save_tl = {"latent_vectors": train_latents.detach().cpu().numpy()}
savemat(f"./results/pod/pod_nirom_training_latent_{int(arguments.seed)}.mat", save_tl)

# importing the testing data
if mode == "testing":
    val_data = loadmat("./data_preprocessing/testing_data_struct.mat")
    x_test = torch.tensor(val_data["x"], **args)
    y_test = torch.tensor(val_data["fail"], **args)
elif mode == "ood":
    test_data = loadmat("./data_preprocessing/ood_samples_struct.mat")
    x_test = torch.tensor(test_data["x"], **args)
    y_test = torch.tensor(test_data["failure"], **args)

# testing latent vector
best_state_dict = torch.load(f"./best_models/pod_best_{int(arguments.seed)}.pth", weights_only=True)
model.load_state_dict(best_state_dict)
test_latents = model.POD_model.encoding(y_test.T)
save_testl = {"latent_vectors": test_latents.detach().cpu().numpy()}
savemat(f"./results/pod/pod_nirom_{mode}_latent_{int(arguments.seed)}.mat", save_testl)

# generating the prediction on the dataset
ti = time.time()
vector_predictions = model.predict(x_test)
tf = time.time()
print("Time for inference on testing samples:", tf-ti)

# calculating the error metrics
nrmse_fail = nrmse_calculation(true_values=y_test, predicted_values=vector_predictions)
print("NRMSE for failure:", torch.mean(nrmse_fail).item())

fail_r2 = r2_calculation(true_values=y_test, predicted_values=vector_predictions, dim=1)
print("R2 for failure:", torch.mean(fail_r2).item())

# storing the results
results = {"fail_predictions": vector_predictions.numpy(force=True), "nrmse_fail": nrmse_fail.numpy(force=True),
        "fail_r2": fail_r2.numpy(force=True)}

if mode == "testing":
    savemat(f"./results/pod/struct_results_pod_nirom_{int(arguments.seed)}.mat", results)
elif mode == "ood":
    savemat(f"./results/pod/ood_struct_results_pod_nirom_{int(arguments.seed)}.mat", results)


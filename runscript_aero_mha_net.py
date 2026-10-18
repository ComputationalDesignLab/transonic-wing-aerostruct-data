# importing packages
import torch
from scipy.io import loadmat, savemat
import numpy as np
from models import MHANet
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

# loading the training data
data = loadmat("training_data_aero.mat")
x_train = torch.tensor(data["x"], **args)
h_train = torch.tensor(np.hstack([data["cp"], data["cf"]]), **args).flatten(1)
y_train = torch.tensor(np.hstack([data["cl"], data["cd"], data["cm"]]), **args)
x_data = x_train
vector_data = h_train
coeff_data = y_train

# loading the validation data
val_data = loadmat("validation_data_aero.mat")
x_val = torch.tensor(val_data["x"], **args)
h_val = torch.tensor(np.hstack([val_data["cp"], val_data["cf"]]), **args).flatten(1)
y_val = torch.tensor(np.hstack([val_data["cl"], val_data["cd"], val_data["cm"]]), **args)

# defining the MHA networks
latent_dim = 128
encoder = build_network(input_dim=vector_data.shape[-1], hidden_dims=[1024, 512, 256], output_dim=latent_dim, activation=torch.nn.SiLU()).to(**args)
decoder = build_network(input_dim=latent_dim, hidden_dims=[256, 512, 1024], output_dim=vector_data.shape[-1], activation=torch.nn.SiLU()).to(**args)
latent_nn = build_network(input_dim=x_data.shape[-1], hidden_dims=[64, 64, 64], output_dim=latent_dim, activation=torch.nn.SiLU()).to(**args)
coeff_nn = build_network(input_dim=latent_dim, hidden_dims=[64, 64, 64], output_dim=coeff_data.shape[-1], activation=torch.nn.SiLU()).to(**args)

# creating the MHA model
model = MHANet(x_train=x_train,y_train=y_train,h_train=h_train, x_val=x_val,y_val=y_val,h_val=h_val,
            encoder=encoder,decoder=decoder,latent_nn=latent_nn,
            coeff_nn=coeff_nn,input_transform=Standardize(x_train),scalar_output_transform=Standardize(y_train), 
            vector_output_transform=Standardize(h_train))

# training the model
optimizer = torch.optim.Adam(model.parameters, lr=1e-3, weight_decay=1e-8)
loss_func = torch.nn.MSELoss() 
ti = time.time()
model.fit(seed = int(arguments.seed), optimizer=optimizer, loss_func=loss_func, epochs=5000)
tf = time.time()
print("Time for training:", tf-ti)

# training latent vector
os.makedirs("./results/mha/", exist_ok=True)
train_latents = model.encoder(h_train)
save_tl = {"latent_vectors": train_latents.detach().cpu().numpy()}
savemat(f"./results/mha/mha_net_training_latent_{int(arguments.seed)}.mat", save_tl)

# importing the testing data
if mode == "testing":
    test_data = loadmat("testing_data_aero.mat")
    x_test = torch.tensor(test_data["x"], **args)
    h_test = torch.tensor(np.hstack([test_data["cp"], test_data["cf"]]), **args).flatten(1)
    y_test = torch.tensor(np.hstack([test_data["cl"], test_data["cd"], test_data["cm"]]), **args)
elif mode == "ood":
    test_data = loadmat("ood_samples_aero.mat")
    x_test = torch.tensor(test_data["x"], **args)
    h_test = torch.tensor(np.hstack([test_data["CoefPressure"], test_data["SkinFrictionMagnitude"]]), **args).flatten(1)
    test_scalar_data = test_data["y"]
    y_test = torch.tensor(np.hstack([test_scalar_data[:,0].reshape(-1,1), test_scalar_data[:,3].reshape(-1,1), test_scalar_data[:,7].reshape(-1,1)]), **args)

# testing latent vector
best_state_dict = torch.load(f"./best_models/mha_net_best_{int(arguments.seed)}.pth", weights_only=True)
model.load_state_dict(best_state_dict)
test_latents = model.encoder(h_test)
save_testl = {"latent_vectors": test_latents.detach().cpu().numpy()}
savemat(f"./results/mha/mha_net_{mode}_latent_{int(arguments.seed)}.mat", save_testl)

# generating the prediction on the dataset
ti = time.time()
scalar_predictions, vector_predictions = model.predict(x_test)
tf = time.time()
print("Time for inference on testing samples:", tf-ti)
n_points = data["cp"].shape[1]
cp_predictions = vector_predictions[:,:n_points]
cf_predictions = vector_predictions[:,n_points:]

# calculating the error metrics
nrmse_cp = nrmse_calculation(true_values=h_test[:,:n_points], predicted_values=cp_predictions)
print("NRMSE for CP:", torch.mean(nrmse_cp).item())

nrmse_cf = nrmse_calculation(true_values=h_test[:,n_points:], predicted_values=cf_predictions)
print("NRMSE for CF:", torch.mean(nrmse_cf).item())

cp_r2 = r2_calculation(true_values=h_test[:,:n_points], predicted_values=cp_predictions, dim=1)
print("R2 for CP:", torch.mean(cp_r2).item())

cf_r2 = r2_calculation(true_values=h_test[:,n_points:], predicted_values=cf_predictions, dim=1)
print("R2 for CF:", torch.mean(cf_r2).item())

nrmse_coeff = nrmse_calculation(true_values=y_test, predicted_values=scalar_predictions)
print("NRMSE for coefficients:", nrmse_coeff)

coeff_r2 = r2_calculation(true_values=y_test, predicted_values=scalar_predictions, dim=0)
print("R2 for coeff:", coeff_r2)

# storing the results
results = {"cp_predictions": cp_predictions.numpy(force=True), "cf_predictions": cf_predictions.numpy(force=True), 
            "cp_nrmse": nrmse_cp.numpy(force=True), "cf_nrmse": nrmse_cf.numpy(force=True), "coeff_nrmse": nrmse_coeff.numpy(force=True),
            "cp_r2": cp_r2.numpy(force=True), "cf_r2": cf_r2.numpy(force=True), "coeff_r2": coeff_r2.numpy(force=True)}

if mode == "testing":
    savemat(f"./results/mha/mha_results_{int(arguments.seed)}.mat", results)
elif mode == "ood":
    savemat(f"./results/mha/ood_mha_results_{int(arguments.seed)}.mat", results)





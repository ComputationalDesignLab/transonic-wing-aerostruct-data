import os
import h5py
import numpy as np
from scipy.io import loadmat, savemat

# looping through all the files
num_samples = 3
dtype = "ood_samples"
cp_list = []
cf_list = []
mesh_points = []
valid_indices = []
for i in range(num_samples):
    path = f"./{dtype}/{i+1}/surface_outputs.hdf5"
    if not os.path.exists(path):
        continue
    print(i+1)
    with h5py.File(path, "r") as f:
        cp_list.append(f["CoefPressure"][()].reshape(-1,1))
        cf_list.append(f["SkinFrictionMagnitude"][()].reshape(-1,1))
        mesh_points.append(f["MeshPoints"][()])
    valid_indices.append(i)

cp_array = np.stack(cp_list, axis=0)
print(cp_array.shape)
cf_array = np.stack(cf_list, axis=0)
mesh_array = np.stack(mesh_points, axis=0)

# loading the scalar data
scalar_data = loadmat(f"./{dtype}/data.mat")
x_data = scalar_data["x"]
target_keys = ['cd', 'cdp', 'cdv', 'cl', 'clp', 'clv', 'cmx', 'cmy', 'cmz']
y_data = np.hstack([scalar_data[k].reshape(-1,1) for k in target_keys if k in scalar_data])

# filter to only valid indices
x_data = x_data[valid_indices]
y_data = y_data[valid_indices]

print(x_data.shape)
print(y_data.shape)

# saving the data
data = {"CoefPressure": cp_array, "SkinFrictionMagnitude": cf_array, "MeshPoints": mesh_array, "x": x_data, "y": y_data, "valid_indices": valid_indices}
savemat(f"./data_preprocessing/ood_samples_aero.mat", data)


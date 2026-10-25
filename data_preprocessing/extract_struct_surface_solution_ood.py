import os
import pyvista as pv
import numpy as np
from scipy.io import loadmat, savemat

# looping through all the files
num_samples = 3
dtype = "ood_samples"
failure_list = []
mesh_points = []
valid_indices = []
for i in range(num_samples):
    path = f"./{dtype}/{i+1}/structural_solution.vtk"
    if not os.path.exists(path):
        continue
    print(i+1)
    mesh = pv.read(path)
    mesh_points.append(mesh.points)
    failure_list.append(mesh.point_data["failure"])
    
    valid_indices.append(i)

fail_array = np.stack(failure_list, axis=0)
mesh_array = np.stack(mesh_points, axis=0)

# loading the scalar data
scalar_data = loadmat(f"./{dtype}/data.mat")
x_data = scalar_data["x"]
target_keys = ['cd', 'cdp', 'cdv', 'cl', 'clp', 'clv', 'cmx', 'cmy', 'cmz']
y_data = np.hstack([scalar_data[k].reshape(-1,1) for k in target_keys if k in scalar_data])

# filter to only valid indices
x_data = x_data[valid_indices]
y_data = y_data[valid_indices]

# saving the data
data = {"failure": fail_array, "MeshPoints": mesh_array, "x": x_data, "y": y_data, "valid_indices":valid_indices}
savemat(f"./data_preprocessing/ood_samples_struct.mat", data)


from scipy.io import loadmat, savemat
import numpy as np

# loading the data
data = loadmat("full_data_aero.mat")
x_data = data["x"]
cp_data = data["CoefPressure"]
cf_data = data["SkinFrictionMagnitude"]
scalar_data = data["y"]
valid_indices = data["valid_indices"][0]
cd_data = scalar_data[:,0].reshape(-1,1)
cl_data = scalar_data[:,3].reshape(-1,1)
cm_data = scalar_data[:,7].reshape(-1,1)
print(valid_indices)
# splitting the data
rng = np.random.default_rng(seed=1)
permutated_inds = rng.permutation(x_data.shape[0])
train_inds = permutated_inds[:int(0.80*x_data.shape[0])]
val_inds = permutated_inds[int(0.80*x_data.shape[0]):int(0.90*x_data.shape[0])]
test_inds = permutated_inds[int(0.90*x_data.shape[0]):]
valid_test_inds = valid_indices[test_inds]
print(train_inds, len(val_inds), len(test_inds))

# saving the split data
train = {"x": x_data[train_inds], "cp": cp_data[train_inds], "cf": cf_data[train_inds],
        "cl": cl_data[train_inds], "cd": cd_data[train_inds], "cm": cm_data[train_inds]}
savemat("./data_preprocessing/training_data_aero.mat", train)

val = {"x": x_data[val_inds], "cp": cp_data[val_inds], "cf": cf_data[val_inds],
        "cl": cl_data[val_inds], "cd": cd_data[val_inds], "cm": cm_data[val_inds]}
savemat("./data_preprocessing/validation_data_aero.mat", val)

test = {"x": x_data[test_inds], "cp": cp_data[test_inds], "cf": cf_data[test_inds],
        "cl": cl_data[test_inds], "cd": cd_data[test_inds], "cm": cm_data[test_inds], "indices": test_inds, "valid_indices": valid_test_inds}
savemat("./data_preprocessing/testing_data_aero.mat", test)



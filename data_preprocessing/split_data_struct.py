from scipy.io import loadmat, savemat
import numpy as np
import random

# loading the data
data = loadmat("full_data_struct.mat")
x_data = data["x"]
fail_data = data["failure"]
valid_indices = data["valid_indices"][0]
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
train = {"x": x_data[train_inds], "fail": fail_data[train_inds]}
savemat("./data_preprocessing/training_data_struct.mat", train)

val = {"x": x_data[val_inds], "fail": fail_data[val_inds]}
savemat("./data_preprocessing/validation_data_struct.mat", val)

test = {"x": x_data[test_inds], "fail": fail_data[test_inds], "indices": test_inds, "valid_test_inds": valid_test_inds}
savemat("./data_preprocessing/testing_data_struct.mat", test)



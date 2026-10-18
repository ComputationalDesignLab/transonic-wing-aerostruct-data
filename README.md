# A High-Fidelity Aerostructural Dataset of Transonic Wings in Cruise Conditions

This repository accompanies the aerostructural dataset available at the following link (insert link) and provides implementations that can be used to train machine learning (ML) models using the dataset to create aerodynamic and structural mechanics surrogate models. This is to demonstrate the usage of the dataset in performing ML-driven static aerostructural analysis of a transonic wing geometry. The code in this repository is also provided to reproduce the results presented in the paper associated with the dataset. The dataset should be downloaded from the provided link before proceeding further with the code.

## Requirements

It is recommended to use a conda environment to run the code in this repository. This conda environment should have the following dependencies already installed before the code can be run in the environment. The versions in parentheses indicate those versions which were used to generate the results presented in the accompanying paper. 

* Python (3.12.11)
* PyTorch (2.11.0 preferably with CUDA support)
* Numpy (2.4.4)
* Scipy (1.17.1)
* h5py (3.16.0)
* PyVista (0.48.0) 
* [scimlstudio](https://github.com/ComputationalDesignLab/scimlstudio) (1.7.1)

Once the conda environment is set up with the correct dependencies, the given code can be run or modified by a user.

## Consolidating the data

Once the aerostructural dataset has been downloaded, the data from the individual folders must be consolidated for ease of training and evaluating the ML models. This can be done using scripts that are provided in this repository. To consolidate the data into `.mat` files that can be loaded to train and evaluate the ML models, the following commands can be used to run the relevant scripts.

```
python extract_aero_solution.py
python extract_struct_solution.py
```

Similarly, the out-of-distribution (OOD) data can be extracted from the sample files using the following scripts and commands. 

```
python extract_aero_solution_ood.py
python extract_struct_solution_ood.py
```

>NOTE: Before running the above scripts, it is necessary to change the `path` variable within these scripts to point to the directory where the dataset is stored.

Running these scripts should produce `full_data_aero.mat`, and `full_data_struct.mat` within this same directory. Once these `.mat` files are generated, the aerodynamics and structures data can be split randomly into training, validation and testing data. This can be done using the following commands. 

```
python split_data_aero.py
python split_data_struct.py
```

This should split the data into approximately 80% training, 10% validation and 10% testing data. Running these scripts should produce `training_data_aero.mat`, `training_data_struct.mat`, `testing_data_aero.mat`, `testing_data_struct.mat`, `validation_data_aero.mat`, and `validation_data_struct.mat`. After the data is split, the remaining scripts can be used to train and validate the models.    

## Training and evaluating the models

The runscripts provided in the repository both train and evaluate the model using the normalized root mean squared error (NRMSE) on a per-point basis and the coefficient of determination ($R^2$) on a per-sample or per-point basis. These are the same error metrics that are used in the paper and described in its appendix. The following command can be run to execute any one of the runscripts to train and evaluate a model. Here, it is important to remember to provide the seed for the initialization of the neural networks and the testing mode (testing or ood) as arguments in the command line when running the script. In the paper, 10 repetitions of the model training and evaluation were run with the seed varying from 1 to 10.  

```
python runscript_aero_mha_net.py --seed 1 --test_mode testing
```
The correspondence between the runscripts and the models that were described in the paper is given in the following table. Here, NIROM refers to non-intrusive reduced order model.

| Model | Runscript | 
| -------- | -------- |
| [MHA-Net](https://www.sciencedirect.com/science/article/pii/S1270963824000415) | `runscript_aero_mha_net.py` |
| [UQAeroNet](https://arc.aiaa.org/doi/abs/10.2514/6.2026-1208)| `runscript_aero_uq_aero_net.py` |
| Autoencoder NIROM |`runscript_struct_auto_nirom.py` | 
| POD NIROM |`runscript_struct_pod_nirom.py`| 

Running any of the runscripts will save a `.mat` file with the predictions of the model for various physical quantities and the calculated error metrics for those physical quantities based on the testing data. This `.mat` file can then be used for visualizations or other postprocessing.

## Results

The models achieve the following results when trained using the training and validation samples from the dataset and evaluated on the testing sample points. This is summarized from the results presented in the accompanying paper. These results represent the median performance across 10 different random initialization of the neural network parameters.

#### Aerodynamics 
| Model | NRMSE $C_p$ | $R^2$ $C_p$ | NRMSE $C_f$ | $R^2$ $C_f$ | NRMSE $C_D$ | $R^2$ $C_D$ | NRMSE $C_L$ | $R^2$ $C_L$ |
|---|---|---|---|---|---|---|---|---|
| MHA-Net   | $0.0213$ | $0.9984$ | $0.0307$ | $0.9967$ | $0.0107$ | $0.9975$ | $0.0087$ | $0.9975$ |
| UQAeroNet | $0.0212$ | $0.9984$ | $0.0309$ | $0.9968$ | $0.0109$ | $0.9974$ | $0.0084$ | $0.9977$ |

#### Structures
| Model | NRMSE for failure distribution | $R^2$ for failure distribution |
|---|---|---|
| Autoencoder | $0.1138$ | $0.834$ |
| POD | $0.1189$ | $0.76412$ |

## Citation
(ADD ARXIV CITATION)

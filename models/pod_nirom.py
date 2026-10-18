import torch
import torch.nn as nn
import os
from scimlstudio.utils import Standardize, Normalize
from scimlstudio.models import POD
import os

# implementation for nirom with autoencoder and neural network
args = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}
class PODNIROM(nn.Module):

    def __init__(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_val: torch.Tensor,
            y_val: torch.Tensor,
            ric: float,
            latent_nn: nn.Sequential,
            input_transform: Standardize | Normalize | None = None,
            vector_output_transform: Standardize | Normalize | None = None,
            latent_output_transform: Standardize | Normalize | None = None
    ):
        
        """
            Definition of the POD+NN NIROM architecture

            Args
            ----
            x_train: torch.Tensor
                input data for the model
            y_train: torch.Tensor
                vector output data for the model
            x_val: torch.Tensor
                validation input data for the model
            y_val: torch.Tensor
                validation vector output data for the model
            ric: float
                relative information content value for POD
            latent_nn: nn.Sequential
                latent space neural network architecture
            input_transform: Standardize or Normalize or None
                data transform for the input data
            vector_output_transform: Standardize or Normalize or None
                data transform for the vector output data
            latent_output_transform: Standardize or Normalize or None
                latent space data transform
        """

        super().__init__()

        assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, "input data must be a two dimensional tensor"
        assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, "vector output data must be a two dimensional tensor"
        assert isinstance(x_val, torch.Tensor) and x_val.ndim == 2, "validation input data must be a two dimensional tensor"
        assert isinstance(y_val, torch.Tensor) and y_val.ndim == 2, "validation scalar output data must be a two dimensional tensor"

        # assigning data attributes
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        assert isinstance(latent_nn, nn.Sequential), "latent space network must be an instance of the nn.Sequential class"

        # assigning network attributes
        self.latent_nn = latent_nn

        # assigning the data transforms
        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), "input data transform must be an instance of the Normalize or Standardize class"
        self.input_transform = input_transform

        if vector_output_transform is not None:
            assert isinstance(vector_output_transform, Normalize) or isinstance(vector_output_transform, Standardize), "output data transform must be an instance of the Normalize or Standardize class"  
        self.vector_output_transform = vector_output_transform

        self.latent_output_transform = latent_output_transform # defining latent transform

        # defining the POD model
        self.POD_model = POD(s_train=self.y_train.mT, ric=ric, snapshot_transform=self.vector_output_transform)

    @property
    def parameters(self):
        return list(self.latent_nn.parameters())

    def fit_POD(self):
        # fitting and applying POD
        self.POD_model.fit()
        self.latent_vars = self.POD_model.encoding(self.y_train.mT)

        # update the latent nn model
        last_layer = self.latent_nn[-1]
        self.latent_nn[-1] = torch.nn.Linear(last_layer.in_features, self.POD_model.k).to(**args)

    def fit(
            self,
            seed: int,
            optimizer: torch.optim.Optimizer,
            epochs: int = 1000,
            loss_func: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
    ):
        
        """
            Method to train the POD+NN architecture

            Args
            ----
            optimizer: torch.optim.Optimizer
                Optimizer that will be used to train the MHA-Net
            epochs: int
                Number of epochs to train the model
            seed: int
                Seed used for random initialization of the model
            loss_func: torch.nn.modules.loss._Loss
                Loss function for training the neural network 
        """

        assert isinstance(optimizer, torch.optim.Optimizer), "optimizer must be an instance of torch.optim.Optimizer"
        assert isinstance(epochs, int), "number of epochs must be an integer"
        assert isinstance(loss_func, torch.nn.modules.loss._Loss), "loss function must be an instance of torch.nn.modules.loss._Loss"

        # applying the data transforms
        xtrain = self.input_transform.transform(self.x_train) if self.input_transform is not None else self.x_train
        xval = self.input_transform.transform(self.x_val) if self.input_transform is not None else self.x_val

        # generating validation latent vars
        val_latent = self.POD_model.encoding(self.y_val.mT)

        if self.latent_output_transform is not None:
            self.latent_transform = self.latent_output_transform(self.latent_vars)
            encoding_train = self.latent_transform.transform(self.latent_vars)
            encoding_val = self.latent_transform.transform(val_latent)
        else:
            encoding_train = self.latent_vars
            encoding_train = val_latent
        best_error = float("inf")
        for epoch in range(epochs):

            optimizer.zero_grad()
            
            # latent loss
            latent_pred = self.latent_nn(xtrain)
            latent_loss = loss_func(latent_pred, encoding_train)

            latent_loss.backward()
            optimizer.step()

            # validation predictions - validation loss only calculated for vector predictions
            val_preds = self.latent_nn(xval)
            val_loss = loss_func(encoding_val, val_preds)
            os.makedirs("./best_models/", exist_ok=True)
            if epoch%100==0:
                if val_loss < best_error:
                    best_error = val_loss
                    torch.save(self.state_dict(),f"./best_models/pod_best_{seed}.pth")

    def predict(
            self,
            x_predict: torch.Tensor
    ) -> torch.Tensor:
        
        """
            Method to generate a prediction using the model for given inputs

            Args
            ----
            x_predict: torch.Tensor
                inputs at which prediction is desired
        """
        assert x_predict.device == self.x_train.device, "the provided data must have the same device as the training data"
        
        if self.input_transform is not None:
            x_predict = self.input_transform.transform(x_predict)

        with torch.no_grad():
            latent_prediction = self.latent_nn(x_predict)
            if self.latent_output_transform is not None:
                latent_prediction = self.latent_transform.inverse_transform(latent_prediction)
            vector_prediction = self.POD_model.decoding(latent_prediction)
        
        return vector_prediction
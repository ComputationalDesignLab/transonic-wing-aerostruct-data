import torch
import torch.nn as nn
from scimlstudio.utils import Standardize, Normalize
import os

# implementation for MHA Net - https://www.sciencedirect.com/science/article/pii/S1270963824000415
args = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}
class MHANet(nn.Module):

    def __init__(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            h_train: torch.Tensor, 
            x_val: torch.Tensor,
            y_val: torch.Tensor,
            h_val: torch.Tensor, 
            encoder: nn.Sequential,
            decoder: nn.Sequential,
            latent_nn: nn.Sequential,
            coeff_nn: nn.Sequential,
            input_transform: Standardize | Normalize | None = None,
            scalar_output_transform: Standardize | Normalize | None = None,
            vector_output_transform: Standardize | Normalize | None = None,
    ):
        
        """
            Definition of the MHA-Net architecture

            Args
            ----
            x_train: torch.Tensor
                input data for the model
            y_train: torch.Tensor
                scalar output data for the model
            h_train: torch.Tensor
                vector output data for the model
            x_val: torch.Tensor
                validation input data for the model
            y_val: torch.Tensor
                validation scalar output data for the model
            h_val: torch.Tensor
                validation vector output data for the model            
            encoder: nn.Sequential
                encoder network architecture
            decoder: nn.Sequential
                decoder network architecture
            latent_nn: nn.Sequential
                latent space neural network architecture
            coeff_nn: nn.Sequential
                coefficient neural network architecture
            input_transform: Standardize or Normalize or None
                data transform for the input data
            scalar_output_transform: Standardize or Normalize or None
                data transform for the scalar output data
            vector_output_transform: Standardize or Normalize or None
                data transform for the vector output data
        """

        super().__init__()

        assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, "training input data must be a two dimensional tensor"
        assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, "training scalar output data must be a two dimensional tensor"
        assert isinstance(h_train, torch.Tensor) and h_train.ndim == 2, "training vector output data must be a two dimensional tensor"
        assert isinstance(x_val, torch.Tensor) and x_val.ndim == 2, "validation input data must be a two dimensional tensor"
        assert isinstance(y_val, torch.Tensor) and y_val.ndim == 2, "validation scalar output data must be a two dimensional tensor"
        assert isinstance(h_val, torch.Tensor) and h_val.ndim == 2, "validation vector output data must be a two dimensional tensor"

        # assigning data attributes
        self.x_train = x_train
        self.y_train = y_train
        self.h_train = h_train
        self.x_val = x_val
        self.y_val = y_val
        self.h_val = h_val

        assert isinstance(encoder, nn.Sequential), "encoder network must be an instance of the nn.Sequential class"
        assert isinstance(decoder, nn.Sequential), "decoder network must be an instance of the nn.Sequential class"
        assert isinstance(latent_nn, nn.Sequential), "latent space network must be an instance of the nn.Sequential class"
        assert isinstance(coeff_nn, nn.Sequential), "coefficient network must be an instance of the nn.Sequential class"

        # assigning network attributes
        self.encoder = encoder
        self.decoder = decoder
        self.latent_nn = latent_nn
        self.coeff_nn = coeff_nn

        # assigning the data transforms
        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), "input data transform must be an instance of the Normalize or Standardize class"
        self.input_transform = input_transform

        if scalar_output_transform is not None:
            assert isinstance(scalar_output_transform, Normalize) or isinstance(scalar_output_transform, Standardize), "output data transform must be an instance of the Normalize or Standardize class"  
        self.scalar_output_transform = scalar_output_transform

        if vector_output_transform is not None:
            assert isinstance(vector_output_transform, Normalize) or isinstance(vector_output_transform, Standardize), "output data transform must be an instance of the Normalize or Standardize class"  
        self.vector_output_transform = vector_output_transform

    @property
    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.latent_nn.parameters()) + list(self.coeff_nn.parameters())
    
    def fit(
            self,
            seed: int,
            optimizer: torch.optim.Optimizer,
            epochs: int = 1000,
            loss_func: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
    ):
        
        """
            Method to train the MHA-Net architecture

            Args
            ----
            optimizer: torch.optim.Optimizer
                Optimizer that will be used to train the MHA-Net
            epochs: int
                Number of epochs to train the model
            loss_func: torch.nn.modules.loss._Loss
                Loss function for training the neural network 
            seed: int
                Seed value set for the training
        """

        assert isinstance(optimizer, torch.optim.Optimizer), "optimizer must be an instance of torch.optim.Optimizer"
        assert isinstance(epochs, int), "number of epochs must be an integer"
        assert isinstance(loss_func, torch.nn.modules.loss._Loss), "loss function must be an instance of torch.nn.modules.loss._Loss"

        # applying the data transforms
        xtrain = self.input_transform.transform(self.x_train) if self.input_transform is not None else self.x_train
        ytrain = self.scalar_output_transform.transform(self.y_train) if self.scalar_output_transform is not None else self.y_train
        htrain = self.vector_output_transform.transform(self.h_train) if self.vector_output_transform is not None else self.h_train

        xval = self.input_transform.transform(self.x_val) if self.input_transform is not None else self.x_val
        yval = self.scalar_output_transform.transform(self.y_val) if self.scalar_output_transform is not None else self.y_val
        hval = self.vector_output_transform.transform(self.h_val) if self.vector_output_transform is not None else self.h_val

        best_error = float("inf")
        for epoch in range(epochs):

            optimizer.zero_grad()
            
            # autoencoding loss
            encoding = self.encoder(htrain)
            reconstruction = self.decoder(encoding)
            auto_loss = loss_func(reconstruction, htrain)

            # latent loss
            latent_pred = self.latent_nn(xtrain)
            latent_loss = loss_func(latent_pred, encoding)

            # coeff loss
            coeff_pred = self.coeff_nn(encoding)
            coeff_loss = loss_func(coeff_pred, ytrain)

            # total loss
            total_loss = auto_loss + latent_loss + coeff_loss
            total_loss.backward()
            optimizer.step()

            # validation predictions - validation loss only calculated for coeff and vector predictions
            scalar_preds, vector_preds = self.predict_val(xval)
            val_loss = loss_func(vector_preds, hval) + loss_func(scalar_preds, yval)
            os.makedirs("./best_models/", exist_ok=True)
            if epoch%100==0:
                if val_loss < best_error:
                    best_error = val_loss
                    torch.save(self.state_dict(), f"./best_models/mha_net_best_{seed}.pth")

    def predict_val(
            self,
            x_val: torch.Tensor
    ) -> torch.Tensor:
        
        """
            Method to generate a prediction using the model for standardized validations inputs

            Args
            ----
            x_val: torch.Tensor
                standardized validation inputs at which prediction is desired
        """
        assert x_val.device == self.x_train.device, "the provided data must have the same device as the training data"

        with torch.no_grad():
            latent_prediction = self.latent_nn(x_val)
            vector_prediction = self.decoder(latent_prediction)
            scalar_prediction = self.coeff_nn(latent_prediction)        
        return scalar_prediction, vector_prediction

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
            vector_prediction = self.decoder(latent_prediction)
            scalar_prediction = self.coeff_nn(latent_prediction)

        if self.scalar_output_transform is not None:
            scalar_prediction = self.scalar_output_transform.inverse_transform(scalar_prediction)
        if self.vector_output_transform is not None:
            vector_prediction = self.vector_output_transform.inverse_transform(vector_prediction)
        
        return scalar_prediction, vector_prediction
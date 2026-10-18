import torch

# defining helper function for calculating error metrics
def r2_calculation(
    true_values: torch.Tensor, 
    predicted_values: torch.Tensor,
    dim: int = 1,
):
    """

        Helper function for calculating the r^2 value

        Args
        ----
        true_values: torch.Tensor
            ground truth values for calculation 
        predicted_values: torch.Tensor
            predicted values from the model for calculation
        dim: int
            0 or 1 for a per point or per sample calculation

        Returns
        -------
        r2 : torch.Tensor
            r^2 value for the provided data
            
    """
    r2 = 1 - torch.sum((true_values - predicted_values) ** 2, dim=dim) / torch.sum((true_values - torch.mean(true_values, dim=dim, keepdim=True)) ** 2, dim=dim)
    return r2

def nrmse_calculation(
    true_values: torch.Tensor, 
    predicted_values: torch.Tensor
):
    """

        Helper function for calculating the nrmse value in a per point manner

        Args
        ----
        true_values: torch.Tensor
            ground truth values for calculation 
        predicted_values: torch.Tensor
            predicted values from the model for calculation

        Returns
        -------
        nrmse : torch.Tensor
            nrmse value for the provided data
            
    """
    rmse = torch.sqrt(torch.mean((true_values-predicted_values)**2, dim=0))
    maximum, _ = torch.max(true_values, dim=0)
    minimum, _ = torch.min(true_values, dim=0)
    nrmse = rmse / (maximum - minimum) 
    return nrmse
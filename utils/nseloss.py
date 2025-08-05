import torch


class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1):
        super().__init__()
        eps = torch.tensor(eps, dtype=torch.float32)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true) ** 2
        self.eps = self.eps.to(q_stds.device)
        weights = 1 / (q_stds + self.eps) ** 2
        # print(squared_error.shape, weights.shape)  # TEST

        weights = weights.reshape(-1, 1, 1) #TEST，是否必须
        # print(weights.shape)  # TEST

        scaled_loss = weights * squared_error
        # print(torch.mean(scaled_loss))  # TEST
        # raise ValueError # TEST

        return torch.mean(scaled_loss)

# class NewLoss(torch.nn.Module):
#     def __init__(self, eps: float = 0.1):
#         super(NewLoss, self).__init__()
#         self.eps = eps
#
#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
#         # print(y_pred.shape)
#
#         mse = torch.mean((y_pred - y_true) ** 2)
#         sigma2 = q_stds ** 2
#
#         # print(mse.shape,sigma2.shape)
#
#         first_coff = sigma2 / (sigma2 + 1)
#         first = first_coff * (mse / (sigma2 + mse))
#         second_coff = 1 / (sigma2 + 1)
#         second = second_coff * (mse / (1 + mse))
#
#         new_loss = first + second
#
#         # print(new_loss.shape)
#         # #test
#         # print(new_loss)
#         # print("----------------------")
#         # nse = NSELoss()
#         # print(nse(y_pred, y_true, q_stds))
#
#         return torch.mean(new_loss)

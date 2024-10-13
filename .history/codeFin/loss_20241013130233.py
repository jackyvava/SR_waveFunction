import torch
import torch.nn as nn

# MSE Loss function
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, u_pred, u_true):
        return self.mse(u_pred, u_true)


# GL2 Physical Loss function
class GL2Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(GL2Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, u, psi):
        # Gradient of u (approximating the Laplacian)
        grad_u_x = torch.gradient(u, dim=2)  # x-direction gradient
        grad_u_y = torch.gradient(u, dim=3)  # y-direction gradient

        # Laplacian of u (second order derivatives)
        laplacian_u_x = torch.gradient(grad_u_x, dim=2)
        laplacian_u_y = torch.gradient(grad_u_y, dim=3)
        laplacian_u = laplacian_u_x + laplacian_u_y

        # |psi|^2 term
        psi_magnitude = torch.abs(psi)**2

        # Physical equation term from GL2
        gl2_term = laplacian_u + (1 / self.epsilon**2) * (1 - psi_magnitude) * psi

        # Loss: The residual of the GL2 equation should be minimized
        return torch.mean(gl2_term**2)

class TotalLoss(nn.Module):
    def __init__(self, lambda_gl2=0.1, epsilon=1.0):
        super(TotalLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.gl2_loss = GL2Loss(epsilon=epsilon)
        self.lambda_gl2 = lambda_gl2

    def forward(self, u_pred, u_true, psi_pred):
        # MSE loss
        mse_loss = self.mse_loss(u_pred, u_true)
        
        # GL2 physical loss
        gl2_loss = self.gl2_loss(u_pred, psi_pred)
        
        # Total loss with lambda as the weight
        total_loss = mse_loss + self.lambda_gl2 * gl2_loss
        
        return total_loss


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
import torch

# GL2 Physical Loss function with consideration for 2D velocity field (u_x, u_y)
class GL2Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(GL2Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, u, psi):
        batch_size = u.size(0)  # Extract the batch size
        total_loss = 0.0
        
        # Loop over each sample in the batch
        for i in range(batch_size):
            # Extract velocity field (u_x, u_y) and wave function (Re(psi), Im(psi)) for this sample
            u_x = u[i, 0, :, :]  # u_x component
            u_y = u[i, 1, :, :]  # u_y component
            psi_real = psi[i, 0, :, :]  # Real part of psi
            psi_imag = psi[i, 1, :, :]  # Imaginary part of psi

            # Calculate gradients for u_x and u_y
            grad_u_x_x = torch.gradient(u_x, dim=0)[0]  # x-direction gradient of u_x
            grad_u_x_y = torch.gradient(u_x, dim=1)[0]  # y-direction gradient of u_x
            grad_u_y_x = torch.gradient(u_y, dim=0)[0]  # x-direction gradient of u_y
            grad_u_y_y = torch.gradient(u_y, dim=1)[0]  # y-direction gradient of u_y

            # Compute Laplacian of u_x and u_y
            laplacian_u_x = torch.gradient(grad_u_x_x, dim=0)[0] + torch.gradient(grad_u_x_y, dim=1)[0]
            laplacian_u_y = torch.gradient(grad_u_y_x, dim=0)[0] + torch.gradient(grad_u_y_y, dim=1)[0]
            
            # Total Laplacian
            laplacian_u = laplacian_u_x + laplacian_u_y
            
            # |psi|^2 term (magnitude of the complex wave function)
            psi_magnitude = torch.sqrt(psi_real**2 + psi_imag**2)
            
            # GL2 equation residual term
            gl2_term = laplacian_u + (1 / self.epsilon**2) * (1 - psi_magnitude**2) * (psi_real + 1j * psi_imag)

            # Sum the squared residuals of the GL2 equation for this sample
            sample_loss = torch.mean(torch.abs(gl2_term)**2)
            total_loss += sample_loss

        # Return the averaged loss over the batch
        return total_loss / batch_size


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


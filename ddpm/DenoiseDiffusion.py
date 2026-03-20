from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

from utils import gather


class DenoiseDiffusion:
    """
    Denoise Diffusion
    """
    
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """"
        Parameters:
            eps_model: The noise prediction model (U-Net)
            n_steps: The number of diffusion steps
            device: The device to run the model on
        """
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        # Create a linear schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
        

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Diffusion Process] Calculate the mean and variance of the distribution q(xt|x0)
        
        Parameters:
            x0: The original image (batch_size, channels, height, width)
            t: The time step
            
        Returns:
            mean: the mean of the distribution q(xt|x0)
            var: the variance of the distribution q(xt|x0)
        """
        
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        
        return mean, var
    
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        [Diffusion Process] Sample from the distribution q(xt|x0)
        
        Parameters:
            x0: The original image (batch_size, channels, height, width)
            t: The time step
        
        Returns:
            xt: The noisy image (batch_size, channels, height, width)
        """
        
        if eps is None:
            eps = torch.randn_like(x0)
        
        mean, var = self.q_xt_x0(x0, t)
        xt = mean + var ** 0.5 * eps
        
        return xt
    
    
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        [Reverse Process] Sample from the distribution p(xt-1|xt)
        
        Parameters:
            xt: The noisy image at time step t (batch_size, channels, height, width)
            t: The time step 
            
        Returns:
            xt_prev: The denoised image at time step t-1 (batch_size, channels, height, width)
        """
        
        # eps_model: U-Net Model
        # eps_theta: the predicted noise from U-Net
        eps_theta = self.eps_model(xt, t)
        
        # Sampling
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        
        return mean + var ** 0.5 * eps
    
    
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        calculate the loss between epsilon & epsilon_theta

        Args:
            x0 (torch.Tensor): The original image (batch_size, channels, height, width)
            noise (Optional[torch.Tensor], optional): epsilon ~ N(0, I)
            
        Returns:
            loss (torch.Tensor): The loss between the true noise (epsilon) and the predicted noise (epsilon_theta)
        """
        
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # noise is None: find noise from x0
        if noise is None:
            noise = torch.randn_like(x0)

        # run Diffusion Process to get xt
        xt = self.q_sample(x0, t, eps=noise)
        # run U-Net to get eps_theta
        eps_theta = self.eps_model(xt, t)

        # return the loss between the true noise (epsilon) and the predicted noise (epsilon_theta)
        return F.mse_loss(noise, eps_theta)

        
        
        

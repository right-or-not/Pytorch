from re import X

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize

from labml import experiment, monit
from DenoiseDiffusion import DenoiseDiffusion, gather
from experiment import Configs


class Sampler:
    """
    Sampler Class
    """

    def __int__(self, deffusion: DenoiseDiffusion, image_channels: int, image_size: int, device: torch.device):
        """
        Init Sampler Class
        """
        
        self.diffusion = deffusion
        self.image_channels = image_channels
        self.image_size = image_size
        self.device = device

        # T
        self.n_steps = self.diffusion.n_steps
        self.eps_model = self.diffusion.eps_model
        self.beta = self.diffusion.beta
        self.alpha = self.diffusion.alpha
        self.alpha_bar = self.diffusion.alpha_bar
        alpha_bar_tm1 = torch.cat([
            self.alpha_bar.new_ones((1,)),
            self.alpha_bar[:-1]
        ])

        # Calculate
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        self.mu_tilde_coef1 = self.beta * (alpha_bar_tm1 ** 0.5) / (1 - self.alpha_bar)
        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        self.sigma2 = self.beta


    def show_image(self, img, title=""):
        """Helper function to show an image"""
        img = img.clip(0, 1)
        img = img.cpu().numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def make_video(self, frames, path="video.mp4"):
        """Helper function to create a video"""
        import imageio
        # 20 second video
        writer = imageio.get_writer(path, fps=len(frames) // 20)
        # Add each image
        for f in frames:
            f = f.clip(0, 1)
            f = to_pil_image(resize(f, [368, 368]))
            writer.append_data(np.array(f))

        writer.close()

    
    def sample_animation(self, n_frams: int = 1000, create_video: bool = True):
        """
        Sample an image step-by-step
        """

        # xt
        xt = torch.randn(
            (1, self.image_channels, self.image_size, self.image_size), 
            device=self.device
        )

        # Interval to log x_0
        interval = self.n_steps // n_frams
        # Frames for video
        frames = []
        # Sample T steps
        for t_inv in monit.iterate("Denoise", self.n_steps):
            # t
            t_ = self.n_steps - t_inv - 1
            # t in a tensor
            t = xt.new_full((1,), t_, dtype=torch.long)
            # eps_theta(x_t, t)
            eps_theta = self.eps_model(xt, t)
            if t_ % interval == 0:
                x0 = self.p_x0(xt, t, eps_theta)
                frames.append(x0[0])
                if not create_video:
                    self.show_image(x0[0], title=f"Step {t_}")
            # Sample p_theta(x_{t-1} | x_t)
            xt = self.p_sample(xt, t, eps_theta)

        if create_video:
            self.make_video(frames)

        

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, lambda_: float = 0.5, t_: int = 100):
        """
        Interpolate between two images
        """

        # Number of samples
        n_samples = x1.shape[0]
        # t in a tensor
        t = torch.full(
            (n_samples,),
            t_,
            device=self.device
        )
        # xt
        xt = (1 - lambda_) * self.diffusion.q_sample(x1, t) + lambda_ * self.diffusion.q_sample(x2, t)

        # x_0
        return self._sample_x0(xt, t_)
    

    def interpolate_animate(self, x1: torch.Tensor, x2: torch.Tensor, n_frames: int = 100, t_: int = 100, create_video=True):
        """
        Interpolate between two images step-by-step
        """

        # Show original images
        self.show_image(x1, "x1")
        self.show_image(x2, "x2")
        # Add batch dimension
        x1 = x1[None, :, :, :]        
        x2 = x2[None, :, :, :]
        # t tensor
        t = torch.full((1, ), t_, device=self.device)
        # x_t q(x_t | x_0)
        x1t = self.diffusion.q_sample(x1, t)
        x2t = self.diffusion.q_sample(x2, t)

        frames = []
        # get frames with different lambda_
        for i in monit.iterate("Interpolate", n_frames + 1, is_children_silent=True):
            lambda_ = i / n_frames
            xt = (1 - lambda_) * x1t + lambda_ * x2t
            x0 = self._sample_x0(xt, t_)
            frames.append(x0[0])
            if not create_video:
                self.show_image(x0[0], title=f"Lambda {lambda_:.2f}")
   
        if create_video:
            self.make_video(frames=frames)


    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        pass

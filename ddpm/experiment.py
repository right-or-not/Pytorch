from typing import List

from labml_nn import diffusion
from torch import optim
import torchvision
from PIL import Image

import torch
import torch.utils.data
from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from zmq import has
from DenoiseDiffusion import DenoiseDiffusion
from unet import UNet
from labml_nn.helpers.device import DeviceConfigs


class Configs(BaseConfigs):
    """
    Configurations
    """

    device: torch.device = DeviceConfigs()

    eps_model: UNet
    diffusion: DenoiseDiffusion

    image_channels: int = 3
    image_size: int = 32
    n_channels: int = 64
    channels_multipliers: List[int] = [1, 2, 2, 4]
    has_attention: List[bool] = [False, False, False, True]

    n_steps: int = 1_000
    batch_size: int = 64
    n_samples: int = 16
    learning_rate: float = 2e-5

    epochs: int = 1_000
    dataset: torch.utils.data.Dataset
    data_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Adam

        

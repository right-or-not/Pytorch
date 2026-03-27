from typing import List

import torchvision
from PIL import Image

import torch
import torch.utils.data
from torch import optim
from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from DenoiseDiffusion import DenoiseDiffusion
from unet import UNet
from labml_nn.helpers.device import DeviceConfigs


class Configs(BaseConfigs):
    """
    Configurations
    """

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def init(self):
        """
        Initialize the experiment
        """

        self.eps_model = UNet(
            self.image_channels,
            self.n_channels,
            self.channels_multipliers,
            self.has_attention
        ).to(self.device)

        self.diffusion = DenoiseDiffusion(
            self.eps_model,
            self.n_steps,
            self.device
        )

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            params=self.eps_model.parameters(),
            lr=self.learning_rate
        )

        # Newer labml versions don't expose tracker.set_image().
        # We save generated samples to disk in sample() instead.


        print(f"Dataset: {self.dataset.__class__.__name__}")
        print(f"Module: {self.eps_model}")
        print(f"Using device: {self.device}")

        
    def sample(self):
        """
        Sample Images
        """

        with torch.no_grad():
            x = torch.randn(
                [self.n_samples, self.image_channels, self.image_size, self.image_size],
                device=self.device
            )

            for t_ in monit.iterate("Sample", self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(
                    xt=x,
                    t=x.new_full((self.n_samples,), t, dtype=torch.long)
                )

            samples_dir = lab.get_experiments_path() / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            sample_path = samples_dir / f"{tracker.get_global_step():08d}.png"
            torchvision.utils.save_image(x.clamp(0, 1), str(sample_path), nrow=int(self.n_samples ** 0.5))

    
    def train(self):
        """
        Train
        """

        for data in monit.iterate("Train", self.data_loader):
            tracker.add_global_step()
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            tracker.save("loss", loss)


    def run(self):
        """
        Training Loop
        """
        for _ in monit.loop(self.epochs):
            self.train()
            self.sample()
            tracker.new_line()

        checkpoints_dir = lab.get_experiments_path() / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.eps_model.state_dict(), checkpoints_dir / "eps_model.pt")

    

class CelebADataset(torch.utils.data.Dataset):
    """
    Celebrate a HQ dataset
    """

    def __init__(self):
        super().__init__()

        folder = lab.get_data_path() / "celebA"
        self._files = [p for p in folder.glob("**/*.jpg")]

        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        """
        Size of the dataset
        """

        return len(self._files)
    
    def __getitem__(self, index: int):
        """
        Get a image
        """
        img = Image.open(self._files[index])
        img = self._transform(img)
        return img
    


@option(Configs.dataset, 'CelebA')
def celeb_dataset(c: Configs):
    """
    Create a CelebA dataset
    """
    return CelebADataset(c.image_size)


class MNistDataset(torchvision.datasets.MNIST):
    """
    MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(
            str(lab.get_data_path()),
            train=True,
            download=True,
            transform=transform
        )
    
    def __getitem__(self, index):
        return super().__getitem__(index)[0]
    

@option(Configs.dataset, 'MNIST')
def mnist_dataset(c: Configs):
    """
    Create a MNIST dataset
    """
    return MNistDataset(c.image_size)


def main():
    # Create experiment
    experiment.create(name="diffuse", writers={"screen", "labml"})

    # Create configurations
    configs = Configs()

    # Set configurations
    experiment.configs(configs, {
        "dataset": "CeleA",
        "image_channels": 3,
        "epoches": 100,
    })

    # Initialize
    configs.init()

    # Start and run the training loop
    with experiment.start():
        configs.run()


if __name__ == '__main__':
    main()

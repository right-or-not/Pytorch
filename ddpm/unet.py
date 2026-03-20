import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

class Swish(nn.Module):
    """
    Swish activation function
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class TimeEmbedding(nn.Module):
    """
    Time embedding module: embedding for x
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.linear1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.activation = Swish()
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dim = self.n_channels // 4
        half_dim = dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb =  t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.linear1(emb)
        emb = self.activation(emb)
        emb = self.linear2(emb)

        return emb
    

class ResidualBlock(nn.Module):
    """
    Residual block with time embedding
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 16, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.activation1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.activation2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_activation = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # save x for the shortcut connection
        residual = x

        # Perform x
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        # Perform time embedding
        t = self.time_emb(t)
        t = self.time_activation(t)
        
        # Add time embedding to x
        h = x + t[:, :, None, None]

        # Perform h
        h = self.norm2(h)
        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Add shortcut: Residual connection
        h = h + self.shortcut(residual)

        return h
    

class AttentionBlock(nn.Module):
    """
    Attention block: Multi-head self attention Module
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 16):
        super().__init__()

        # parameters for multi-head attention
        if d_k is None:
            d_k = n_channels
        self.n_heads = n_heads
        self.d_k = d_k

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        # t is not used
        _ = t

        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attention = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
        attention = torch.softmax(attention, dim=-1)
        residual = torch.einsum('bhqk,bkhd->bqhd', attention, v)
        residual = residual.view(batch_size, -1, self.n_heads * self.d_k)
        residual = self.output(residual)
        residual += x
        residual = residual.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return residual


class DownBlock(nn.Module):
    """
    Down block: Residual block followed by downsampling
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attention: bool):
        super().__init__()

        self.residual = ResidualBlock(in_channels, out_channels, time_channels)

        if has_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual(x, t)
        x = self.attention(x)
        return x
    

class UpBlock(nn.Module):
    """
    Up block: Residual block followed by upsampling
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attention: bool):
        super().__init__()

        self.residual = ResidualBlock(in_channels + out_channels, out_channels, time_channels)

        if has_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual(x, t)
        x = self.attention(x)
        return x
    
class MiddleBlock(nn.Module):
    """
    Middle block: Residual block followed by attention
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()

        self.residual1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attention = AttentionBlock(n_channels)
        self.residual2 = ResidualBlock(n_channels, n_channels, time_channels)

    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual1(x, t)
        x = self.attention(x)
        x = self.residual2(x, t)
        return x
        
    
class Upsample(nn.Module):
    """
    Upsample block
    """

    def __init__(self, n_channels: int):
        super().__init__()

        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsample block
    """

    def __init__(self, n_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    """
    U-Net Module
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64, 
                 channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 has_attention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()

        n_resolutions = len(channel_multipliers)
        self.image_project = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_embedding = TimeEmbedding(n_channels * 4)

        # First half of the U-Net: decreasing resolution
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = n_channels * channel_multipliers[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, has_attention[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # Second half of the U-Net: increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, has_attention[i]))
            out_channels = in_channels // channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, has_attention[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
    
        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.activation = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_embedding(t)
        x = self.image_project(x)

        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
            
        x = self.norm(x)
        x = self.activation(x)
        x = self.final(x) 
        return x



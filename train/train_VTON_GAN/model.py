import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ResidualBlock(nn.Module):
    """Residual block with optional spectral normalization"""
    def __init__(self, channels, use_spectral_norm=False):
        super().__init__()
        
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        else:
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual


class Generator(nn.Module):
    """
    VTON-GAN Generator
    
    Input: Concatenated [person, garment, pose]
    Output: Try-on result
    """
    def __init__(self, input_channels=9, output_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, 1, 3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, 2, 1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_channels*4, config.USE_SPECTRAL_NORM) 
            for _ in range(6)
        ])
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(self, person, garment, pose):
        """
        Args:
            person: [B, 3, H, W]
            garment: [B, 3, H, W]
            pose: [B, 3, H, W]
        Returns:
            tryon_result: [B, 3, H, W]
        """
        x = torch.cat([person, garment, pose], dim=1)  # [B, 9, H, W]
        
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Residual
        r = self.res_blocks(e3)
        
        # Decode
        d3 = self.dec3(r)
        d2 = self.dec2(d3)
        out = self.dec1(d2)
        
        return out


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    
    Input: Try-on result (real or fake)
    Output: Patch-wise real/fake predictions
    """
    def __init__(self, input_channels=3, base_channels=64):
        super().__init__()
        
        use_sn = config.USE_SPECTRAL_NORM
        
        def make_layer(in_ch, out_ch, stride=2):
            if use_sn:
                conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride, 1))
            else:
                conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1)
            return nn.Sequential(conv, nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2))
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            make_layer(base_channels, base_channels*2),
            make_layer(base_channels*2, base_channels*4),
            make_layer(base_channels*4, base_channels*8, stride=1),
            
            nn.Conv2d(base_channels*8, 1, 4, 1, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] - Image
        Returns:
            out: [B, 1, H', W'] - Patch predictions
        """
        return self.model(x)


class VTONGANModel(nn.Module):
    """VTON-GAN: GAN-based Virtual Try-On"""
    def __init__(self):
        super().__init__()
        
        self.generator = Generator(
            input_channels=9,
            output_channels=3,
            base_channels=config.GENERATOR_CHANNELS
        )
        
        self.discriminator = Discriminator(
            input_channels=3,
            base_channels=config.DISCRIMINATOR_CHANNELS
        )
        
    def forward(self, person, garment, pose):
        """Generate try-on result"""
        return self.generator(person, garment, pose)


def get_vtongan_model():
    """Factory function"""
    return VTONGANModel()

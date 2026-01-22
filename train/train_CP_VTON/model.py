import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class GeometricMatchingModule(nn.Module):
    """
    GMM: Predicts TPS transformation parameters to warp garment to person shape
    
    Input: person representation + garment
    Output: TPS parameters for warping
    """
    def __init__(self, input_channels=6, feature_channels=256):
        super().__init__()
        
        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # -> 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 64x64
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # -> 32x32
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(512, 512, 4, 2, 1),  # -> 8x8
            nn.ReLU(),
        )
        
        # Regression head for TPS parameters
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * 5 * 5)  # 5x5 grid of control points (x, y)
        )
        
    def forward(self, person_repr, garment):
        """
        Args:
            person_repr: [B, 3, H, W] - Person representation (pose/segmentation)
            garment: [B, 3, H, W] - Garment image
        Returns:
            tps_params: [B, 50] - TPS transformation parameters
            warped_garment: [B, 3, H, W] - Warped garment
        """
        # Concatenate inputs
        x = torch.cat([person_repr, garment], dim=1)  # [B, 6, H, W]
        
        # Extract features and predict TPS
        features = self.encoder(x)
        tps_params = self.regressor(features)
        
        # Apply TPS transformation (simplified - use kornia in practice)
        warped_garment = garment  # Placeholder
        
        return tps_params, warped_garment


class TryOnModule(nn.Module):
    """
    TOM: Synthesizes final try-on result with composition mask
    
    Input: person, warped garment, person representation
    Output: try-on result, composition mask
    """
    def __init__(self, input_channels=9, feature_channels=96):
        super().__init__()
        
        # UNet-like architecture
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels*2, 4, 2, 1),
            nn.BatchNorm2d(feature_channels*2),
            nn.LeakyReLU(0.2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(feature_channels*2, feature_channels*4, 4, 2, 1),
            nn.BatchNorm2d(feature_channels*4),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels*4, feature_channels*2, 4, 2, 1),
            nn.BatchNorm2d(feature_channels*2),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels*4, feature_channels, 4, 2, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels*2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Output heads
        self.output_img = nn.Conv2d(64, 3, 3, 1, 1)
        self.output_mask = nn.Conv2d(64, 1, 3, 1, 1)
        
    def forward(self, person, warped_garment, person_repr):
        """
        Args:
            person: [B, 3, H, W] - Person image
            warped_garment: [B, 3, H, W] - Warped garment from GMM
            person_repr: [B, 3, H, W] - Person representation
        Returns:
            tryon_result: [B, 3, H, W] - Final try-on image
            composition_mask: [B, 1, H, W] - Composition mask
        """
        # Concatenate inputs
        x = torch.cat([person, warped_garment, person_repr], dim=1)  # [B, 9, H, W]
        
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decode with skip connections
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Outputs
        tryon_img = torch.tanh(self.output_img(d1))
        composition_mask = torch.sigmoid(self.output_mask(d1))
        
        # Compose final result
        tryon_result = composition_mask * warped_garment + (1 - composition_mask) * person
        
        return tryon_result, composition_mask


class CPVTONModel(nn.Module):
    """
    CP-VTON: Characteristic-Preserving Virtual Try-On
    
    Two-stage architecture:
    1. GMM: Geometric Matching Module - warps garment to person
    2. TOM: Try-On Module - synthesizes final result with composition mask
    """
    def __init__(self):
        super().__init__()
        
        self.gmm = GeometricMatchingModule(
            input_channels=6,
            feature_channels=config.GMM_FEATURE_CHANNELS
        )
        
        self.tom = TryOnModule(
            input_channels=9,
            feature_channels=config.TOM_FEATURE_CHANNELS
        )
        
    def forward(self, person, garment, person_repr):
        """
        Args:
            person: [B, 3, H, W] - Person image
            garment: [B, 3, H, W] - Garment image
            person_repr: [B, 3, H, W] - Person representation (pose/segmentation)
        Returns:
            tryon_result: [B, 3, H, W] - Final try-on result
            warped_garment: [B, 3, H, W] - Warped garment
            composition_mask: [B, 1, H, W] - Composition mask
            tps_params: [B, 50] - TPS parameters
        """
        # Stage 1: Geometric Matching
        tps_params, warped_garment = self.gmm(person_repr, garment)
        
        # Stage 2: Try-On Synthesis
        tryon_result, composition_mask = self.tom(person, warped_garment, person_repr)
        
        return tryon_result, warped_garment, composition_mask, tps_params


def get_cpvton_model():
    """Factory function"""
    return CPVTONModel()

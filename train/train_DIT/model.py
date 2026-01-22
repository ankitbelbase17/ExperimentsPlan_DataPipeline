import torch
import torch.nn as nn
import math
from diffusers import AutoencoderKL, DDPMScheduler
import config

def modulate(x, shift, scale):
    """Modulation operation for adaptive layer norm"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations"""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance"""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: [B, N, D] - Input tokens
            c: [B, D] - Conditioning vector (timestep + class)
        Returns:
            x: [B, N, D] - Output tokens
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """Final layer to project back to pixel space"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTModel(nn.Module):
    """
    Diffusion Transformer (DiT)
    
    Input: Noisy latents [B, C, H, W]
    Conditioning: Timestep + Class label
    Output: Predicted noise [B, C, H, W]
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Patchify
        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size, bias=True)
        
        # Positional embedding
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # Conditioning embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights"""
        # Initialize positional embedding
        pos_embed = self._get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[-2] ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize other layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim, grid_size):
        """2D sinusoidal positional embeddings"""
        import numpy as np
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = DiTModel._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid)
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """1D sinusoidal positional embeddings"""
        import numpy as np
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def unpatchify(self, x):
        """
        x: [B, N, patch_size^2 * C]
        imgs: [B, C, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y):
        """
        Args:
            x: [B, C, H, W] - Noisy latents
            t: [B] - Timesteps
            y: [B] - Class labels
        Returns:
            output: [B, C, H, W] - Predicted noise (or noise + variance if learn_sigma)
        """
        # Patchify
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * self.patch_size * self.patch_size)
        x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        
        # Conditioning
        t_emb = self.t_embedder(t)  # [B, D]
        y_emb = self.y_embedder(y, self.training)  # [B, D]
        c = t_emb + y_emb  # [B, D]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # [B, N, patch_size^2 * out_channels]
        x = self.unpatchify(x)  # [B, out_channels, H, W]
        
        return x


def get_dit_model():
    """Factory function to create DiT model with VAE"""
    dit = DiTModel(
        input_size=config.LATENT_SIZE,
        patch_size=config.DIT_CONFIG["patch_size"],
        in_channels=config.IN_CHANNELS,
        hidden_size=config.DIT_CONFIG["hidden_size"],
        depth=config.DIT_CONFIG["depth"],
        num_heads=config.DIT_CONFIG["num_heads"],
        mlp_ratio=config.DIT_CONFIG["mlp_ratio"],
        class_dropout_prob=config.DIT_CONFIG["class_dropout_prob"],
        num_classes=config.DIT_CONFIG["num_classes"],
        learn_sigma=config.DIT_CONFIG["learn_sigma"],
    )
    
    # Load VAE from SD2 Base
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae")
    vae.requires_grad_(False)
    
    return dit, vae

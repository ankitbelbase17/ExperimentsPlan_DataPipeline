"""
Calculate DiT Model Parameter Count

Based on DiT-XL/2 configuration:
- hidden_size: 1152
- depth: 28 (transformer blocks)
- num_heads: 16
- patch_size: 2
- input_size: 32 (256/8 latent size)
"""

def calculate_dit_params():
    # Configuration
    hidden_size = 1152
    depth = 28
    num_heads = 16
    patch_size = 2
    in_channels = 4
    out_channels = 8  # 4 * 2 (with learn_sigma)
    mlp_ratio = 4.0
    num_classes = 1000
    input_size = 32  # 256 / 8
    
    # Calculate components
    num_patches = (input_size // patch_size) ** 2  # 16x16 = 256 patches
    mlp_hidden_dim = int(hidden_size * mlp_ratio)  # 4608
    
    print("=" * 60)
    print("DiT-XL/2 Parameter Count Calculation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Depth (Blocks): {depth}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Patch Size: {patch_size}")
    print(f"  Input Size: {input_size}x{input_size}")
    print(f"  Num Patches: {num_patches}")
    print(f"  MLP Hidden Dim: {mlp_hidden_dim}")
    
    # 1. Patchify (x_embedder)
    patchify_params = (patch_size * patch_size * in_channels) * hidden_size + hidden_size
    print(f"\n1. Patchify Layer:")
    print(f"   Linear: ({patch_size}*{patch_size}*{in_channels}) x {hidden_size} + bias")
    print(f"   Parameters: {patchify_params:,}")
    
    # 2. Positional Embedding (frozen, not counted)
    pos_embed_params = num_patches * hidden_size
    print(f"\n2. Positional Embedding (frozen):")
    print(f"   Shape: {num_patches} x {hidden_size}")
    print(f"   Parameters: {pos_embed_params:,} (not trainable)")
    
    # 3. Timestep Embedder
    freq_embed_size = 256
    t_embedder_params = (
        freq_embed_size * hidden_size + hidden_size +  # First linear
        hidden_size * hidden_size + hidden_size         # Second linear
    )
    print(f"\n3. Timestep Embedder:")
    print(f"   MLP: {freq_embed_size} -> {hidden_size} -> {hidden_size}")
    print(f"   Parameters: {t_embedder_params:,}")
    
    # 4. Label Embedder
    use_cfg = True  # class dropout > 0
    label_embedder_params = (num_classes + use_cfg) * hidden_size
    print(f"\n4. Label Embedder:")
    print(f"   Embedding: {num_classes + use_cfg} x {hidden_size}")
    print(f"   Parameters: {label_embedder_params:,}")
    
    # 5. DiT Blocks (repeated depth times)
    # Each block has:
    # - LayerNorm1 (no params, elementwise_affine=False)
    # - MultiheadAttention
    # - LayerNorm2 (no params)
    # - MLP
    # - adaLN modulation (for attention)
    # - adaLN modulation (for MLP)
    
    # MultiheadAttention params
    attn_params = (
        hidden_size * hidden_size * 3 +  # Q, K, V projections (no bias in standard)
        hidden_size * hidden_size         # Output projection
    )
    
    # MLP params
    mlp_params = (
        hidden_size * mlp_hidden_dim + mlp_hidden_dim +  # First linear + bias
        mlp_hidden_dim * hidden_size + hidden_size        # Second linear + bias
    )
    
    # adaLN modulation params (6 * hidden_size for each block)
    adaln_params = hidden_size * (6 * hidden_size) + (6 * hidden_size)
    
    block_params = attn_params + mlp_params + adaln_params
    total_blocks_params = block_params * depth
    
    print(f"\n5. DiT Blocks (x{depth}):")
    print(f"   Per Block:")
    print(f"     - MultiheadAttention: {attn_params:,}")
    print(f"     - MLP: {mlp_params:,}")
    print(f"     - adaLN Modulation: {adaln_params:,}")
    print(f"     - Total per block: {block_params:,}")
    print(f"   Total for {depth} blocks: {total_blocks_params:,}")
    
    # 6. Final Layer
    # - LayerNorm (no params)
    # - Linear projection
    # - adaLN modulation
    final_linear_params = hidden_size * (patch_size * patch_size * out_channels) + (patch_size * patch_size * out_channels)
    final_adaln_params = hidden_size * (2 * hidden_size) + (2 * hidden_size)
    final_layer_params = final_linear_params + final_adaln_params
    
    print(f"\n6. Final Layer:")
    print(f"   Linear: {hidden_size} -> {patch_size * patch_size * out_channels}")
    print(f"   adaLN Modulation: {hidden_size} -> {2 * hidden_size}")
    print(f"   Parameters: {final_layer_params:,}")
    
    # Total
    total_params = (
        patchify_params +
        t_embedder_params +
        label_embedder_params +
        total_blocks_params +
        final_layer_params
    )
    
    print("\n" + "=" * 60)
    print("TOTAL PARAMETERS")
    print("=" * 60)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Total in Millions: {total_params / 1e6:.2f}M")
    print(f"Total in Billions: {total_params / 1e9:.3f}B")
    
    # Comparison with paper
    print("\n" + "=" * 60)
    print("Comparison with DiT Paper")
    print("=" * 60)
    print("DiT-XL/2 (from paper): ~675M parameters")
    print(f"Our calculation: {total_params / 1e6:.2f}M parameters")
    
    # Memory estimate
    print("\n" + "=" * 60)
    print("Memory Estimates")
    print("=" * 60)
    
    # FP32: 4 bytes per parameter
    fp32_memory = total_params * 4 / (1024**3)
    print(f"Model weights (FP32): {fp32_memory:.2f} GB")
    
    # BF16: 2 bytes per parameter
    bf16_memory = total_params * 2 / (1024**3)
    print(f"Model weights (BF16): {bf16_memory:.2f} GB")
    
    # Training memory (rough estimate: 4x model size for gradients + optimizer states)
    training_memory_fp32 = fp32_memory * 4
    training_memory_bf16 = bf16_memory * 4
    print(f"\nTraining memory (FP32, rough): ~{training_memory_fp32:.2f} GB")
    print(f"Training memory (BF16, rough): ~{training_memory_bf16:.2f} GB")
    
    print("\n" + "=" * 60)
    
    return total_params


if __name__ == "__main__":
    total = calculate_dit_params()

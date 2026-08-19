import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaloAttention(nn.Module):
    """
    Local self-attention with halo padding
    (Vaswani et al., "Scaling Local Self-Attention for Parameter Efficient Visual Backbones", CVPR 2021)

    Each spatial block attends to itself plus a halo of neighboring context.
    Q is computed from the original block; K, V include the halo region.

    Args:
        dim:        number of input channels
        block_size: spatial block side length (default 8)
        halo_size:  number of extra context pixels on each side (default 2)
        num_heads:  attention heads (default 4)
    """

    def __init__(self, dim: int, block_size: int = 8, halo_size: int = 2, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W) with residual connection
        """
        B, C, H, W = x.shape
        residual = x

        # pre-norm (channel-last layernorm)
        x_n = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        qkv = self.qkv(x_n)
        q, k, v = qkv.chunk(3, dim=1)

        bs = self.block_size
        hs = self.halo_size

        # --- pad Q so H, W are divisible by block_size ---
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        q = F.pad(q, [0, pad_w, 0, pad_h])          # (B, C, H', W')
        _, _, H_p, W_p = q.shape

        # --- pad K, V with halo + same block padding ---
        k = F.pad(k, [hs, hs + pad_w, hs, hs + pad_h])
        v = F.pad(v, [hs, hs + pad_w, hs, hs + pad_h])

        num_h = H_p // bs
        num_w = W_p // bs
        block_ext = bs + 2 * hs                       # K/V block side length

        # extract non-overlapping patches via unfold
        # q: (B, C, num_h, num_w, bs, bs)
        q = q.unfold(2, bs, bs).unfold(3, bs, bs)
        # k: (B, C, num_h, num_w, block_ext, block_ext)
        k = k.unfold(2, block_ext, bs).unfold(3, block_ext, bs)
        v = v.unfold(2, block_ext, bs).unfold(3, block_ext, bs)

        # reshape for multi-head attention
        # (B, C, nh, nw, bh, bw) -> (B, nh, nw, heads, bh*bw, head_dim)
        def reshape_heads(t, spatial_prod):
            return (
                t.reshape(B, self.num_heads, self.head_dim, num_h, num_w, spatial_prod)
                 .permute(0, 3, 4, 1, 5, 2)       # (B, nh, nw, heads, spatial, head_dim)
            )

        q_flat = bs * bs
        kv_flat = block_ext * block_ext

        q = reshape_heads(q, q_flat)
        k = reshape_heads(k, kv_flat)
        v = reshape_heads(v, kv_flat)

        # scaled dot-product attention  (all block computations happen in one batched matmul)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v                                 # (B, nh, nw, heads, q_flat, head_dim)

        # reshape back to spatial
        out = (
            out.permute(0, 3, 5, 1, 2, 4)             # (B, heads, head_dim, nh, nw, q_flat)
               .reshape(B, C, num_h, num_w, bs, bs)
        )
        # interleave block dims back to spatial grid
        out = out.permute(0, 1, 2, 4, 3, 5)           # (B, C, num_h, bs, num_w, bs)
        out = out.reshape(B, C, H_p, W_p)

        # crop padding
        out = out[:, :, :H, :W]

        return self.proj(out) + residual


class SpatioTemporalAttention(nn.Module):
    """
    Bridge module between Predrnn encoder and decoder.
    Concatenates the encoder's last-layer hidden state with the current
    decoder feature, fuses via 1x1 conv, then applies Halo attention.

    Args:
        hidden_dim:  feature channels
        block_size:  halo block size (default 8)
        halo_size:   halo padding (default 2)
        num_heads:   attention heads (default 4)
    """

    def __init__(self, hidden_dim: int, block_size: int = 8,
                 halo_size: int = 2, num_heads: int = 4):
        super().__init__()
        self.fuse = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.halo = HaloAttention(hidden_dim, block_size, halo_size, num_heads)

    def forward(self, encoder_hidden, decoder_input):
        """
        encoder_hidden: (B, D, H, W)  last encoder layer at last timestep
        decoder_input:  (B, D, H, W)  current decoder step input
        returns:        (B, D, H, W)  enhanced decoder input
        """
        fused = self.fuse(torch.cat([encoder_hidden, decoder_input], dim=1))
        return self.halo(fused)

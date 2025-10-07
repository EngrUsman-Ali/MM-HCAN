import torch
import torch.nn as nn

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MultiHeadAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x_list):
        x_stack = torch.stack(x_list, dim=0)  # (3, B, D)
        attn_output, _ = self.multihead_attn(x_stack, x_stack, x_stack)
        fused = torch.mean(attn_output, dim=0)  # (B, D)
        return fused


class SingleModalityFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SingleModalityFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        """
        x: (B, D) tensor of concatenated features
        Returns:
            fused: (B, D) → refined via self-attention
        """
        # Expand to (seq_len=1, B, D) for attention
        x_expanded = x.unsqueeze(0)  # (1, B, D)

        # Apply self-attention
        attn_output, _ = self.multihead_attn(x_expanded, x_expanded, x_expanded)

        # Remove sequence dimension
        fused = attn_output.squeeze(0)  # (B, D)

        return fused
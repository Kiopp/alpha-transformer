import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_actions, num_meta_features, embed_dim=64, num_heads=4, num_blocks=3):
        super().__init__()
        
        # Instantiate embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len, 
            embedding_dim=embed_dim
        )

        # Project meta-data rules into embedding space
        self.meta_projection = nn.Linear(num_meta_features, embed_dim)

        # Regularization
        self.drop = nn.Dropout(0.1)

        # Instantiate transformer block
        self.t_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )

        # Instantiate final layer that projects from embedding space back to vocab space
        self.final_norm = nn.LayerNorm(embed_dim)

        # +1 to max_seq_len because meta_data token is prepended to the gameboard tokens
        flattened_dim = embed_dim
        
        # Policy head predicts move probabilities
        self.policy_head = nn.Linear(flattened_dim, num_actions)

        # Value head predicts win/loss/draw
        self.value_head = nn.Sequential(
            nn.Linear(flattened_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Squeezes the output to be strictly between -1 and 1
        )

    def forward(self, board_idx, meta_features, legal_moves_mask=None):
        # idx is your input tensor of shape (batch_size, seq_len)
        batch_size, seq_len = board_idx.shape
        
        # Token Embeddings
        # (batch_size, seq_len, embed_dim)
        tok_emb = self.token_embedding(board_idx) 
        
        # Position Embeddings
        pos_ids = torch.arange(seq_len, device=board_idx.device)
        pos_emb = self.position_embedding(pos_ids) 

        # Combine Embeddings
        x = tok_emb + pos_emb 

        # Process meta features
        # Also adds sequence dimension
        meta_emb = self.meta_projection(meta_features).unsqueeze(1)
        
        # Concatinate meta token
        x = torch.cat([meta_emb, x], dim=1)
        
        x = self.drop(x)

        # Pass transformer blocks
        x = self.t_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Flatten sequence
        x_flat = x[:, 0, :]

        # Pass through output heads
        policy_logits = self.policy_head(x_flat)
        value = self.value_head(x_flat)

        # Mask illegal moves
        # Replace illegal moves in logits with negative infinity
        if legal_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_moves_mask, -1e9)
        
        return policy_logits, value

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()

        # Ensure clean maths
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.head_size = embed_dim // num_heads

        # Create parallell attention heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Create projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension

        # Project all heads simultaneously
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and transpose
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # Apply scaled dot-product attention
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.1 if self.training else 0.0,
            is_causal=False
        )

        # Transpose back and concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.out_proj(out)
        out = self.drop(out)

        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            # Expand dimention by facor 4
            nn.Linear(embed_dim, 4*embed_dim),

            # Apply GELU activation function
            nn.GELU(),

            # Contract back to original embedding space
            nn.Linear(4*embed_dim, embed_dim),

            # Regularization
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim
        )

        self.feedforward = FeedForwardNetwork(embed_dim)

    def forward(self, x):
        # Attention with pre-normalization and residual connection
        x = x + self.attention(self.norm1(x))
        # Feed-Forward network with pre-normalization and residual connection
        x = x + self.feedforward(self.norm2(x))
        return x
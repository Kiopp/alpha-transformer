import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessResNet(nn.Module):
    def __init__(self, vocab_size=13, num_actions=4096, num_meta_features=6, num_channels=128, num_blocks=8, piece_embed_dim=32):
        super().__init__()
        
        # Input Processing
        # We learn an embedding for each piece type (0-12)
        self.piece_embedding = nn.Embedding(vocab_size, piece_embed_dim)
        
        # Total input channels = piece embeddings + broadcasted meta features
        in_channels = piece_embed_dim + num_meta_features
        
        # Initial Convolutional Block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual Tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_blocks)]
        )
        
        # Policy Head (Predicts the move)
        # Squeeze channels down to 2, then flatten to dense layer
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, num_actions)
        )
        
        # Value Head (Predicts win/draw/loss)
        # Squeeze channels to 1, then pass through dense layers
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Outputs between -1 and 1
        )

    def forward(self, board_idx, meta_features, legal_moves_mask=None):
        batch_size = board_idx.size(0)
        
        # --- Transform 1D Inputs into 2D Spatial Tensors ---
        
        # Embed pieces: (Batch, 64) -> (Batch, 64, EmbedDim)
        x_board = self.piece_embedding(board_idx)
        
        # Reshape to spatial grid: (Batch, EmbedDim, 8, 8)
        # permute moves the channel dimension to the PyTorch standard position
        x_board = x_board.view(batch_size, 8, 8, -1).permute(0, 3, 1, 2)
        
        # Broadcast meta features: (Batch, 6) -> (Batch, 6, 8, 8)
        # We stretch the 6 global values across every square of the 8x8 board
        x_meta = meta_features.view(batch_size, -1, 1, 1).expand(batch_size, -1, 8, 8)
        
        # Concatenate along the channel dimension -> (Batch, EmbedDim + 6, 8, 8)
        x = torch.cat([x_board, x_meta], dim=1)
        
        # --- Pass through the Neural Network ---
        
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        # --- Mask Illegal Moves ---
        if legal_moves_mask is not None:
            # Replace illegal moves in logits with negative infinity
            policy_logits = policy_logits.masked_fill(~legal_moves_mask, -1e9)
            
        return policy_logits, value
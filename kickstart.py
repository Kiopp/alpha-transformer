import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# --- Robust Dataset ---
class ChessDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        board, meta, mask, pi, z = self.experiences[idx]
        
        board_t = torch.as_tensor(board, dtype=torch.long).cpu().squeeze()
        meta_t = torch.as_tensor(meta, dtype=torch.float32).cpu().squeeze()
        mask_t = torch.as_tensor(mask, dtype=torch.bool).cpu().squeeze()
        pi_t = torch.as_tensor(pi, dtype=torch.float32).cpu().squeeze() 
        z_t = torch.tensor([z], dtype=torch.float32)

        return board_t, meta_t, mask_t, pi_t, z_t

def run_kickstart(epochs=15, batch_size=128):
    game = ChessGame()
    
    print("Initializing Medium Model Architecture...")
    # --- MEDIUM MODEL ---
    model = ChessTransformer(
        vocab_size=13, 
        max_seq_len=64, 
        num_actions=game.action_size, 
        num_meta_features=6, 
        embed_dim=256,   
        num_heads=8,     
        num_blocks=10
    ).to(game.device)

    buffer_path = "replay_buffer.pt"
    if not os.path.exists(buffer_path):
        print(f"Error: Could not find {buffer_path}!")
        return

    print(f"Loading Golden Data from {buffer_path}...")
    master_replay_buffer = torch.load(buffer_path, weights_only=False)
    print(f"Successfully loaded {len(master_replay_buffer)} historical positions.")

    dataset = ChessDataset(master_replay_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Uses a lower learning rate and higher weight decay for the larger model
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    value_criterion = nn.MSELoss()
    
    print(f"\nStarting Supervised Kickstart for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_val_loss = 0
        total_pol_loss = 0
        
        for boards, metas, masks, target_pis, target_zs in dataloader:
            boards, metas, masks = boards.to(game.device), metas.to(game.device), masks.to(game.device)
            target_pis, target_zs = target_pis.to(game.device), target_zs.to(game.device)
            
            optimizer.zero_grad()
            
            policy_logits, values = model(boards, metas, masks)
            value_loss = value_criterion(values, target_zs)
            
            log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.sum(target_pis * log_probs) / target_pis.size(0) 
            
            loss = value_loss + policy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_val_loss += value_loss.item()
            total_pol_loss += policy_loss.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_val = total_val_loss / len(dataloader)
        avg_pol = total_pol_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{epochs} | Total: {avg_loss:.4f} (Value: {avg_val:.4f} | Policy: {avg_pol:.4f})")

    save_path = "chess_transformer_iter_0.pth"
    checkpoint = {
        'iteration': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"\n>>> Kickstart Complete! Medium model saved to: {save_path} <<<")
    print("You can now resume normal training with the training script.")

if __name__ == "__main__":
    run_kickstart()
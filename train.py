import torch
torch.cuda.set_per_process_memory_fraction(0.8, device=0) # Allocate maximum of 80% of available vram
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import glob
import re
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Since I use an amd gpu

# --- Dataset ---
class ChessDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        board, meta, mask, pi, z = self.experiences[idx]
        return board.squeeze(0), meta.squeeze(0), mask.squeeze(0), torch.tensor(pi, dtype=torch.float32), torch.tensor([z], dtype=torch.float32)

# --- Self-play generator ---
def execute_episode(model, game, mcts_simulations=100):
    mcts = MCTS(model, game, num_simulations=mcts_simulations)
    state = game.get_initial_state()
    train_examples = []
    
    while True:
        board_tensor, meta_tensor, legal_mask = game.prepare_inputs(state)
        pi = mcts.search(state)
        current_player = 1 if state.turn else -1
        train_examples.append([board_tensor, meta_tensor, legal_mask, pi, current_player])
        
        # Choose action based on temperature
        tau = max(0.1, 1.0 - (state.fullmove_number / 30.0))
        adjusted_pi = np.power(pi + 1e-8, 1.0 / tau)
        adjusted_pi = adjusted_pi / np.sum(adjusted_pi) # re-normalize
        action = np.random.choice(len(adjusted_pi), p=adjusted_pi)
            
        state = game.get_next_state(state, action)
        is_terminal, reward = game.get_reward_and_terminal(state)
        
        if is_terminal:
            final_data = []
            for example in train_examples:
                is_white_turn = (example[4] == 1)
                if reward == 0:
                    z = 0.0
                elif (reward == 1 and is_white_turn) or (reward == -1 and not is_white_turn):
                    z = 1.0
                else:
                    z = -1.0 
                final_data.append((example[0], example[1], example[2], example[3], z))
            return final_data

# --- Find latest checkpoint ---
def get_latest_checkpoint():
    """Scans the directory for the latest checkpoint file."""
    files = glob.glob("chess_transformer_iter_*.pth")
    if not files:
        return None, -1
    
    max_iter = -1
    latest_file = None
    for f in files:
        match = re.search(r"iter_(\d+)\.pth", f)
        if match:
            it = int(match.group(1))
            if it > max_iter:
                max_iter = it
                latest_file = f
    return latest_file, max_iter

# --- Training loop ---
def train_alphazero(model, game, episodes_per_iter=20, epochs=4, batch_size=16, keep_last_n_checkpoints=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    value_criterion = nn.MSELoss()
    
    # Check for existing saves to resume
    latest_file, last_iter = get_latest_checkpoint()
    
    if latest_file:
        print(f"Found checkpoint: {latest_file}. Resuming training...")
        # weights_only=False is required to load optimizer dictionaries in newer PyTorch versions
        checkpoint = torch.load(latest_file, map_location=game.device, weights_only=False)
        
        # Handle backwards compatibility if you load an older save that only had model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        current_iter = last_iter + 1
    else:
        print("No previous checkpoints found. Starting fresh.")
        current_iter = 0

    # Load buffer
    buffer_path = "replay_buffer.pt"
    if os.path.exists(buffer_path):
        print(f"Loading replay buffer from {buffer_path}...")
        # weights_only=False is required because the buffer contains lists and standard Python types
        master_replay_buffer = torch.load(buffer_path, weights_only=False)
        print(f"Successfully loaded {len(master_replay_buffer)} historical positions.")
    else:
        print("No existing replay buffer found. Starting with an empty buffer.")
        master_replay_buffer = [] # Keep games played in previous iterations

    try:
        while True: # INFINITE LOOP
            print(f"\n======================================")
            print(f"     STARTING ITERATION {current_iter}")
            print(f"======================================")
            
            # Step 1: Self-Play
            print(f"Playing {episodes_per_iter} games of self-play...")
            for e in range(episodes_per_iter):
                with torch.no_grad():
                    model.eval() 
                    episode_data = execute_episode(model, game)
                    master_replay_buffer.extend(episode_data)
                if (e+1) % 2 == 0:
                    print(f"  Completed {e+1}/{episodes_per_iter} games...")

            # Step 2: Prepare Dataset
            if len(master_replay_buffer) > 50000: 
                master_replay_buffer = master_replay_buffer[-50000:] # Set limit for previous games remembered
            dataset = ChessDataset(master_replay_buffer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Step 3: Train
            print(f"\nTraining network for {epochs} epochs on {len(master_replay_buffer)} positions...")
            model.train()
            
            for epoch in range(epochs):
                total_loss = 0
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
                    # Clip gradients to a maximum norm of 1.0 to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                print(f"  Epoch {epoch+1}/{epochs} | Average Loss: {total_loss/len(dataloader):.4f}")

            # Step 4: Save Checkpoint with Optimizer State
            save_path = f"chess_transformer_iter_{current_iter}.pth"
            checkpoint = {
                'iteration': current_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, save_path)
            print(f"\n>>> Checkpoint saved: {save_path} <<<")
            print(f"Saving replay buffer to disk...")
            torch.save(master_replay_buffer, buffer_path)
            print(">>> Replay buffer saved successfully. <<<")

            # Cleanup old checkpoints to save disk space
            old_checkpoint = f"chess_transformer_iter_{current_iter - keep_last_n_checkpoints}.pth"
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"Deleted old checkpoint: {old_checkpoint}")
                
            current_iter += 1

    except KeyboardInterrupt:
        # Graceful exit when you press Ctrl+C
        print("\n\n" + "="*50)
        print(" TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("="*50)
        print("Performing emergency save of current replay buffer...")
        torch.save(master_replay_buffer, buffer_path)
        print("Emergency save complete.")
        if current_iter > 0:
            print(f"Safe exit complete. You can resume later from: iter_{current_iter-1}.pth")
        else:
            print("Exited before completing the first iteration.")

if __name__ == "__main__":
    game = ChessGame()
    model = ChessTransformer(
        vocab_size=13, max_seq_len=64, num_actions=game.action_size, 
        num_meta_features=6, embed_dim=128, num_heads=8, num_blocks=6
    ).to(game.device)
    
    # Starts the infinite training loop.
    # Press Ctrl+C in your terminal when you wake up to stop it safely!
    train_alphazero(model, game, episodes_per_iter=40, epochs=4)
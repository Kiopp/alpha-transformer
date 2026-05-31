import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
torch.cuda.set_per_process_memory_fraction(0.8, device=0) # Allocate maximum of 80% of available vram
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
import re
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS
import torch.multiprocessing as mp
import signal

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Since I use an amd gpu

def init_worker():
    """Forces child processes to ignore CTRL+C so the main process can handle it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
# --- Dataset ---
class ChessDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        board, meta, mask, pi, z = self.experiences[idx]
        
        # torch.as_tensor safely handles both old PyTorch tensors and new NumPy arrays.
        board_t = torch.as_tensor(board, dtype=torch.long).cpu().squeeze()
        meta_t = torch.as_tensor(meta, dtype=torch.float32).cpu().squeeze()
        mask_t = torch.as_tensor(mask, dtype=torch.bool).cpu().squeeze()
        
        # pi and z don't have the batch dimension issue, but we standardize them anyway
        pi_t = torch.as_tensor(pi, dtype=torch.float32).cpu().squeeze() 
        z_t = torch.tensor([z], dtype=torch.float32)

        return board_t, meta_t, mask_t, pi_t, z_t

# --- Self-play generator ---
def parallel_execute_episode(worker_args):
    model, game, mcts_sims, worker_id = worker_args
    
    # Force this specific process to only use 1 CPU thread for tensor math
    # so that the concurrent workers don't cause thread contention.
    torch.set_num_threads(1) 
    
    # Run the exact same episode logic
    return execute_episode(model, game, mcts_simulations=mcts_sims)

def execute_episode(model, game, mcts_simulations=100):
    mcts = MCTS(model, game, num_simulations=mcts_simulations)
    state = game.get_initial_state()
    train_examples = []
    
    while True:
        # Boost simulation count when board is sparce
        if len(state.piece_map()) <= 10:
            mcts.num_simulations = mcts_simulations * 2  # Boosts to 800
        else:
            mcts.num_simulations = mcts_simulations

        board_tensor, meta_tensor, legal_mask = game.prepare_inputs(state)
        pi = mcts.search(state)
        current_player = 1 if state.turn else -1
        train_examples.append([board_tensor.cpu().numpy(), meta_tensor.cpu().numpy(), legal_mask.cpu().numpy(), pi.astype(np.float32), current_player])
        
        # Choose action based on temperature
        if state.fullmove_number <= 30:
            tau = 1.0  # Opening variation
        elif state.fullmove_number <= 70:
            tau = 0.5  # Keep middlegame fluid to force pawn pushes/trades
        else:
            tau = 0.05  # Focused play (Lock-in)
            
        valid_moves_mask = pi > 0
        adjusted_pi = np.zeros_like(pi)
        adjusted_pi[valid_moves_mask] = np.power(pi[valid_moves_mask], 1.0 / tau)
        adjusted_pi = adjusted_pi / np.sum(adjusted_pi) # re-normalize
        action = np.random.choice(len(adjusted_pi), p=adjusted_pi)
            
        state = game.get_next_state(state, action)
        is_terminal, reward = game.get_reward_and_terminal(state)
        
        if is_terminal:
            if reward != 0.0:
                if state.is_checkmate():
                    reason = "Checkmate"
                else:
                    reason = "Tiebreaker"
            else:
                # Find specific draw trigger
                if state.can_claim_fifty_moves() or state.is_fifty_moves():
                    reason = "Draw_50_Move"
                elif state.can_claim_threefold_repetition() or state.is_repetition(3):
                    reason = "Draw_Threefold"
                elif state.is_insufficient_material():
                    reason = "Draw_Insufficient"
                elif state.is_stalemate():
                    reason = "Draw_Stalemate"
                else:
                    reason = "Draw_Max_Horizon"

            final_data = []
            for example in train_examples:
                is_white_turn = (example[4] == 1)
                
                if reward == 0.0:
                    z = 0.0
                # If White won and it was White's turn OR Black won and it was Black's turn
                elif (reward == 1.0 and is_white_turn) or (reward == -1.0 and not is_white_turn):
                    z = 1.0  # The player who made this move won the game
                else:
                    z = -1.0 # The player who made this move lost the game
                    
                final_data.append((example[0], example[1], example[2], example[3], z))
            return (final_data, reward, reason)

# --- Find latest checkpoint ---
def get_latest_checkpoint(filename):
    """Scans the directory for the latest checkpoint file."""
    files = glob.glob(f"{filename}_iter_*.pth")
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
def train_alphazero(model, game, episodes_per_iter=20, epochs=4, batch_size=128, keep_last_n_checkpoints=5, num_workers=4, num_sims=100, max_buffer_size=50000, max_buffer_sample=10000):
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    value_criterion = nn.MSELoss()
    filename = "chess_medium"

    # Guard agains overfitting on value loss
    value_loss_weight = 0.5
    
    # Check for existing saves to resume
    latest_file, last_iter = get_latest_checkpoint(filename)
    
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

    # Setup cpu isolation for workers
    print("Setting up CPU isolation for self-play workers...")

    # Create CPU-bound game environment to avoid GPU allocation
    cpu_game = ChessGame()
    cpu_game.device = torch.device('cpu')

    # CPU replica of the model
    cpu_model = ChessTransformer(
        vocab_size=13, max_seq_len=64, num_actions=game.action_size, 
        num_meta_features=6, embed_dim=256, num_heads=8, num_blocks=10
    ).to('cpu')
    cpu_model.share_memory()

    try:
        while True: # INFINITE LOOP
            print(f"\n======================================")
            print(f"     STARTING ITERATION {current_iter}")
            print(f"======================================")

            # Sync weights to CPU model
            print("Syncing updated GPU weights to CPU model for self-play...")
            cpu_model.load_state_dict(model.state_dict())
            cpu_model.eval()
            
            # Step 1: Self-Play
            print(f"Playing {episodes_per_iter} games of self-play using {num_workers} CPU workers with {num_sims} simulations...")

            # Prepare arguments for each worker
            worker_tasks = [(cpu_model, cpu_game, num_sims, i) for i in range(episodes_per_iter)]
            
            outcomes = {"White Wins": 0, "Black Wins": 0, "Draws": 0}
            win_methods = {"Checkmate": 0, "Tiebreaker": 0}
            draw_reasons = {"Draw_50_Move": 0, "Draw_Threefold": 0, "Draw_Insufficient": 0, "Draw_Stalemate": 0, "Draw_Max_Horizon": 0}
            total_plies = 0
            with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
                # pool.imap_unordered yields results as soon as any game finishes
                for i, result in enumerate(pool.imap_unordered(parallel_execute_episode, worker_tasks)):
                    episode_data, absolute_reward, reason = result
                    
                    # Update stats
                    total_plies += len(episode_data)
                    if absolute_reward == 1.0: 
                        outcomes["White Wins"] += 1
                        win_methods[reason] += 1
                    elif absolute_reward == -1.0: 
                        outcomes["Black Wins"] += 1
                        win_methods[reason] += 1
                    else: 
                        outcomes["Draws"] += 1
                        if reason in draw_reasons:
                            draw_reasons[reason] += 1
                        else:
                            draw_reasons["Draw_Max_Horizon"] += 1 # Fallback for edge cases

                    # Add to replay buffer
                    master_replay_buffer.extend(episode_data)
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Completed {i + 1}/{episodes_per_iter} games...")

            # Display self-play stats
            # Map the raw dictionary keys to cleaner display names
            draw_display_names = {
                "Draw_50_Move": "50-Move",
                "Draw_Threefold": "Threefold",
                "Draw_Insufficient": "Insuff. Mat",
                "Draw_Stalemate": "Stalemate",
                "Draw_Max_Horizon": "Horizon"
            }

            # Filter and format only the draw reasons that actually occurred
            active_draws = [f"{draw_display_names[k]}: {v}" for k, v in draw_reasons.items() if v > 0]
            draw_string = " | ".join(active_draws) if active_draws else "None"

            avg_len = total_plies / episodes_per_iter
            print(f"\n--- Self-play stats ---")
            print(f"Outcomes: {outcomes}")
            print(f"Win Methods: Checkmates: {win_methods['Checkmate']} | Tiebreakers: {win_methods['Tiebreaker']}")
            print(f"Draw Reasons: {draw_string}")
            print(f"Avg Game Length: {avg_len:.1f} plies")
            print(f"Buffer Size: {len(master_replay_buffer)}")
            print(f"-------------------------------\n")

            # Step 2: Prepare Dataset
            if len(master_replay_buffer) > max_buffer_size:
                print(f"Flushing {len(master_replay_buffer)-max_buffer_size} positions from replay buffer...") 
                master_replay_buffer = master_replay_buffer[-max_buffer_size:] # Set limit for previous games remembered
            
            sample_size = min(len(master_replay_buffer), max_buffer_sample) # Only train on a subset of positions per iteration to avoid overfitting
            sampled_buffer = random.sample(master_replay_buffer, sample_size)
            dataset = ChessDataset(sampled_buffer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Step 3: Train
            print(f"Training mid-sized network for {epochs} epochs on {len(sampled_buffer)} / {len(master_replay_buffer)} positions...")
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
                    
                    policy_loss = torch.nn.functional.cross_entropy(policy_logits, target_pis)
                    
                    loss = (value_loss * value_loss_weight) + policy_loss

                    loss.backward()
                    # Clip gradients to a maximum norm of 1.0 to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_val_loss += value_loss.item()
                    total_pol_loss += policy_loss.item()
                
                # Print 
                avg_loss = total_loss / len(dataloader)
                avg_val = total_val_loss / len(dataloader)
                avg_pol = total_pol_loss / len(dataloader)
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Value: {avg_val:.4f} | Policy: {avg_pol:.4f})")

            # Step 4: Save Checkpoint with Optimizer State
            save_path = f"{filename}_iter_{current_iter}.pth"
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
            old_checkpoint = f"{filename}_iter_{current_iter - keep_last_n_checkpoints}.pth"
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
        temp_filepath = f"{buffer_path}.tmp"
        torch.save(master_replay_buffer, temp_filepath)
        os.replace(temp_filepath, buffer_path)
        print("Emergency save complete.")
        if current_iter > 0:
            print(f"Safe exit complete. You can resume later from: iter_{current_iter-1}.pth")
        else:
            print("Exited before completing the first iteration.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    game = ChessGame()
    model = ChessTransformer(
        vocab_size=13, max_seq_len=64, num_actions=game.action_size, 
        num_meta_features=6, embed_dim=256, num_heads=8, num_blocks=10
    ).to(game.device)
    
    # Starts the infinite training loop.
    train_alphazero(model, game, episodes_per_iter=30, epochs=2, batch_size=256, num_workers=6, num_sims=400, max_buffer_size=250000, max_buffer_sample=50000)

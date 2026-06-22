import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Since I use an amd gpu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
import re
import queue 
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS
import torch.multiprocessing as mp
import signal
    
# --- Dataset ---
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

# --- Asynchronous Inference Server Architecture ---

class InferenceClient:
    """Acts as a dummy PyTorch model for the MCTS, routing requests to the GPU server."""
    def __init__(self, worker_id, req_queue, res_pipe):
        self.worker_id = worker_id
        self.req_queue = req_queue
        self.res_pipe = res_pipe

    def eval(self):
        pass 

    def __call__(self, board_tensor, meta_tensor, legal_mask):
        self.req_queue.put((self.worker_id, board_tensor, meta_tensor, legal_mask))
        policy_logits, value_tensor = self.res_pipe.recv()
        return policy_logits, value_tensor

def inference_server_process(model, req_queue, res_pipes, num_workers, device):
    """Sits on the GPU, batches requests from CPU workers, and evaluates them simultaneously."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_grad_enabled(False)
    model.eval()

    active_workers = num_workers
    while active_workers > 0:
        requests = []
        try:
            req = req_queue.get(timeout=1.0)
            if req[1] is None: 
                active_workers -= 1
            else:
                requests.append(req)
        except queue.Empty:
            continue

        while len(requests) < active_workers:
            try:
                req = req_queue.get_nowait()
                if req[1] is None:
                    active_workers -= 1
                else:
                    requests.append(req)
            except queue.Empty:
                break

        if not requests:
            continue

        worker_ids = [r[0] for r in requests]
        
        boards = torch.cat([r[1] for r in requests]).to(device)
        metas = torch.cat([r[2] for r in requests]).to(device)
        masks = torch.cat([r[3] for r in requests]).to(device)

        policy_logits, values = model(boards, metas, masks)

        policy_logits = policy_logits.cpu()
        values = values.cpu()

        for i, w_id in enumerate(worker_ids):
            res_pipes[w_id].send((policy_logits[i].unsqueeze(0), values[i].unsqueeze(0)))

def self_play_worker(worker_id, num_games, game_instance, mcts_sims, req_queue, res_pipe, return_dict):
    """The CPU worker loop. Plays N games sequentially by querying the InferenceClient."""
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    client = InferenceClient(worker_id, req_queue, res_pipe)
    worker_buffer = []
    outcomes = {"White Wins": 0, "Black Wins": 0, "Draws": 0}
    win_methods = {"Checkmate": 0, "Tiebreaker": 0}
    draw_reasons = {"Draw_50_Move": 0, "Draw_Threefold": 0, "Draw_Insufficient": 0, "Draw_Stalemate": 0, "Draw_Max_Horizon": 0}
    total_plies = 0

    for i in range(num_games):
        episode_data, absolute_reward, reason = execute_episode(client, game_instance, mcts_sims)
        worker_buffer.extend(episode_data)
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
                draw_reasons["Draw_Max_Horizon"] += 1
                
        if (i + 1) == num_games:
            print(f"  -> Worker {worker_id} finished all {num_games} games!")
        elif num_games > 1 and (i + 1) == (num_games // 2):
            print(f"  -> Worker {worker_id} is 50% done ({i+1}/{num_games} games)...")

    req_queue.put((worker_id, None, None, None))
    
    return_dict[worker_id] = {
        "buffer": worker_buffer,
        "outcomes": outcomes,
        "win_methods": win_methods,
        "draw_reasons": draw_reasons,
        "total_plies": total_plies
    }

def execute_episode(model, game, mcts_simulations=100):
    mcts = MCTS(model, game, num_simulations=mcts_simulations)
    state = game.get_initial_state()
    train_examples = []
    
    while True:
        piece_count = len(state.piece_map())
        if piece_count <= 6:
            mcts.num_simulations = int(mcts_simulations * 2.5)
        elif piece_count <= 10:
            mcts.num_simulations = int(mcts_simulations * 2)
        elif piece_count <= 12:
            mcts.num_simulations = int(mcts_simulations * 1.5)
        else:
            mcts.num_simulations = mcts_simulations

        board_tensor, meta_tensor, legal_mask = game.prepare_inputs(state)
        pi = mcts.search(state)
        current_player = 1 if state.turn else -1
        train_examples.append([board_tensor.cpu().numpy(), meta_tensor.cpu().numpy(), legal_mask.cpu().numpy(), pi.astype(np.float32), current_player])
        
        if state.fullmove_number <= 30:
            tau = 1.0  
        elif state.fullmove_number <= 70:
            tau = 0.5  
        elif state.fullmove_number <= 100:
            tau = 0.25  
        else:
            tau = 0.125 
            
        valid_moves_mask = pi > 0
        adjusted_pi = np.zeros_like(pi)
        adjusted_pi[valid_moves_mask] = np.power(pi[valid_moves_mask], 1.0 / tau)
        adjusted_pi = adjusted_pi / np.sum(adjusted_pi) 
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
                elif (reward == 1.0 and is_white_turn) or (reward == -1.0 and not is_white_turn):
                    z = 1.0  
                else:
                    z = -1.0 
                final_data.append((example[0], example[1], example[2], example[3], z))
            return (final_data, reward, reason)

def get_latest_checkpoint(filename):
    files = glob.glob(f"{filename}_iter_*.pth")
    if not files: return None, -1
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
def train_alphazero(model, game, episodes_per_iter=40, epochs=2, batch_size=512, keep_last_n_checkpoints=5, num_workers=10, num_sims=400, max_buffer_size=250000, max_buffer_sample=50000, enable_scheduler=True):
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    value_criterion = nn.MSELoss()
    filename = "chess_medium"

    value_loss_weight = 2.5
    
    latest_file, last_iter = get_latest_checkpoint(filename)
    
    if latest_file:
        print(f"Found checkpoint: {latest_file}. Resuming training...")
        checkpoint = torch.load(latest_file, map_location=game.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state.")
                #print("Skipping scheduler loading.")
            else:
                print("No scheduler state in checkpoint. Starting scheduler fresh.")
        else:
            model.load_state_dict(checkpoint)
            
        current_iter = last_iter + 1
    else:
        print("No previous checkpoints found. Starting fresh.")
        current_iter = 0

    buffer_path = "replay_buffer.pt"
    if os.path.exists(buffer_path):
        print(f"Loading replay buffer from {buffer_path}...")
        master_replay_buffer = torch.load(buffer_path, weights_only=False)
        print(f"Successfully loaded {len(master_replay_buffer)} historical positions.")
    else:
        print("No existing replay buffer found. Starting with an empty buffer.")
        master_replay_buffer = []

    # CPU environment to avoid GPU allocation on workers
    cpu_game = ChessGame()
    cpu_game.device = torch.device('cpu')

    workers = []
    server_process = None

    try:
        while True:
            print(f"\n======================================")
            print(f"     STARTING ITERATION {current_iter}")
            print(f"======================================")
            print(f"Playing {episodes_per_iter} games using Asynchronous GPU Dynamic Batching...")

            manager = mp.Manager()
            return_dict = manager.dict()
            req_queue = mp.Queue()

            # Distribute games evenly across workers
            games_per_worker = [episodes_per_iter // num_workers] * num_workers
            for i in range(episodes_per_iter % num_workers):
                games_per_worker[i] += 1

            parent_pipes = []
            worker_pipes = []
            for _ in range(num_workers):
                p, w = mp.Pipe()
                parent_pipes.append(p)
                worker_pipes.append(w)

            model.eval()
            server_process = mp.Process(
                target=inference_server_process, 
                args=(model, req_queue, parent_pipes, num_workers, game.device)
            )
            server_process.start()

            workers = []
            for i in range(num_workers):
                w = mp.Process(
                    target=self_play_worker, 
                    args=(i, games_per_worker[i], cpu_game, num_sims, req_queue, worker_pipes[i], return_dict)
                )
                w.start()
                workers.append(w)

            for w in workers:
                w.join()

            server_process.join()

            outcomes = {"White Wins": 0, "Black Wins": 0, "Draws": 0}
            win_methods = {"Checkmate": 0, "Tiebreaker": 0}
            draw_reasons = {"Draw_50_Move": 0, "Draw_Threefold": 0, "Draw_Insufficient": 0, "Draw_Stalemate": 0, "Draw_Max_Horizon": 0}
            total_plies = 0

            for w_id, w_data in return_dict.items():
                master_replay_buffer.extend(w_data["buffer"])
                total_plies += w_data["total_plies"]
                for k in outcomes: outcomes[k] += w_data["outcomes"][k]
                for k in win_methods: win_methods[k] += w_data["win_methods"][k]
                for k in draw_reasons: draw_reasons[k] += w_data["draw_reasons"][k]

            draw_display_names = {
                "Draw_50_Move": "50-Move", "Draw_Threefold": "Threefold", 
                "Draw_Insufficient": "Insuff. Mat", "Draw_Stalemate": "Stalemate", "Draw_Max_Horizon": "Horizon"
            }
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

            if len(master_replay_buffer) > max_buffer_size:
                print(f"Flushing {len(master_replay_buffer)-max_buffer_size} positions from replay buffer...") 
                master_replay_buffer = master_replay_buffer[-max_buffer_size:] 
            
            sample_size = min(len(master_replay_buffer), max_buffer_sample)
            sampled_buffer = random.sample(master_replay_buffer, sample_size)
            dataset = ChessDataset(sampled_buffer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_val_loss += value_loss.item()
                    total_pol_loss += policy_loss.item()
                
                avg_loss = total_loss / len(dataloader)
                avg_val = total_val_loss / len(dataloader)
                avg_pol = total_pol_loss / len(dataloader)
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Value: {avg_val:.4f} | Policy: {avg_pol:.4f})")

            if enable_scheduler:
                scheduler.step(avg_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr:.6f}")

            save_path = f"{filename}_iter_{current_iter}.pth"
            checkpoint = {
                'iteration': current_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, save_path)
            print(f"\n>>> Checkpoint saved: {save_path} <<<")
            print(f"Saving replay buffer to disk...")
            torch.save(master_replay_buffer, buffer_path)
            print(">>> Replay buffer saved successfully. <<<")

            old_checkpoint = f"{filename}_iter_{current_iter - keep_last_n_checkpoints}.pth"
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"Deleted old checkpoint: {old_checkpoint}")
                
            current_iter += 1

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print(" TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("="*50)
        print("Performing emergency save of current replay buffer...")
        temp_filepath = f"{buffer_path}.tmp"
        torch.save(master_replay_buffer, temp_filepath)
        os.replace(temp_filepath, buffer_path)
        print("Emergency save complete.")

        if server_process and server_process.is_alive():
            server_process.terminate()
            server_process.join()
            
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join()

        if current_iter > 0:
            print(f"Safe exit complete. You can resume later from: iter_{current_iter-1}.pth")
        else:
            print("Exited before completing the first iteration.")

if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8, device=0) 
    
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    game = ChessGame()
    model = ChessTransformer(
        vocab_size=13, max_seq_len=64, num_actions=game.action_size, 
        num_meta_features=6, embed_dim=256, num_heads=8, num_blocks=10
    ).to(game.device)
    
    train_alphazero(
        model, game, 
        episodes_per_iter=40,   # Increased throughput 
        epochs=2, 
        batch_size=512,         # Utilizing more VRAM
        num_workers=10,         # Maxing out the Ryzen 5 cores
        num_sims=400, 
        max_buffer_size=250000, 
        max_buffer_sample=50000, 
        enable_scheduler=True
    )
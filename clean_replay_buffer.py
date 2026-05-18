import torch
import random
import os
from collections import Counter
# This script is a one-time use to clean up an overfitted replay buffer that is almost completely full of draws. It will clear out most of the draws and keep the decisive games. 

def clean_replay_buffer(input_path="replay_buffer.pt", output_path="cleaned_replay_buffer.pt", draw_keep_ratio=0.10):
    if not os.path.exists(input_path):
        print(f"Error: Could not find '{input_path}'")
        return

    print(f"Loading original buffer from {input_path}...")
    # Load with weights_only=False because the buffer contains standard Python lists
    master_buffer = torch.load(input_path, map_location='cpu', weights_only=False)
    
    decisive_games = []
    draw_games = []

    # Extract based on the 'z' outcome value (index 4)
    # Structure: (board, meta, mask, pi, z)
    for exp in master_buffer:
        z = float(exp[4])
        if z == 0.0:
            draw_games.append(exp)
        else:
            decisive_games.append(exp)

    print(f"Original stats: {len(decisive_games):,} decisive positions, {len(draw_games):,} draw positions.")
    
    # Randomly sample a fraction of the draws
    num_draws_to_keep = int(len(draw_games) * draw_keep_ratio)
    sampled_draws = random.sample(draw_games, num_draws_to_keep)
    
    print(f"Filtering draws down to {(draw_keep_ratio * 100):.1f}% (Keeping {num_draws_to_keep:,})...")

    # Combine the data
    cleaned_buffer = decisive_games + sampled_draws
    
    # Shuffle thoroughly so the decisive games aren't entirely at the beginning
    print("Shuffling new buffer...")
    random.shuffle(cleaned_buffer)

    # Calculate final distribution for verification
    outcomes = [float(exp[4]) for exp in cleaned_buffer]
    counts = Counter(outcomes)
    total = len(cleaned_buffer)
    
    print("\n" + "="*30)
    print("   CLEANED BUFFER STATS")
    print("="*30)
    print(f"Total Positions: {total:,}")
    print(f"Mover Won (1.0)  : {counts.get(1.0, 0):>7,} ({(counts.get(1.0, 0)/total)*100:.1f}%)")
    print(f"Mover Lost (-1.0): {counts.get(-1.0, 0):>7,} ({(counts.get(-1.0, 0)/total)*100:.1f}%)")
    print(f"Draw (0.0)       : {counts.get(0.0, 0):>7,} ({(counts.get(0.0, 0)/total)*100:.1f}%)")
    print("="*30)
    
    # Save the new buffer to disk
    print(f"\nSaving cleaned buffer to '{output_path}'...")
    torch.save(cleaned_buffer, output_path)
    print(">>> Purge complete! <<<")

if __name__ == "__main__":
    # You can adjust the ratio here if you want more or fewer draws
    clean_replay_buffer(draw_keep_ratio=0.10)
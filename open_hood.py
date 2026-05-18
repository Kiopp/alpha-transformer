import torch
import os
from collections import Counter
# To be used in conjunction with the clean_replay_buffer script. First run this script which will open the hood of the replay buffer. Then run the clean_replay_buffer script if needed.

def analyze_buffer(buffer_path="replay_buffer.pt"):
    if not os.path.exists(buffer_path):
        print(f"Error: {buffer_path} not found.")
        return

    print(f"--- Loading Replay Buffer: {buffer_path} ---")
    # Load with weights_only=False because the buffer contains python lists/objects
    master_replay_buffer = torch.load(buffer_path, map_location='cpu', weights_only=False)
    
    total_positions = len(master_replay_buffer)
    if total_positions == 0:
        print("Buffer is empty.")
        return

    # Extract the 'z' value (outcome) from each experience
    # Structure: (board, meta, mask, pi, z)
    outcomes = [float(exp[4]) for exp in master_replay_buffer]
    
    counts = Counter(outcomes)
    
    # Map values to human-readable labels
    # Note: In your training code, z=1 means 'mover won', z=-1 means 'mover lost'
    stats = {
        "Mover Won (1.0)": counts.get(1.0, 0),
        "Mover Lost (-1.0)": counts.get(-1.0, 0),
        "Draw (0.0)": counts.get(0.0, 0)
    }

    print(f"Total Positions: {total_positions:,}")
    print("-" * 30)
    
    for label, count in stats.items():
        percentage = (count / total_positions) * 100
        print(f"{label:<18}: {count:>8,} ({percentage:>6.2f}%)")
    
    print("-" * 30)

    # Tactical Check: Are games too short or too long?
    if total_positions > 0:
        # Assuming we can't easily group by game without more metadata, 
        # but we can see if the buffer is 'stale'
        unique_outcomes = len(set(outcomes))
        if unique_outcomes == 1:
            print("WARNING: Outcome Collapse! Only one outcome type detected.")
        elif stats["Draw (0.0)"] > 80:
            print("WARNING: High Draw Rate. Potential 'Draw Death' detected.")

if __name__ == "__main__":
    analyze_buffer()
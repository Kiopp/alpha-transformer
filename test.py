import os
import argparse
import random
import torch
import chess
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Since I use an amd gpu

def print_board_with_labels(board):
    # Get the default string representation (8 lines of text)
    board_str = str(board)
    lines = board_str.split('\n')
    
    print() # Add a little breathing room
    # Loop through the lines and prepend the rank number
    for i, line in enumerate(lines):
        rank = 8 - i
        print(f"{rank}  {line}")
        
    # Print the file letters at the bottom, aligned with the columns
    print("   a b c d e f g h\n")

def play_human_vs_ai(mcts, game):
    board = game.get_initial_state()
    human_turn = True # Human plays White (True), AI plays Black (False)

    print("Game Start! Enter moves in Standard Algebraic Notation (e.g., 'e4', 'Nf3').")
    
    while not board.is_game_over():
        print_board_with_labels(board)
        
        if human_turn:
            move_str = input("Your move: ")
            try:
                board.push_san(move_str)
                human_turn = False
            except ValueError:
                print("Invalid move. Try again.")
        else:
            print("AI is thinking...")
            mcts_probs = mcts.search(board)
            best_action = mcts_probs.argmax()
            ai_move = game._action_to_move(best_action, board)
            
            if ai_move not in board.legal_moves:
                print("Warning: AI suggested illegal move. Defaulting to first legal move.")
                ai_move = list(board.legal_moves)[0]

            board.push(ai_move)
            print(f"AI plays: {ai_move}")
            human_turn = True

    print("\nGame Over!")
    print_board_with_labels(board)
    print("Result:", board.result())

def play_random_vs_ai(mcts, game):
    board = game.get_initial_state()
    is_random_white = True 

    print("Game Start: Random (White) vs AI (Black)")
    
    while not board.is_game_over():
        print_board_with_labels(board)
        current_turn_is_white = (board.turn == chess.WHITE)
        
        if (is_random_white and current_turn_is_white) or (not is_random_white and not current_turn_is_white):
            print("Random Move's turn...")
            available_actions = game.get_legal_actions(board)
            random_action = random.choice(available_actions)
            move = game._action_to_move(random_action, board)
            board.push(move)
            print(f"Random played: {move}")
            
        else:
            print("AI is thinking...")
            mcts_probs = mcts.search(board)
            best_action = mcts_probs.argmax()
            ai_move = game._action_to_move(best_action, board)
            
            if ai_move not in board.legal_moves:
                print("Warning: AI suggested illegal move. Defaulting to first legal move.")
                ai_move = list(board.legal_moves)[0]

            board.push(ai_move)
            print(f"AI plays: {ai_move}")

    print("\nGame Over!")
    print_board_with_labels(board)
    
    result = board.result()
    print(f"Final Result: {result}")
    
    if result == "1-0":
        print("Winner: Random (White)")
    elif result == "0-1":
        print("Winner: AI (Black)")
    else:
        print("Result: Draw")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Play against the AlphaZero Chess AI.")
    parser.add_argument("--model", type=str, default="chess_transformer_iter_0.pth", 
                        help="Path to the model checkpoint file (.pth)")
    parser.add_argument("--mode", type=str, choices=["human", "random"], default="random", 
                        help="Choose who plays against the AI: 'human' or 'random'")
    parser.add_argument("--sims", type=int, default=100, 
                        help="Number of MCTS simulations per turn")
    args = parser.parse_args()

    game = ChessGame()

    model = ChessTransformer(
        vocab_size=13,            
        max_seq_len=64,           
        num_actions=game.action_size, 
        num_meta_features=6,      
        embed_dim=128,            
        num_heads=8,
        num_blocks=6
    ).to(game.device)

    # --- UPDATED MODEL LOADING CODE ---
    if os.path.exists(args.model):
        print(f"Loading trained weights from {args.model}...")
        
        # weights_only=False because newer versions of PyTorch flag dictionaries with optimizer states
        checkpoint = torch.load(args.model, map_location=game.device, weights_only=False)
        
        # Check if the file contains the dictionary from train_alphazero or just raw weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from training iteration {checkpoint.get('iteration', 'Unknown')}.")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval() 
        print("Model loaded successfully!")
    else:
        print(f"Warning: '{args.model}' not found! The AI will play using random initialized weights.")
        model.eval() 
    # ------------------------------

    mcts = MCTS(model, game, num_simulations=args.sims)

    # Launch selected mode
    if args.mode == "human":
        play_human_vs_ai(mcts, game)
    else:
        play_random_vs_ai(mcts, game)

if __name__ == "__main__":
    main()

# Play command
# python test.py --model chess_transformer_iter_0.pth --mode human --sims 400
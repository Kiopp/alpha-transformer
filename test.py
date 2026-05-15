import os
#os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0" # For my Ryzen AI 7 350 laptop
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Since I use an amd gpu
import argparse
import random
import torch
import chess
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS

def print_board_with_labels(board, message=None):
    # Clear the terminal screen first
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print the last action or status message at the very top
    if message:
        print(f"\n>> {message}")
    else:
        print()

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
    status_message = "Game Start! Enter moves in SAN (e.g., 'e4', 'Nf3')."
    
    while not board.is_game_over():
        print_board_with_labels(board, status_message)
        
        if human_turn:
            move_str = input("Your move: ")
            try:
                board.push_san(move_str)
                human_turn = False
                status_message = f"You played: {move_str}"
            except ValueError:
                status_message = "Invalid move. Try again."
        else:
            print("AI is thinking...")
            mcts_probs = mcts.search(board)
            best_action = mcts_probs.argmax()
            ai_move = game._action_to_move(best_action, board)
            
            if ai_move not in board.legal_moves:
                print("Warning: AI suggested illegal move. Defaulting to first legal move.")
                ai_move = list(board.legal_moves)[0]

            board.push(ai_move)
            status_message = f"AI plays: {ai_move}"
            human_turn = True

    result = board.result()
    print_board_with_labels(board, f"Game Over! Result: {board.result()}")

def play_random_vs_ai(mcts, game):
    board = game.get_initial_state()
    is_random_white = True 
    status_message = "Game Start: Random (White) vs AI (Black)"
    
    while not board.is_game_over():
        print_board_with_labels(board, status_message)
        current_turn_is_white = (board.turn == chess.WHITE)
        
        if (is_random_white and current_turn_is_white) or (not is_random_white and not current_turn_is_white):
            print("Random Move's turn...")
            available_actions = game.get_legal_actions(board)
            random_action = random.choice(available_actions)
            move = game._action_to_move(random_action, board)
            board.push(move)
            status_message = f"Random played: {move}"
            
        else:
            print("AI is thinking...")
            mcts_probs = mcts.search(board)
            best_action = mcts_probs.argmax()
            ai_move = game._action_to_move(best_action, board)
            
            if ai_move not in board.legal_moves:
                print("Warning: AI suggested illegal move. Defaulting to first legal move.")
                ai_move = list(board.legal_moves)[0]
            
            board.push(ai_move)
            status_message = f"AI plays: {ai_move}"
    result = board.result()
    print_board_with_labels(board, f"Game Over! Result: {board.result()}")
    
    if result == "1-0":
        print("Winner: Random (White)")
    elif result == "0-1":
        print("Winner: AI (Black)")
    else:
        print("Result: Draw")

def play_ai_vs_ai(mcts_white, mcts_black, game):
    board = game.get_initial_state()
    status_message = "Game Start: AI (White) vs AI (Black)"
    
    while not board.is_game_over():
        print_board_with_labels(board, status_message)
        current_turn_is_white = (board.turn == chess.WHITE)
        
        color_str = "White" if current_turn_is_white else "Black"
        print(f"{color_str} AI is thinking...")
        
        # Select the correct MCTS instance for the current turn
        active_mcts = mcts_white if current_turn_is_white else mcts_black
        
        mcts_probs = active_mcts.search(board)
        best_action = mcts_probs.argmax()
        ai_move = game._action_to_move(best_action, board)
        
        if ai_move not in board.legal_moves:
            print(f"Warning: {color_str} AI suggested illegal move. Defaulting to first legal move.")
            ai_move = list(board.legal_moves)[0]

        board.push(ai_move)
        status_message = f"{color_str} AI plays: {ai_move}"

    result = board.result()
    print_board_with_labels(board, f"Game Over! Result: {board.result()}")
    
    if result == "1-0":
        print("Winner: AI (White)")
    elif result == "0-1":
        print("Winner: AI (Black)")
    else:
        print("Result: Draw")

def load_model(model_path, model_type, game):
    """Helper function to initialize and load weights for a model."""
    if model_type == "small":
        model = ChessTransformer(
            vocab_size=13,            
            max_seq_len=64,           
            num_actions=game.action_size, 
            num_meta_features=6,      
            embed_dim=128,            
            num_heads=8,
            num_blocks=6
        ).to(game.device)
    elif model_type == "medium":
        model = ChessTransformer(
            vocab_size=13,            
            max_seq_len=64,           
            num_actions=game.action_size, 
            num_meta_features=6,      
            embed_dim=256,            
            num_heads=8,
            num_blocks=10
        ).to(game.device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if os.path.exists(model_path):
        print(f"Loading trained weights for {model_type} model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=game.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from training iteration {checkpoint.get('iteration', 'Unknown')}.")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval() 
        print("Model loaded successfully!")
    else:
        print(f"Warning: '{model_path}' not found! The AI will play using random initialized weights.")
        model.eval() 

    return model

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Play against or between AlphaZero Chess AIs.")
    parser.add_argument("--mode", type=str, choices=["human", "random", "ai"], default="ai", 
                        help="Choose the mode: 'human' vs AI, 'random' vs AI, or 'ai' vs AI")
    parser.add_argument("--model_white", type=str, default="chess_transformer_white.pth", 
                        help="Path to the model checkpoint file for White (.pth)")
    parser.add_argument("--model_black", type=str, default="chess_transformer_black.pth", 
                        help="Path to the model checkpoint file for Black (.pth)")
    parser.add_argument("--size_white", type=str, choices=["small", "medium"], default="medium",
                        help="Size architecture of the White model (small or medium)")
    parser.add_argument("--size_black", type=str, choices=["small", "medium"], default="medium",
                        help="Size architecture of the Black model (small or medium)")
    parser.add_argument("--sims", type=int, default=100, 
                        help="Number of MCTS simulations per turn")
    args = parser.parse_args()

    game = ChessGame()

    if args.mode == "ai":
        # Load both models for AI vs AI
        print("--- Loading White AI ---")
        model_white = load_model(args.model_white, args.size_white, game)
        print("\n--- Loading Black AI ---")
        model_black = load_model(args.model_black, args.size_black, game)
        
        mcts_white = MCTS(model_white, game, num_simulations=args.sims, self_play=False)
        mcts_black = MCTS(model_black, game, num_simulations=args.sims, self_play=False)
        
        play_ai_vs_ai(mcts_white, mcts_black, game)
    else:
        # For Human vs AI or Random vs AI, the human/random defaults to White, AI plays Black.
        # We'll just load the model_black config to act as the single AI opponent.
        print("--- Loading AI Opponent (Black) ---")
        model_black = load_model(args.model_black, args.size_black, game)
        mcts_black = MCTS(model_black, game, num_simulations=args.sims, self_play=False)
        
        if args.mode == "human":
            play_human_vs_ai(mcts_black, game)
        elif args.mode == "random":
            play_random_vs_ai(mcts_black, game)

if __name__ == "__main__":
    main()

# python test.py --mode ai --model_white chess_medium_iter_10.pth --size_white medium --model_black chess_small_iter_10.pth --size_black small --sims 200
# python test.py --mode ai --model_white chess_medium_iter_10.pth --size_white medium --model_black chess_medium_iter_10.pth --size_black medium --sims 200
# python test.py --mode random --model_black chess_medium_iter_10.pth --size_black medium --sims 200
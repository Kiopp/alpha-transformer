from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS
import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Since I use an amd gpu
import random
import torch
import chess

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
    # Initialize the board state
    board = game.get_initial_state()
    
    # Human plays White (True), AI plays Black (False)
    human_turn = True 

    print("Game Start! Enter moves in Standard Algebraic Notation (e.g., 'e4', 'Nf3').")
    
    while not board.is_game_over():
        # Use our new custom print function
        print_board_with_labels(board)
        
        if human_turn:
            move_str = input("Your move: ")
            try:
                # Push the human move to the board
                board.push_san(move_str)
                human_turn = False
            except ValueError:
                print("Invalid move. Try again.")
        else:
            print("AI is thinking...")
            # 1. Run MCTS to get the best move probabilities
            mcts_probs = mcts.search(board)
            
            # 2. Pick the action with the highest visit count/probability
            best_action = mcts_probs.argmax()
            
            # 3. Convert that action index back to a chess Move and apply it
            ai_move = game._action_to_move(best_action, board)
            board.push(ai_move)
            print(f"AI plays: {ai_move}")
            human_turn = True

    print("\nGame Over!")
    print_board_with_labels(board)
    print("Result:", board.result())

def play_random_vs_ai(mcts, game):
    # Initialize the board state
    board = game.get_initial_state()
    
    # Random plays White (True), AI plays Black (False)
    # Note: In chess.Board(), board.turn == True is White's turn.
    is_random_white = True 

    print("Game Start: Random (White) vs AI (Black)")
    
    while not board.is_game_over():
        # Use your custom print function (ensure this is defined in your script)
        print_board_with_labels(board)
        
        # Check whose turn it is based on the board state
        current_turn_is_white = (board.turn == chess.WHITE)
        
        if (is_random_white and current_turn_is_white) or (not is_random_white and not current_turn_is_white):
            print("Random Move's turn...")
            # 1. Get all legal action indices from your wrapper
            available_actions = game.get_legal_actions(board)
            
            # 2. Select one at random
            random_action = random.choice(available_actions)
            
            # 3. Convert action index to Move and apply
            move = game._action_to_move(random_action, board)
            board.push(move)
            print(f"Random played: {move}")
            
        else:
            print("AI is thinking...")
            # 1. Run MCTS to get the best move probabilities (root visits)
            # Ensure mcts.search returns a numpy array or tensor of size 4096
            mcts_probs = mcts.search(board)
            
            # 2. Pick the action with the highest probability/visit count
            best_action = mcts_probs.argmax()
            
            # 3. Convert action index back to a chess Move and apply
            ai_move = game._action_to_move(best_action, board)
            
            # Safety check: MCTS might occasionally suggest an illegal move 
            # if the policy head isn't perfectly masked.
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
    # 1. Initialize the Game rules
    game = ChessGame()

    # 2. Initialize the Brain architecture
    model = ChessTransformer(
        vocab_size=13,            
        max_seq_len=64,           
        num_actions=game.action_size, 
        num_meta_features=6,      
        embed_dim=128,            
        num_heads=8,
        num_blocks=6
    ).to(game.device)

    # --- NEW MODEL LOADING CODE ---
    # Specify the file name of your latest checkpoint
    model_path = "chess_transformer_iter_2.pth" 
    
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        
        # Load the dictionary of weights. 
        # map_location ensures it loads correctly whether on CPU or your AMD GPU.
        state_dict = torch.load(model_path, map_location=game.device, weights_only=True)
        
        # Apply the weights to your model architecture
        model.load_state_dict(state_dict)
        
        # CRITICAL: Put the model in evaluation mode to disable Dropout!
        model.eval() 
        
        print("Model loaded successfully!")
    else:
        print(f"Warning: '{model_path}' not found! The AI will play using random initialized weights.")
        # Put it in eval mode anyway to disable dropout during inference
        model.eval() 
    # ------------------------------

    # 3. Initialize the Calculator
    mcts = MCTS(model, game, num_simulations=100)

    # 4. Start game
    play_random_vs_ai(mcts, game)

main()
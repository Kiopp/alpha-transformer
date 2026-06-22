import os
import argparse
import random
import threading
import torch
import chess
import tkinter as tk
from tkinter import messagebox
import glob
import re
from ChessGame import ChessGame
from ChessPlayer import ChessTransformer
from MCTS import MCTS

# Hardware compatibility optimization
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Unicode character dictionary mapping for clean vector rendering of pieces
UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White Pieces
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',  # Black Pieces
    None: ''
}

class ChessGUI:
    def __init__(self, root, game, mode, mcts_white=None, mcts_black=None, sims=200):
        self.root = root
        self.game = game
        self.mode = mode
        self.mcts_white = mcts_white
        self.mcts_black = mcts_black
        self.sims = sims
        
        self.board = game.get_initial_state()
        self.selected_square = None
        self.square_size = 70
        
        self.root.title(f"Alpha-Transformer Engine — Mode: {mode.upper()}")
        
        # Main Layout Canvas
        self.canvas = tk.Canvas(
            root, 
            width=self.square_size * 8, 
            height=self.square_size * 8
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Interactive Control & Status Bar
        self.status_label = tk.Label(
            root, 
            text="Game Start! White's turn.", 
            font=("Helvetica", 12, "bold"), 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Bind Mouse Interactions for Player Control
        self.canvas.bind("<Button-1>", self.on_square_clicked)
        
        # Initialize Visual State
        self.draw_board()
        
        # Automatically kickstart the cycle if the AI handles White
        self.check_turn_and_trigger_ai()

    def draw_board(self):
        self.canvas.delete("all")
        
        # Alternating background palette definitions
        light_color = "#E8EDF2"
        dark_color = "#7D94A6"
        highlight_color = "#BAC759"
        
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                x1, y1 = col * self.square_size, row * self.square_size
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                
                color = light_color if (row + col) % 2 == 0 else dark_color
                
                if self.selected_square == square:
                    color = highlight_color
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                piece = self.board.piece_at(square)
                if piece:
                    symbol = UNICODE_PIECES.get(piece.symbol(), '')
                    self.canvas.create_text(
                        x1 + self.square_size // 2, 
                        y1 + self.square_size // 2, 
                        text=symbol, 
                        font=("Helvetica", 36), 
                        fill="#000000" if piece.color == chess.WHITE else "#222222"
                    )

    def on_square_clicked(self, event):
        if self.mode == "ai":
            return
        if self.mode == "random" and self.board.turn == chess.WHITE:
            return
        if self.mode == "human" and self.board.turn == chess.BLACK:
            return
            
        col = event.x // self.square_size
        row = event.y // self.square_size
        clicked_square = chess.square(col, 7 - row)
        
        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, clicked_square)
            
            if self.board.piece_at(self.selected_square) and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) in [0, 7]:
                    move.promotion = chess.QUEEN
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                self.update_status_string(f"Player executed: {move.uci()}")
                
                if not self.check_game_over():
                    self.root.after(100, self.check_turn_and_trigger_ai)
            else:
                self.selected_square = None
                self.draw_board()

    def check_turn_and_trigger_ai(self):
        if self.board.is_game_over():
            return
            
        current_turn_white = (self.board.turn == chess.WHITE)
        
        if self.mode == "ai":
            active_mcts = self.mcts_white if current_turn_white else self.mcts_black
            color_label = "White AI" if current_turn_white else "Black AI"
            self.execute_background_search(active_mcts, color_label)
            
        elif self.mode == "random" and current_turn_white:
            self.status_label.config(text="Random baseline agent generating selection...")
            self.root.after(500, self.execute_random_move)
            
        elif self.mode == "human" and not current_turn_white:
            self.execute_background_search(self.mcts_black, "Transformer Engine (Black)")

    def execute_background_search(self, mcts_instance, label_string):
        self.status_label.config(text=f"Thinking... [{label_string}] processing search tree...")
        
        def search_thread_worker():
            cloned_state = self.board.copy()
            mcts_probs = mcts_instance.search(cloned_state)
            best_action = mcts_probs.argmax()
            ai_move = self.game._action_to_move(best_action, self.board)
            
            if ai_move not in self.board.legal_moves:
                ai_move = list(self.board.legal_moves)[0]
                
            self.root.after(0, self.apply_ai_move, ai_move, label_string)
            
        threading.Thread(target=search_thread_worker, daemon=True).start()

    def apply_ai_move(self, move, label_string):
        self.board.push(move)
        self.draw_board()
        self.update_status_string(f"{label_string} deployed: {move.uci()}")
        
        if not self.check_game_over():
            self.root.after(100, self.check_turn_and_trigger_ai)

    def execute_random_move(self):
        available_actions = self.game.get_legal_actions(self.board)
        random_action = random.choice(available_actions)
        move = self.game._action_to_move(random_action, self.board)
        self.board.push(move)
        self.draw_board()
        self.update_status_string(f"Random Baseline deployed: {move.uci()}")
        
        if not self.check_game_over():
            self.root.after(100, self.check_turn_and_trigger_ai)

    def update_status_string(self, action_taken_text):
        next_player = "White" if self.board.turn == chess.WHITE else "Black"
        self.status_label.config(text=f"{action_taken_text} | Next up: {next_player}")

    def check_game_over(self):
        if self.board.is_game_over():
            result = self.board.result()
            self.status_label.config(text=f"TERMINAL STATE REACHED. Result: {result}")
            messagebox.showinfo("Game Concluded", f"The match ended with outcome metrics: {result}")
            return True
        return False

def get_latest_checkpoint(prefix="chess_model_iter_", suffix=".pth"):
    """Scans the local directory to automatically detect the highest iteration checkpoint."""
    pattern = f"{prefix}*{suffix}"
    files = glob.glob(pattern)
    if not files:
        return None
    
    discovered_iterations = []
    for f in files:
        # Isolate and extract numeric characters explicitly containing the iteration indices
        match = re.search(r'(\d+)', os.path.basename(f))
        if match:
            discovered_iterations.append((int(match.group(1)), f))
            
    if not discovered_iterations:
        return None
        
    # Sort logically by the raw iteration integer value and return the absolute path of the highest index
    discovered_iterations.sort(key=lambda x: x[0])
    return discovered_iterations[-1][1]

def load_model(model_path, game):
    """Initializes and builds weights exclusively for the 9M standard Medium architecture."""
    model = ChessTransformer(
        vocab_size=13,            
        max_seq_len=64,           
        num_actions=game.action_size, 
        num_meta_features=6,      
        embed_dim=256,            
        num_heads=8,              
        num_blocks=10             
    ).to(game.device)

    if os.path.exists(model_path):
        print(f"Acquiring trained weights file from checkpoint path: {model_path}...")
        checkpoint = torch.load(model_path, map_location=game.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Weights verification complete. Sourced iteration metadata: {checkpoint.get('iteration', 'Unknown')}")
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        print(f"Warning Configuration Vector Check: '{model_path}' unreachable! Model initialized with untuned weights.")
        model.eval()

    return model

def main():
    game = ChessGame()
    
    # Run structural query detection to grab the latest training state automatically
    detected_checkpoint = get_latest_checkpoint()
    if detected_checkpoint:
        print(f"--> Auto-Detected Latest Directory Checkpoint: {detected_checkpoint}")
        default_model_path = detected_checkpoint
    else:
        # No training runs are located locally
        print(f"--> No active local checkpoints detected.")


    parser = argparse.ArgumentParser(description="AlphaZero Chess System Assessment Toolkit")
    parser.add_argument("--mode", type=str, choices=["human", "random", "ai"], default="human", 
                        help="Match Architecture Strategy: 'human' (vs AI), 'random' (vs AI), or 'ai' (AI vs AI)")
    parser.add_argument("--model_white", type=str, default=default_model_path, 
                        help="Checkpoint targeted for White evaluation paths (Defaults to latest detected)")
    parser.add_argument("--model_black", type=str, default=default_model_path, 
                        help="Checkpoint targeted for Black evaluation paths (Defaults to latest detected)")
    parser.add_argument("--sims", type=int, default=200, 
                        help="Baseline computation limit for individual MCTS evaluations")
    args = parser.parse_args()

    mcts_white, mcts_black = None, None
    print(f"System Initializing execution environment utilizing target device: {game.device}")

    # Build evaluation engines matching runtime options definitions
    if args.mode == "ai":
        print("\n--- Constructing White Processing Pipeline ---")
        model_white = load_model(args.model_white, game)
        mcts_white = MCTS(model_white, game, num_simulations=args.sims, self_play=False)
        
        print("\n--- Constructing Black Processing Pipeline ---")
        model_black = load_model(args.model_black, game)
        mcts_black = MCTS(model_black, game, num_simulations=args.sims, self_play=False)
    else:
        print("\n--- Constructing Adversarial Engine (Black Position) ---")
        model_black = load_model(args.model_black, game)
        mcts_black = MCTS(model_black, game, num_simulations=args.sims, self_play=False)

    # Launch Native Tkinter Graphics Application Environment Container
    root = tk.Tk()
    app = ChessGUI(root, game, args.mode, mcts_white, mcts_black, args.sims)
    root.mainloop()

if __name__ == "__main__":
    main()
import os
import argparse
import random
import threading
import torch
import chess
import tkinter as tk
from tkinter import messagebox, ttk
import tkinter.scrolledtext as scrolledtext
import glob
import re

from ChessGame import ChessGame
from MCTS import MCTS

from CNN_Player import ChessResNet
from Transformer_Player import ChessTransformer

# Hardware compatibility optimization
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Unicode character dictionary mapping for clean vector rendering of pieces
UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
    None: ''
}

class ChessGUI:
    def __init__(self, root, game, mode, mcts_white=None, mcts_black=None, 
                 white_label="White AI", black_label="Black AI", sims=200):
        self.root = root
        self.game = game
        self.mode = mode
        self.mcts_white = mcts_white
        self.mcts_black = mcts_black
        self.white_label = white_label
        self.black_label = black_label
        self.sims = sims
        
        self.board = game.get_initial_state()
        self.selected_square = None
        self.square_size = 80  # Default starting size
        
        self.root.title(f"AlphaZero Chess Engine — Mode: {mode.upper()}")
        self.root.configure(bg="#2C2F33")
        
        # Enable Resizing and set a minimum window size to prevent UI crushing
        self.root.resizable(True, True)
        self.root.minsize(750, 550)

        # Style Configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2C2F33")
        style.configure("TLabel", background="#2C2F33", foreground="#FFFFFF", font=("Segoe UI", 11))
        
        # Main Layout Container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # --- Left Side: Chess Board ---
        self.board_frame = ttk.Frame(self.main_frame)
        self.board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self.canvas = tk.Canvas(
            self.board_frame, 
            width=self.square_size * 8, 
            height=self.square_size * 8,
            highlightthickness=0,
            bd=2,
            relief=tk.RAISED
        )
        self.canvas.pack(expand=True, anchor=tk.CENTER)
        self.canvas.bind("<Button-1>", self.on_square_clicked)
        self.board_frame.bind("<Configure>", self.on_resize)

        # --- Right Side: Sidebar & Controls ---
        self.sidebar = ttk.Frame(self.main_frame, width=300)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False) 

        # Status Display
        self.status_label = tk.Label(
            self.sidebar, 
            text="Game Start!\nWhite's turn.", 
            font=("Segoe UI", 13, "bold"), 
            bg="#23272A", 
            fg="#7289DA",
            relief=tk.FLAT, 
            anchor=tk.CENTER,
            justify=tk.CENTER,
            pady=10,
            wraplength=280
        )
        self.status_label.pack(fill=tk.X, pady=(0, 15))

        # Progress Bar (for AI thinking)
        self.progress = ttk.Progressbar(self.sidebar, mode='indeterminate', length=280)
        self.progress.pack(fill=tk.X, pady=(0, 15))

        # Move History Box
        history_label = ttk.Label(self.sidebar, text="Move History", font=("Segoe UI", 12, "bold"))
        history_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.history_box = scrolledtext.ScrolledText(
            self.sidebar, 
            wrap=tk.WORD, 
            width=30, 
            height=15, 
            font=("Consolas", 11),
            bg="#23272A",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=10
        )
        self.history_box.pack(fill=tk.BOTH, expand=True)
        self.history_box.config(state=tk.DISABLED)

        # Color Palette
        self.colors = {
            "light": "#F0D9B5",        # Lichess light wood
            "dark": "#B58863",         # Lichess dark wood
            "highlight": "#CDD26A",    # Selection highlight
            "last_move": "#AAA23A",    # Previous move highlight
            "dot": "#1A1A1A"           # Valid move indicator
        }
        
        self.draw_board()
        self.check_turn_and_trigger_ai()

    def on_resize(self, event):
        min_dimension = min(event.width, event.height)
        new_square_size = int(min_dimension / 8)
        
        if abs(new_square_size - self.square_size) > 2 and new_square_size > 15:
            self.square_size = new_square_size
            self.canvas.config(width=self.square_size * 8, height=self.square_size * 8)
            self.draw_board()

    def add_to_history(self, san_move):
        self.history_box.config(state=tk.NORMAL)
        move_number = self.board.fullmove_number
        if self.board.turn == chess.BLACK:
            self.history_box.insert(tk.END, f"{move_number}. {san_move} ")
        else:
            self.history_box.insert(tk.END, f"{san_move}\n")
            
        self.history_box.see(tk.END)
        self.history_box.config(state=tk.DISABLED)

    def draw_board(self):
        self.canvas.delete("all")
        last_move = self.board.peek() if self.board.move_stack else None
        font_size = max(12, int(self.square_size * 0.45))
        
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                x1, y1 = col * self.square_size, row * self.square_size
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                
                color = self.colors["light"] if (row + col) % 2 == 0 else self.colors["dark"]
                if last_move and square in (last_move.from_square, last_move.to_square):
                    color = self.colors["last_move"]
                if self.selected_square == square:
                    color = self.colors["highlight"]
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                piece = self.board.piece_at(square)
                if piece:
                    symbol = UNICODE_PIECES.get(piece.symbol(), '')
                    text_color = "#FFFFFF" if piece.color == chess.WHITE else "#000000"
                    self.canvas.create_text(
                        x1 + self.square_size // 2, 
                        y1 + self.square_size // 2, 
                        text=symbol, 
                        font=("Arial", font_size), 
                        fill=text_color
                    )

        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    to_col = chess.square_file(move.to_square)
                    to_row = 7 - chess.square_rank(move.to_square)
                    
                    cx = to_col * self.square_size + self.square_size // 2
                    cy = to_row * self.square_size + self.square_size // 2
                    radius = max(3, self.square_size // 7)
                    
                    self.canvas.create_oval(
                        cx - radius, cy - radius, cx + radius, cy + radius,
                        fill=self.colors["dot"], stipple="gray50", outline=""
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
            if clicked_square == self.selected_square:
                self.selected_square = None
                self.draw_board()
                return
                
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.draw_board()
                return

            move = chess.Move(self.selected_square, clicked_square)
            
            if self.board.piece_at(self.selected_square) and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) in [0, 7]:
                    move.promotion = chess.QUEEN
            
            if move in self.board.legal_moves:
                san_move = self.board.san(move)
                self.board.push(move)
                self.add_to_history(san_move)
                self.selected_square = None
                self.draw_board()
                self.update_status_string(f"Human played: {san_move}")
                
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
            color_label = self.white_label if current_turn_white else self.black_label
            self.execute_background_search(active_mcts, color_label)
            
        elif self.mode == "random" and current_turn_white:
            self.status_label.config(text="Random baseline agent\ngenerating selection...")
            self.progress.start(10)
            self.root.after(500, self.execute_random_move)
            
        elif self.mode == "human" and not current_turn_white:
            self.execute_background_search(self.mcts_black, self.black_label)

    def execute_background_search(self, mcts_instance, label_string):
        self.status_label.config(text=f"{label_string}\nProcessing search tree...")
        self.progress.start(15)
        
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
        self.progress.stop()
        san_move = self.board.san(move)
        self.board.push(move)
        self.add_to_history(san_move)
        self.draw_board()
        self.update_status_string(f"{label_string} deployed: {san_move}")
        
        if not self.check_game_over():
            self.root.after(100, self.check_turn_and_trigger_ai)

    def execute_random_move(self):
        self.progress.stop()
        available_actions = self.game.get_legal_actions(self.board)
        random_action = random.choice(available_actions)
        move = self.game._action_to_move(random_action, self.board)
        
        san_move = self.board.san(move)
        self.board.push(move)
        self.add_to_history(san_move)
        self.draw_board()
        self.update_status_string(f"Random Baseline deployed:\n{san_move}")
        
        if not self.check_game_over():
            self.root.after(100, self.check_turn_and_trigger_ai)

    def update_status_string(self, action_taken_text):
        next_player = "White" if self.board.turn == chess.WHITE else "Black"
        self.status_label.config(text=f"{action_taken_text}\nNext up: {next_player}")

    def check_game_over(self):
        if self.board.is_game_over():
            result = self.board.result()
            self.status_label.config(text=f"TERMINAL STATE REACHED\nResult: {result}")
            self.progress.stop()
            messagebox.showinfo("Game Concluded", f"The match ended with outcome metrics: {result}")
            return True
        return False

def get_latest_checkpoint(arch=None):
    """
    Searches for the newest iteration checkpoint.
    If arch is provided ('cnn' or 'transformer'), targets checkpoints matching that architecture.
    """
    prefixes = []
    if arch:
        prefixes = [f"{arch}_chess_model_iter_", f"curri_{arch}_chess_model_iter_"]
    else:
        # Check both modern and legacy prefixes
        prefixes = [
            "cnn_chess_model_iter_", "curri_cnn_chess_model_iter_",
            "transformer_chess_model_iter_", "curri_transformer_chess_model_iter_",
            "chess_model_iter_", "curri_chess_model_iter_"
        ]

    candidates = []
    for prefix in prefixes:
        files = glob.glob(f"{prefix}*.pth")
        for f in files:
            match = re.search(r'iter_(\d+)', os.path.basename(f))
            if match:
                candidates.append((int(match.group(1)), f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def inspect_arch_from_checkpoint(state_dict):
    """Inspects parameter tensor names to automatically infer the architecture."""
    for key in state_dict.keys():
        if "res_blocks" in key or "initial_conv" in key:
            return "cnn"
        if "t_blocks" in key or "token_embedding" in key or "meta_projection" in key:
            return "transformer"
    return None

def instantiate_model(arch, game):
    """Helper to instantiate the appropriate architecture with standard hyperparams."""
    if arch == "cnn":
        return ChessResNet(
            vocab_size=13, 
            num_actions=game.action_size, 
            num_meta_features=6, 
            num_channels=128, 
            num_blocks=8, 
            piece_embed_dim=32
        ).to(game.device)
    else:
        return ChessTransformer(
            vocab_size=13,            
            max_seq_len=64,           
            num_actions=game.action_size, 
            num_meta_features=6,      
            embed_dim=256,            
            num_heads=8,              
            num_blocks=10             
        ).to(game.device)

def load_model(model_path, game, preferred_arch=None):
    """
    Loads weights into a neural network.
    Automatically detects architecture if a checkpoint exists, or falls back to preferred_arch.
    """
    checkpoint = None
    state_dict = None
    detected_arch = None

    if model_path and os.path.exists(model_path):
        print(f"Acquiring trained weights file from: {model_path}...")
        checkpoint = torch.load(model_path, map_location=game.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded iteration metadata: {checkpoint.get('iteration', 'Unknown')}")
        else:
            state_dict = checkpoint
            
        detected_arch = inspect_arch_from_checkpoint(state_dict)

    final_arch = detected_arch or preferred_arch or "cnn"
    print(f"Active architecture selected: [{final_arch.upper()}]")
    
    model = instantiate_model(final_arch, game)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print("Weights verification complete.")
    else:
        print(f"Warning: Checkpoint '{model_path}' not found or none specified! Model running with untuned weights.")

    model.eval()
    return model, final_arch

def main():
    game = ChessGame()

    parser = argparse.ArgumentParser(description="AlphaZero Chess System Assessment Toolkit")
    parser.add_argument("--mode", type=str, choices=["human", "random", "ai"], default="human", 
                        help="Match mode: 'human' (vs AI), 'random' (vs AI), or 'ai' (AI vs AI)")
    parser.add_argument("--arch", type=str, choices=["cnn", "transformer"], default=None,
                        help="Global architecture default if not specified individually")
    parser.add_argument("--arch_white", type=str, choices=["cnn", "transformer"], default=None,
                        help="Architecture for White AI (overrides --arch)")
    parser.add_argument("--arch_black", type=str, choices=["cnn", "transformer"], default=None,
                        help="Architecture for Black AI (overrides --arch)")
    parser.add_argument("--model_white", type=str, default=None, 
                        help="Checkpoint path for White AI (auto-detected if omitted)")
    parser.add_argument("--model_black", type=str, default=None, 
                        help="Checkpoint path for Black AI (auto-detected if omitted)")
    parser.add_argument("--sims", type=int, default=200, 
                        help="Baseline computation limit for individual MCTS evaluations")
    args = parser.parse_args()

    arch_white = args.arch_white or args.arch or "cnn"
    arch_black = args.arch_black or args.arch or "cnn"

    path_white = args.model_white or get_latest_checkpoint(arch_white)
    path_black = args.model_black or get_latest_checkpoint(arch_black)

    mcts_white, mcts_black = None, None
    white_label, black_label = "White AI", "Black AI"
    print(f"System Initializing execution environment utilizing target device: {game.device}")

    if args.mode == "ai":
        print("\n--- Constructing White Processing Pipeline ---")
        model_white, real_arch_white = load_model(path_white, game, preferred_arch=arch_white)
        mcts_white = MCTS(model_white, game, num_simulations=args.sims, self_play=False)
        white_label = f"{real_arch_white.upper()} (White)"
        
        print("\n--- Constructing Black Processing Pipeline ---")
        model_black, real_arch_black = load_model(path_black, game, preferred_arch=arch_black)
        mcts_black = MCTS(model_black, game, num_simulations=args.sims, self_play=False)
        black_label = f"{real_arch_black.upper()} (Black)"
    else:
        print("\n--- Constructing Adversarial Engine (Black Position) ---")
        model_black, real_arch_black = load_model(path_black, game, preferred_arch=arch_black)
        mcts_black = MCTS(model_black, game, num_simulations=args.sims, self_play=False)
        black_label = f"{real_arch_black.upper()} (Black)"

    root = tk.Tk()
    start_width, start_height = 950, 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (start_width // 2)
    y = (screen_height // 2) - (start_height // 2)
    root.geometry(f'{start_width}x{start_height}+{x}+{y}')
    
    app = ChessGUI(
        root=root, 
        game=game, 
        mode=args.mode, 
        mcts_white=mcts_white, 
        mcts_black=mcts_black, 
        white_label=white_label,
        black_label=black_label,
        sims=args.sims
    )
    root.mainloop()

if __name__ == "__main__":
    main()
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
from ChessPlayer import ChessTransformer
from MCTS import MCTS

# Hardware compatibility optimization
#os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0" # For my Ryzen AI 7 350 laptop

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Unicode character dictionary mapping for clean vector rendering of pieces
UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
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
        self.square_size = 80  # Default starting size
        
        self.root.title(f"Alpha-Transformer Engine — Mode: {mode.upper()}")
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
        # The board frame expands to fill available space
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
        # Use anchor=tk.CENTER so the board stays perfectly centered in the frame
        self.canvas.pack(expand=True, anchor=tk.CENTER)
        self.canvas.bind("<Button-1>", self.on_square_clicked)
        
        # Bind the resize event to calculate new board size dynamically
        self.board_frame.bind("<Configure>", self.on_resize)

        # --- Right Side: Sidebar & Controls ---
        # Fixed width for the sidebar so it doesn't get distorted
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
            wraplength=280  # <-- Forces text to wrap cleanly instead of clipping
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
        
        # Initialize Visual State
        self.draw_board()
        
        # Automatically kickstart the cycle if the AI handles White
        self.check_turn_and_trigger_ai()

    def on_resize(self, event):
        """Handles dynamic scaling of the chess board based on window size."""
        # Calculate the maximum square size that fits within the current frame (keeping it a perfect square)
        min_dimension = min(event.width, event.height)
        new_square_size = int(min_dimension / 8)
        
        # Prevent micro-stutters by enforcing a threshold change (e.g., +/- 2 pixels) before redrawing
        if abs(new_square_size - self.square_size) > 2 and new_square_size > 15:
            self.square_size = new_square_size
            self.canvas.config(width=self.square_size * 8, height=self.square_size * 8)
            self.draw_board()

    def add_to_history(self, san_move):
        """Adds a move to the SAN history sidebar."""
        self.history_box.config(state=tk.NORMAL)
        
        move_number = self.board.fullmove_number
        if self.board.turn == chess.BLACK: # White just moved
            self.history_box.insert(tk.END, f"{move_number}. {san_move} ")
        else: # Black just moved
            self.history_box.insert(tk.END, f"{san_move}\n")
            
        self.history_box.see(tk.END)
        self.history_box.config(state=tk.DISABLED)

    def draw_board(self):
        self.canvas.delete("all")
        
        last_move = self.board.peek() if self.board.move_stack else None
        
        # Dynamically scale font based on current square size
        font_size = max(12, int(self.square_size * 0.45))
        
        # 1. Draw Squares
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                x1, y1 = col * self.square_size, row * self.square_size
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                
                # Base Color
                color = self.colors["light"] if (row + col) % 2 == 0 else self.colors["dark"]
                
                # Highlights
                if last_move and square in (last_move.from_square, last_move.to_square):
                    color = self.colors["last_move"]
                if self.selected_square == square:
                    color = self.colors["highlight"]
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Draw Pieces
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

        # 2. Draw Valid Move Indicators for Selected Piece
        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    to_col = chess.square_file(move.to_square)
                    to_row = 7 - chess.square_rank(move.to_square)
                    
                    cx = to_col * self.square_size + self.square_size // 2
                    cy = to_row * self.square_size + self.square_size // 2
                    
                    # Dynamically scale the legal move indicator dot
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
            color_label = "White AI" if current_turn_white else "Black AI"
            self.execute_background_search(active_mcts, color_label)
            
        elif self.mode == "random" and current_turn_white:
            self.status_label.config(text="Random baseline agent\ngenerating selection...")
            self.progress.start(10)
            self.root.after(500, self.execute_random_move)
            
        elif self.mode == "human" and not current_turn_white:
            self.execute_background_search(self.mcts_black, "Transformer (Black)")

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

def get_latest_checkpoint(prefix="chess_model_iter_", suffix=".pth"):
    pattern = f"{prefix}*{suffix}"
    files = glob.glob(pattern)
    
    if not files:
        prefix = "curri_chess_model_iter_"
        files = glob.glob(pattern)
        if not files:
            return None
    
    discovered_iterations = []
    for f in files:
        match = re.search(r'(\d+)', os.path.basename(f))
        if match:
            discovered_iterations.append((int(match.group(1)), f))
            
    if not discovered_iterations:
        return None
        
    discovered_iterations.sort(key=lambda x: x[0])
    return discovered_iterations[-1][1]

def load_model(model_path, game):
    model = ChessTransformer(
        vocab_size=13,            
        max_seq_len=64,           
        num_actions=game.action_size, 
        num_meta_features=6,      
        embed_dim=256,            
        num_heads=8,              
        num_blocks=10             
    ).to(game.device)

    if model_path and os.path.exists(model_path):
        print(f"Acquiring trained weights file from checkpoint path: {model_path}...")
        checkpoint = torch.load(model_path, map_location=game.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Weights verification complete. Sourced iteration metadata: {checkpoint.get('iteration', 'Unknown')}")
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        print(f"Warning: Configuration Vector Check: '{model_path}' unreachable! Model initialized with untuned weights.")
        model.eval()

    return model

def main():
    game = ChessGame()
    
    detected_checkpoint = get_latest_checkpoint()
    if detected_checkpoint:
        print(f"--> Auto-Detected Latest Directory Checkpoint: {detected_checkpoint}")
        default_model_path = detected_checkpoint
    else:
        print(f"--> No active local checkpoints detected.")
        default_model_path = None


    parser = argparse.ArgumentParser(description="AlphaZero Chess System Assessment Toolkit")
    parser.add_argument("--mode", type=str, choices=["human", "random", "ai"], default="human", 
                        help="Match Architecture Strategy: 'human' (vs AI), 'random' (vs AI), or 'ai' (AI vs AI)")
    parser.add_argument("--model_white", type=str, default=default_model_path, 
                        help="Checkpoint targeted for White evaluation paths")
    parser.add_argument("--model_black", type=str, default=default_model_path, 
                        help="Checkpoint targeted for Black evaluation paths")
    parser.add_argument("--sims", type=int, default=200, 
                        help="Baseline computation limit for individual MCTS evaluations")
    args = parser.parse_args()

    mcts_white, mcts_black = None, None
    print(f"System Initializing execution environment utilizing target device: {game.device}")

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

    root = tk.Tk()
    
    # Establish a clean default window geometry based on screen size
    start_width, start_height = 950, 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (start_width // 2)
    y = (screen_height // 2) - (start_height // 2)
    root.geometry(f'{start_width}x{start_height}+{x}+{y}')
    
    app = ChessGUI(root, game, args.mode, mcts_white, mcts_black, args.sims)
    root.mainloop()

if __name__ == "__main__":
    main()

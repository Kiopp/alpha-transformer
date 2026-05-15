import chess
import numpy as np
import torch

class ChessGame:
    def __init__(self):
        # 64 squares * 64 squares = 4096 possible moves.
        self.action_size = 4096 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_initial_state(self):
        return chess.Board()

    def clone_state(self, state):
        return state.copy()

    def _move_to_action(self, move):
        """Converts a python-chess Move object to an integer index."""
        return move.from_square * 64 + move.to_square

    def _action_to_move(self, action, board):
        """Converts an integer index back to a python-chess Move object."""
        from_sq = action // 64
        to_sq = action % 64
        move = chess.Move(from_sq, to_sq)
        
        # Handle pawn promotions (auto-promote to Queen for simplicity)
        if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
            if chess.square_rank(to_sq) == 0 or chess.square_rank(to_sq) == 7:
                move.promotion = chess.QUEEN
                
        return move

    def get_next_state(self, state, action):
        """Applies an action to the board and returns the new board state."""
        board = state.copy()
        move = self._action_to_move(action, board)
        board.push(move)
        return board

    def get_legal_actions(self, state):
        """Returns a list of integer indices representing legal moves."""
        return list(set([self._move_to_action(move) for move in state.legal_moves]))

    def get_reward_and_terminal(self, state):
        """
        Returns (is_terminal, absolute_reward)
        Reward: +1.0 for White win, -1.0 for Black win, 0.0 for draw.
        """
        if state.is_game_over():
            result = state.result()
            if result == '1/2-1/2':
                return True, 0.0  
            elif result == '1-0':
                return True, 1.0  # White won
            else:
                return True, -1.0 # Black won
        
        # Aggressive Draw Claiming
        if state.can_claim_draw():
            return True, 0.0
        '''
        # Stop the drunken monkey!
        if state.fullmove_number > 100:
            # Instead of a draw, award the win to whoever has more material!
            # Standard chess piece values: P=1, N=3, B=3, R=5, Q=9
            white_material = len(state.pieces(chess.PAWN, chess.WHITE)) * 1 + \
                             len(state.pieces(chess.KNIGHT, chess.WHITE)) * 3 + \
                             len(state.pieces(chess.BISHOP, chess.WHITE)) * 3 + \
                             len(state.pieces(chess.ROOK, chess.WHITE)) * 5 + \
                             len(state.pieces(chess.QUEEN, chess.WHITE)) * 9
                             
            black_material = len(state.pieces(chess.PAWN, chess.BLACK)) * 1 + \
                             len(state.pieces(chess.KNIGHT, chess.BLACK)) * 3 + \
                             len(state.pieces(chess.BISHOP, chess.BLACK)) * 3 + \
                             len(state.pieces(chess.ROOK, chess.BLACK)) * 5 + \
                             len(state.pieces(chess.QUEEN, chess.BLACK)) * 9

            if white_material > black_material:
                return True, 1.0  # White wins by decision
            elif black_material > white_material:
                return True, -1.0 # Black wins by decision
            else:
                return True, 0.0  # True draw
        '''
        # Over 200 moves in a single game is a draw
        if state.fullmove_number > 200:
            return True, 0.0

        return False, 0.0

    def prepare_inputs(self, state):
        """
        Converts the python-chess board into tensors for the ChessTransformer.
        Adds the batch dimension (shape: 1, ...) so PyTorch can process it.
        """
        # 1. Board Tensor (64 squares)
        piece_map = {
            None: 0,
            chess.Piece(chess.PAWN, chess.WHITE): 1,
            chess.Piece(chess.KNIGHT, chess.WHITE): 2,
            chess.Piece(chess.BISHOP, chess.WHITE): 3,
            chess.Piece(chess.ROOK, chess.WHITE): 4,
            chess.Piece(chess.QUEEN, chess.WHITE): 5,
            chess.Piece(chess.KING, chess.WHITE): 6,
            chess.Piece(chess.PAWN, chess.BLACK): 7,
            chess.Piece(chess.KNIGHT, chess.BLACK): 8,
            chess.Piece(chess.BISHOP, chess.BLACK): 9,
            chess.Piece(chess.ROOK, chess.BLACK): 10,
            chess.Piece(chess.QUEEN, chess.BLACK): 11,
            chess.Piece(chess.KING, chess.BLACK): 12,
        }
        
        board_array = np.zeros(64, dtype=np.int64)
        for sq in chess.SQUARES:
            piece = state.piece_at(sq)
            board_array[sq] = piece_map.get(piece, 0)
            
        # Add batch dimension: shape becomes (1, 64)
        board_tensor = torch.tensor(board_array, dtype=torch.long, device=self.device).unsqueeze(0)

        # 2. Meta Features Tensor (The "Invisible Rules")
        # Vector size: 6
        meta_array = np.zeros(6, dtype=np.float32)
        meta_array[0] = 1.0 if state.turn == chess.WHITE else 0.0
        meta_array[1] = 1.0 if state.has_kingside_castling_rights(chess.WHITE) else 0.0
        meta_array[2] = 1.0 if state.has_queenside_castling_rights(chess.WHITE) else 0.0
        meta_array[3] = 1.0 if state.has_kingside_castling_rights(chess.BLACK) else 0.0
        meta_array[4] = 1.0 if state.has_queenside_castling_rights(chess.BLACK) else 0.0
        meta_array[5] = state.halfmove_clock / 100.0 # Normalize the 50-move rule counter
        
        # Add batch dimension: shape becomes (1, 6)
        meta_tensor = torch.tensor(meta_array, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 3. Legal Moves Mask Tensor
        legal_mask_array = np.zeros(self.action_size, dtype=bool)
        for action in self.get_legal_actions(state):
            legal_mask_array[action] = True
            
        # Add batch dimension: shape becomes (1, 4096)
        legal_mask = torch.tensor(legal_mask_array, dtype=torch.bool, device=self.device).unsqueeze(0)

        return board_tensor, meta_tensor, legal_mask
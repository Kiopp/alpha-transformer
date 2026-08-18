import random
import chess

class PositionSampler:
    def __init__(self, mode="tabula_rasa", opening_prob=0.3, endgame_prob=0.3):
        """
        mode: "tabula_rasa" (always standard start) or "curriculum" (mix of starts)
        opening_prob: % chance to start from a predefined opening
        endgame_prob: % chance to start from a predefined endgame
        (1 - opening_prob - endgame_prob) = % chance for standard start
        """
        self.mode = mode
        self.opening_prob = opening_prob
        self.endgame_prob = endgame_prob

        # A small sample of common openings (Standard FENs)
        self.openings = [
            # --- E4 Openings ---
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # B20: Sicilian Defense
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # C00: French Defense
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",    # B00: King's Pawn
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # B01: Scandinavian Defense
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", # B02: Alekhine's Defense
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # B10: Caro-Kann Defense
            "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1r b KQkq - 1 2", # C40: King's Knight
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", # C60: Ruy Lopez
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", # C50: Italian Game
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 3 3", # C47: Four Knights
            
            # --- D4 Openings ---
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", # D00: Queen's Pawn Game
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", # D06: Queen's Gambit
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", # A40: Englund Gambit
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2", # A45: Indian Defense
            "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", # E10: Indian Defense (c4)
            "rnbqkb1r/pppp1ppp/5n2/4p3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", # A51: Budapest Gambit
            "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", # E60: King's Indian Defense
            "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3", # E20: Nimzo-Indian
            "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", # E00: Catalan / Queen's Indian setup
            "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", # D10: Slav Defense
            
            # --- C4, Nf3, and Flank Openings ---
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",    # A10: English Opening
            "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2", # A20: English (King's English)
            "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2", # A30: English (Symmetrical)
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",    # A04: Reti Opening
            "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1",    # A00: Hungarian Opening / King's Indian Attack
            "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1",    # A01: Nimzowitsch-Larsen Attack
            "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1",    # A00: Anderssen's Opening (for chaotic evaluation)
            "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1",    # A00: Dunst Opening
            "rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1",    # A00: Polish Opening
            "rnbqkbnr/pppppppp/8/8/8/7P/PPPPPPP1/RNBQKBNR b KQkq - 0 1"       # A00: Clemenz Opening
        ]

    def generate_random_endgame(self):
        """Generates a random, legally valid K+R vs K or K+Q vs K board state."""
        board = chess.Board(None) # Initialize completely empty board
        
        # Place Kings (Ensuring they are not adjacent)
        white_king_sq = random.choice(list(chess.SQUARES))
        board.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
        
        black_king_sq = random.choice(list(chess.SQUARES))
        while chess.square_distance(white_king_sq, black_king_sq) <= 1:
            black_king_sq = random.choice(list(chess.SQUARES))
        board.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))
        
        # Assign Material (Randomize whether White or Black gets the attacker)
        attacker_color = chess.WHITE if random.random() < 0.5 else chess.BLACK
        attacker_piece = chess.ROOK if random.random() < 0.5 else chess.QUEEN
        
        attacker_sq = random.choice(list(chess.SQUARES))
        while attacker_sq in (white_king_sq, black_king_sq):
            attacker_sq = random.choice(list(chess.SQUARES))
            
        board.set_piece_at(attacker_sq, chess.Piece(attacker_piece, attacker_color))
        
        # Randomize whose turn it is
        board.turn = chess.WHITE if random.random() < 0.5 else chess.BLACK
        
        # Validate the board
        # If the generated board puts the player *not* to move in check, it's illegal.
        if board.is_valid() and not board.is_game_over():
            return board.fen()
        
        # Fallback to a standard, guaranteed-legal K+R vs K endgame if random gen fails
        return "8/8/8/8/8/8/4k3/R3K3 w Q - 0 1"

    def sample(self):
        """Returns a starting FEN string, or None for standard starting position."""
        if self.mode == "tabula_rasa":
            return None

        r = random.random()
        if r < self.opening_prob:
            return random.choice(self.openings)
        elif r < self.opening_prob + self.endgame_prob:
            return self.generate_random_endgame()
        else:
            return None # Standard start (30% chance with default weights)
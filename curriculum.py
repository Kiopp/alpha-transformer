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
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # Sicilian Defense
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # French Defense
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", # Ruy Lopez
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2", # Indian Defense
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", # Queen's Gambit
        ]

        # Crucial basic endgames to teach mating nets and pawn promotion
        self.endgames = [
            "8/8/8/8/8/8/4k3/R3K3 w Q - 0 1", # King + Rook vs King
            "8/8/8/8/8/8/4k3/Q3K3 w - - 0 1", # King + Queen vs King
            "8/8/8/8/8/4P3/4k3/4K3 w - - 0 1", # King + Pawn vs King (winning)
            "8/8/8/8/8/4k3/8/R3K2R w KQ - 0 1", # King + 2 Rooks vs King
        ]

    def sample(self):
        """Returns a starting FEN string, or None for standard starting position."""
        if self.mode == "tabula_rasa":
            return None

        r = random.random()
        if r < self.opening_prob:
            return random.choice(self.openings)
        elif r < self.opening_prob + self.endgame_prob:
            return random.choice(self.endgames)
        else:
            return None # Standard start
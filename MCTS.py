# Monte-carlo tree search
import math
import numpy as np
import torch

class Node:
    def __init__(self, prior_prob, parent=None, action_taken=None):
        self.parent = parent
        self.action_taken = action_taken # The move taken to get to thois position
        
        self.children = {}

        # AlphaZero statistics
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

    @property
    def q_value(self):
        # Q(s, a) = W / N (Average value of this node)
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    # Create child nodes for all legal moves
    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(prior_prob=prob, parent=self, action_taken=action)
    
    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, model, game, num_simulations=800, c_puct=1.0):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.c_puct = c_puct # Controls exploration vs exploitation

    def search(self, initial_state):
        root = Node(prior_prob=1.0)

        # Simulation loop
        for _ in range(self.num_simulations):
            node = root
            state = self.game.clone_state(initial_state) # Copy board

            # --- Selection ---
            # Traverse tree until leaf is reached
            while node.is_expanded():
                action, node = self._select_child(node)
                state = self.game.get_next_state(state, action) # Play selected move

            # Check game over
            is_gameover, value = self.game.get_reward_and_terminal(state)

            if not is_gameover:
                board_tensor, meta_tensor, legal_mask = self.game.prepare_inputs(state)

                with torch.no_grad():
                    self.model.eval()
                    policy_logits, value_tensor = self.model(board_tensor, meta_tensor, legal_mask)

                value = value_tensor.item()

                # Convert logits to probabilities using softmax
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

                # Filter legal actions to node
                legal_actions = self.game.get_legal_actions(state)
                action_probs = {a: policy_probs[a] for a in legal_actions}

                # Expand node
                node.expand(action_probs)

            # Backpropagate
            self._backpropagate(node, value)

        # After all simulations return probability distribution, based on how often MCTS visited them
        action_visits = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_visits[action] = child.visit_count
        
        return action_visits / np.sum(action_visits) # Normalize

    # Pick best child node using the PUCT formula
    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u_value = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value

            value *= -1 # Zero-sum alternating turn game -> invert at each step
            node = node.parent

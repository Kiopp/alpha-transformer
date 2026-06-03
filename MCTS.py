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
    def __init__(self, model, game, num_simulations=800, c_puct=1.0, self_play=True):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.c_puct = c_puct # Controls exploration vs exploitation
        self.self_play = self_play 

    def search(self, initial_state):
        root = Node(prior_prob=1.0)

        # Expand root to add noise
        board_tensor, meta_tensor, legal_mask = self.game.prepare_inputs(initial_state)
        with torch.no_grad():
            self.model.eval()
            policy_logits, _ = self.model(board_tensor, meta_tensor, legal_mask)
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

            legal_actions = self.game.get_legal_actions(initial_state)
            action_probs = {a: policy_probs[a] for a in legal_actions}
        root.expand(action_probs)

        if self.self_play:
            # Add Dirichlet noise to enable more exploration
            dirichlet_alpha = 0.3
            exploration_fraction = 0.25

            # Generate noise
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_actions))

            # Blend noise into root prior probabilities
            for i, action in enumerate(legal_actions):
                root.children[action].prior_prob = (
                    root.children[action].prior_prob * (1 - exploration_fraction)
                    + noise[i] * exploration_fraction
                )

        # Simulation loop
        for sim in range(self.num_simulations):
            node = root
            state = self.game.clone_state(initial_state)

            # Selection
            while node.is_expanded():
                action, node = self._select_child(node)
                state = self.game.get_next_state(state, action)
            
            is_terminal, value = self.game.get_reward_and_terminal(state)

            is_terminal, absolute_reward = self.game.get_reward_and_terminal(state)

            if is_terminal:
                # Calculate relative value for the player whose turn it currently is
                current_player_is_white = state.turn
                
                if absolute_reward == 0.0:
                    value = 0.0
                elif (absolute_reward == 1.0 and current_player_is_white) or (absolute_reward == -1.0 and not current_player_is_white):
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Evaluation and Expansion (NN outputs relative value)
                board_tensor, meta_tensor, legal_mask = self.game.prepare_inputs(state)
                with torch.no_grad():
                    self.model.eval()
                    policy_logits, value_tensor = self.model(board_tensor, meta_tensor, legal_mask)
                
                value = value_tensor.item()
                if state.is_repetition(2):
                    value = min(1.0, value + 0.25) # Penalize repetitions
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

                legal_actions = self.game.get_legal_actions(state)
                action_probs = {a: policy_probs[a] for a in legal_actions}
                node.expand(action_probs)

            # Backpropagate
            self._backpropagate(node, value)

            # Early stopping
            # If visited move is too far ahead of the others to catch up, CPU time will be saved
            if not self.self_play and (sim > 0 and sim % 10 == 0):
                visits = [child.visit_count for child in root.children.values()]
                visits.sort(reverse=True)
                if len(visits) >= 2:
                    best_visits = visits[0]
                    second_visits = visits[1]
                    remaining_visits = self.num_simulations - sim

                    if best_visits > second_visits + remaining_visits:
                        # Second best visits count cannot catch up
                        break

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
        value = -value 
        while node is not None:
            node.visit_count += 1
            node.value_sum += value

            value *= -1 # Zero-sum alternating turn game -> invert at each step
            node = node.parent

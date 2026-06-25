# alpha-transformer
An Alpha-Zero-Like model replacing the traditional CNN backbone with a pure Transformer architecture.

# Overview
This repository contains a chess engine trained via self-play reinforcement learning. Instead of using convolutional neural networks (CNNs), the model evaluates board states and policy probabilities using a sequence-to-sequence Transformer approach. The project documentation is provided in this README.md file.

# Architecture and Features
* Environment: The game logic, state management, and move validations are handled in ChessGame.py. It includes custom reward systems handling standard terminal states alongside aggressive draw claiming and material tiebreakers.

* Transformer-Based Neural Network: The core model, defined in ChessPlayer.py, relies on token embeddings, position embeddings, and multiple transformer blocks. It projects board states and meta-features into an embedding space.

* Dual-Head Output: The network in ChessPlayer.py features a policy head (predicting move probabilities across 4096 possible actions) and a value head (predicting win, loss, or draw).

* Monte-Carlo Tree Search (MCTS): MCTS.py implements a PUCT-based MCTS algorithm for action selection, utilizing Dirichlet noise for self-play exploration.

* Asynchronous GPU Training: The training loop in train.py utilizes multiprocessing, allowing CPU workers to simulate games in parallel while an inference server batches requests to the GPU for efficient hardware utilization.

# Usage

## Training
To train the model via self-play, execute train.py. The system handles replay buffers, dynamic batching, and saves intermediate network weights as checkpoints.

## Playing
You can evaluate the trained model using the Tkinter-based graphical user interface provided in play.py. This script will automatically detect the most recent model checkpoint to use for evaluation.

Run the script via command line to select your target mode:

human: Play as White against the Transformer model.

random: Watch a random baseline agent play against the Transformer model.

ai: Watch two instances of the model play against each other.

# License
This software is released under the MIT License. Please review the LICENSE file for more details. Copyright (c) 2026 Jesper Wentzell.
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

* Curriculum Learning: Curriculum.py contains a PositionSampler that injects predefined FEN strings (standard openings and endgames) to overcome tabula rasa data starvation.

* Option for tabula rasa or curriculum training using pre-defined Forsyth-Edwards Notation (FEN) strings

# Usage
> **Hardware Note:** The codebase currently includes environment variables optimized for AMD GPUs via ROCm (e.g., Ryzen AI processors). If you are using an NVIDIA GPU or running purely on CPU, you may safely remove or ignore the `TORCH_ROCM...` environment flags at the top of the scripts.

## Prerequisites
This project requires Python 3.8+ and the following dependencies:
* `torch` (PyTorch)
* `chess` (python-chess)
* `numpy`
* `tkinter` (Usually included with standard Python installations, required for the GUI)

You can install the required packages via pip:
```bash
pip install torch chess numpy
```

## Training
The system handles replay buffers, dynamic batching, and saves intermediate network weights(including optimizer and LR scheduler states) as checkpoints.
To start training using the predefined openings and endgames (Curriculum mode):
```bash
python train.py --curriculum curriculum
```

To start purely from scratch (Tabula Rasa):
```bash
python train.py --curriculum tabula_rasa
```

## Playing
You can evaluate the trained model using the Tkinter-based graphical user interface provided in play.py. This script will automatically detect the most recent model checkpoint to use for evaluation.

Run the script via command line to select your target mode:

* human: Play as White against the Transformer model.
```bash
python play.py --mode human
```
* random: Watch a random baseline agent play against the Transformer model.
```bash
python play.py --mode random
```
* ai: Watch two instances of the model play against each other.
```bash
python play.py --mode ai
```
* Advanced configuration: You can pit specific training checkpoints against each other and change the search depth.
```bash
python play.py --mode ai --sims 400 --model_white chess_model_iter_10.pth --model_black chess_model_iter_20.pth
```
# License
This software is released under the MIT License. Please review the LICENSE file for more details. Copyright (c) 2026 Jesper Wentzell.

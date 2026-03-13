**CTM ASR Experiments**

Experimental Automatic Speech Recognition (ASR) framework using Continuous Thought Machines (CTM) trained on gammatone spectrogram features with a speaker-split evaluation protocol.

This project explores how CTM architectures perform on word recognition tasks under varying configurations including:
•	Iteration depth (CTM ticks)
•	Memory length
•	Vocabulary size
•	Model dimension

The framework automatically runs large experiment grids and generates performance comparison plots.

**Features**

•	Continuous Thought Machine (CTM) architecture
•	Gammatone-based audio feature representation
•	Speaker-split dataset evaluation
•	Noise augmentation with configurable SNR
•	Automated hyperparameter experiments
•	Early stopping and LR scheduling
•	Automatic experiment comparison plots
•	Parallel experiment execution (CPU mode)
•	GPU-aware sequential training

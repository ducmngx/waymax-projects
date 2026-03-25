# Waymax Projects

Autonomous vehicle ML projects built on the Waymax simulator and Waymo Open Motion Dataset.

## Projects

### 1. Behavior Cloning (in progress)
Imitation learning baseline trained on expert trajectories. Includes closed-loop evaluation and failure analysis.

### 2. Prediction-Aware Planning (planned)
Extending the BC baseline with motion prediction for surrounding agents.

## Setup
```bash
conda create -n waymax python=3.11
conda activate waymax
pip install --upgrade "jax[cuda12]"
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
```

#!/usr/bin/env bash
set -euo pipefail

# Runs all algorithms on Zelda using CNNPolicy by default.
# Adjust total_steps in configs/algos/*.yaml if needed for shorter/longer runs.

ENV_CFG="configs/envs/zelda-v0.yaml"
declare -A ALGO_CFG=(
  ["PPO"]="configs/algos/ppo.yaml"
  ["A2C"]="configs/algos/a2c.yaml"
  ["REINFORCE"]="configs/algos/reinforce.yaml"
  ["DQN"]="configs/algos/dqn.yaml"
  ["TD3"]="configs/algos/td3.yaml"
  ["SAC"]="configs/algos/sac.yaml"
)

MODEL="CNNPolicy"

for ALG in "${!ALGO_CFG[@]}"; do
  echo "=== Running $ALG on Zelda ==="
  python main.py     --algorithm "$ALG"     --model "$MODEL"     --env_config "$ENV_CFG"     --algo_config "${ALGO_CFG[$ALG]}"
done

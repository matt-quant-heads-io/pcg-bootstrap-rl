# pcg_bootstrap_rl

Bootstrapped RL experiments for `pcg_benchmark` environments using a Gym-compatible wrapper.

## Quickstart

```bash
python -m pip install -r requirements.txt
python main.py   --algorithm PPO   --model CNNPolicy   --env_config configs/envs/zelda.yaml   --algo_config configs/algos/ppo.yaml
```

Artifacts land in `results/{game}/{algorithm}/{timestamp}/`.

Supported algorithms: **PPO, A2C, REINFORCE, DQN, TD3 (discrete), SAC (discrete)**.

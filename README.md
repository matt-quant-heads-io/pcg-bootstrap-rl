# pcg_bootstrap_rl

Bootstrapped RL experiments for `pcg_benchmark` environments using a Gym-compatible wrapper.

## Quickstart

```bash
python -m pip install -r requirements.txt
python main.py   --algorithm PPO   --model CNNPolicy   --env_config configs/envs/zelda.yaml   --algo_config configs/algos/ppo.yaml
```

Artifacts land in `results/{game}/{algorithm}/{timestamp}/`.

Supported algorithms: **PPO, A2C, REINFORCE, DQN, TD3 (discrete), SAC (discrete)**.


- Add checkpoint model saving to all algos
    - logging of map metrics
    - Add level outputs 
- Add logging of map metrics for each game both per checkpoint and aggregated across checkpoints (and accumulated average? to how means change over time with incoming new checkpoints)
- Add level outputs 


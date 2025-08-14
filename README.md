# PCG Bootstrap RL

Bootstrapping reinforcement learning for **procedural content generation (PCG)** built on top of the
[`pcg_benchmark`](https://github.com/amidos2006/pcg_benchmark/tree/main) environment (retrofitted to Gymnasium).
The goal of this research is to:
1) Understand the affects of RL on pre-training (and vice versa) as it pertains to solvability, diversity, and style. 
2) Compare **bootstrapping techniques** (supervised pretraining + RL fine-tuning) across
**on-policy** and **off-policy** algorithms, 
3) Produce clean, reproducible training/evaluation artifacts for a paper.

---

## Algorithms

We evaluate both on-policy and off-policy methods (discrete action spaces):

| Category     | Algorithms                               |
|--------------|-------------------------------------------|
| **On-policy** | `PPO`, `A2C`, `REINFORCE`                 |
| **Off-policy**| `DQN`, `TD3 (discrete)`, `SAC (discrete)` |

---

| Game        | Notes                                                                |
| ----------- | -------------------------------------------------------------------- |
| **Zelda**   |                                                                      |
| **Binary**  |                                                                      |
| **LR**      |                                                                      |
| **smb**     |                                                                      |
| **DD**      |                                                                      |


---

## Repo Layout

```
pcg_bootstrap_rl/
├─ main.py                         # entrypoint
├─ models/                         # PyTorch policies/Q-nets, etc.
├─ rl_agents/
│  ├─ rl_agent_base.py            # base class (preprocess, evaluate, safe env step, ckpt helpers)
│  ├─ ppo.py, a2c.py, reinforce.py
│  ├─ dqn.py, td3_discrete.py, sac_discrete.py
│  └─ supervised_learning_agent.py
├─ utils/
│  ├─ logging.py                  # JSONL logger
│  ├─ replay_buffer.py            # off-policy memory
│  ├─ plots.py                    # plotting utilities for metrics.jsonl
│  └─ config.py                   # run dir helpers, config loading
├─ configs/
│  ├─ env/                        # game configs (e.g., zelda.yaml)
│  └─ algo/                       # algorithm configs (ppo.yaml, a2c.yaml, ...)
├─ pcg_benchmark/                  # vendored or installed pcg_benchmark (GymPCGEnv, problems, rendering)
│  ├─ __init__.py
│  ├─ problems/
│  ├─ rendering/
│  └─ ...
├─ results/                        # training/evaluation outputs (auto-created)
│  ├─ <game>/<algo>/<timestamp>/...
├─ tests/
│  └─ run_dummy.sh                # lightweight smoke test (dummy script)
├─ run_all_zelda.sh               # run all algorithms for zelda
└─ README.md
```

---

## Installation & Setup

1) **Create & activate an environment**
```bash
conda create -n pcg python=3.11 -y
conda activate pcg
```
1B) **OR virtual environment**
```bash
python3 -m venv pcg_env
source pcg_env/bin/activate
```

2) **Install core deps**
```bash
pip install -r requirements.txt
pip install -e. # TODO: integrate requirements.txt into setup.py
```

---

## Running

### Main entrypoint

```bash
python main.py \
  --algorithm PPO \
  --model CNNPolicy \
  --env_config configs/env/zelda.yaml \
  --algo_config configs/algo/ppo.yaml
```

**Optional bootstrapping (pretraining):**
```bash
python main.py \
  --algorithm PPO \
  --model CNNPolicy \
  --env_config configs/env/zelda.yaml \
  --algo_config configs/algo/ppo.yaml \
  --pretrain supervised_learning_agent   # class name lowercased
```

**Arguments**
- `--algorithm` : one of `PPO`, `A2C`, `REINFORCE`, `DQN`, `TD3Discrete`, `SACDiscrete`
- `--model`     : a model key registered in `models/__init__.py` (e.g., `CNNPolicy`)
- `--env_config`: path to game config (e.g., `configs/env/zelda.yaml`)
- `--algo_config`: algorithm hyperparams (e.g., `configs/algo/ppo.yaml`)
- `--pretrain`  : (optional) supervised agent key (lowercased class name), run before RL

**Outputs**
- `results/<game>/<algo>/<timestamp>/`
  - `metrics.jsonl` — training logs
  - `levels/` — episode renders from `evaluate()`
  - `checkpoints/` — saved checkpoints at `checkpoint_interval`
  - `metrics/eval.json` — eval results keyed by checkpoint step
  - `metrics/plots/*.png` — plots vs checkpoint (auto-updated after eval)

---

## Quick Smoke Test (Zelda)

To run **all algorithms** on Zelda with the provided settings:

```bash
bash run_all_zelda.sh
```

This script:
- Calls `main.py` for each algorithm with its algo config and the Zelda env config.
- Produces per-algo result directories under `results/zelda-v0/<ALGO>/<TIMESTAMP>/`
- Saves checkpoints, evals, and plots (if intervals are reached).

If you just want to check plumbing quickly, there’s also a tiny dummy test:
```bash
bash tests/run_dummy.sh
```

---

## Next steps

### 1) Toolbuilding
- [ ] Get the plotting in the evaluate method working properly.
- [ ] Add map metrics (e.g. number of enemies, path length, etc. to collected results for each checkpoint).
- [ ] Finish the SuperviseLearningAgent.
- [ ] Implement pretraining hooks for each agent.
- [ ] Design the visualizations for analyzing the baselines experiments (and consequently the pretraining experiments).
- [ ] Make sure everything runs on the remaining games.

### 2) Experiments
- [ ] Run the baselines for each rl algorithm + supervised learning.
- [ ] **Naive bootstrapping**:
  - [ ] Run PoD for pretraining, take trained model, and for each rl algorithm load in the pretrained model and perform training & and evaluation. Track the map metrics (e.g. mean & standard error for path length, mean number of enemies, etc.), solvability, diversity, and level outputs.
  - [ ] Train a random agent for pretraining, and repeat the steps.
  - [ ] Compare the random pretraining to PoD pretraining across the algos (determine what the plots / visualizations should look like for this).
  - [ ] What are the effects on not just style but q & d both on RL side and on the pretraining side (i.e. how did RL change the pretrained model)?
- [ ] **Learning to Imitate PoD expert via KL divergence in loss term**:
- [ ] TBD

---

## References

- The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games https://arxiv.org/html/2503.21474v1

---

## License

Add your project’s license here.

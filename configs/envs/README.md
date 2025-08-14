# PCG Benchmark Configuration System

This directory contains YAML configuration files for different PCG problems. These files define parameters for problem initialization, making it easier to manage and reproduce experiments.

## Usage

### Loading a Configuration

```python
from pcg_benchmark.config import load_config, get_problem_params

# Load full configuration
config = load_config('zelda-v0')

# Get parameters for problem constructor
params = get_problem_params('zelda-v0')

# Use with pcg_benchmark
import pcg_benchmark
env = pcg_benchmark.make('zelda-v0')
```

### Available Configurations

Use the `list_available_configs()` function to see all available configurations:

```python
from pcg_benchmark.config import list_available_configs
print(list_available_configs())
```

## Configuration File Structure

Each YAML file follows this structure:

```yaml
# Problem identification
problem:
  name: "problem-name-v0"
  class: "ProblemClassName"

# Level generation parameters
level:
  width: 11
  height: 7

# Game-specific parameters
game:
  enemies: 3
  difficulty: 1

# Quality evaluation parameters
quality:
  enemy_range_factor: 0.25

# Rendering parameters
rendering:
  scale: 16

# Tile types (for reference)
tiles:
  solid: 0
  empty: 1
  player: 2

# Control space parameters (if applicable)
control:
  player_key_min_factor: 0.5
```

## Creating New Configurations

1. Create a new YAML file named `{problem-name}.yaml`
2. Follow the structure above
3. Include all necessary parameters for the problem constructor
4. Add comments to explain parameter meanings

## Problem-Specific Notes

### Zelda Problems
- `enemies`: Number of enemies to place
- `diversity`: Diversity threshold for evaluation
- `sol_length`: Target solution length (defaults to width + height)

### Sokoban Problems
- `difficulty`: Puzzle difficulty level (1-4)
- `solver`: Solver timeout/iterations

### Binary Problems
- Simple binary tile problems with just width and height parameters

## Integration with Algorithms

The configuration system works seamlessly with the algorithms in the `algos` directory:

```python
from pcg_benchmark.config import get_problem_params
from pcg_benchmark.algos import train_ppo
import pcg_benchmark

# Load problem with config
env = pcg_benchmark.make('zelda-v0')

# Train PPO agent
agent = train_ppo(env, total_timesteps=100000)
``` 
import os
import pandas as pd
import pcg_benchmark
from config import list_available_configs
import numpy as np
from tqdm import tqdm
import random
# Configurable parameters
NUM_EPISODES = 1  # Number of rollouts per problem

# DataFrame rows will be appended here
solved_data = []

# Get list of all available problem configurations
all_problems = list_available_configs()

print(f"Found {len(all_problems)} problems to evaluate...")

problems_with_errors = []
successful_problems = []

for problem_name in tqdm(all_problems, desc="Evaluating Problems"):
    if problem_name not in ['binary-v0', 'ddave-v0', 'mdungeons-v0', 'sokoban-complex-v0', 'sokoban-v0', 'zelda-enemies-v0', 'zelda-large-v0', 'zelda-v0']:
        continue
    try:
        env = pcg_benchmark.make(problem_name)
    except Exception as e:
        print(f"Could not instantiate problem {problem_name}: {e}")
        problems_with_errors.append(problem_name)
        continue

    for episode in range(NUM_EPISODES):
        try:
            observation, info = env.reset()
            done = False
            steps = 0

            while not done:
                action = random.randint(0, env.action_space-1) 
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if steps > 10:  # Safety cap
                    break

                done = terminated
                    
            if info.get("quality", 0) == 1:
                print(f"Solved level!")

                solved_data.append({
                    "problem_name": problem_name,
                    "obs_dims": str(obs.shape),
                    "level": level,
                    "action_space": str(env.action_space)
                })

            successful_problems.append(problem_name)
        except Exception as e:
            print(f"Error during episode {episode} of {problem_name}: {e}")
            problems_with_errors.append(problem_name)
            continue

# Convert to DataFrame
df = pd.DataFrame(solved_data, columns=["problem_name", "obs_dims", "level", "action_space"])

# Output summary and save
print(f"\nCollected {len(df)} solved levels.")
df.to_csv("solved_levels.csv", index=False)
print("Saved to solved_levels.csv.")

print(f"\nproblems_with_errors:\n{problems_with_errors}")
print(f"\successful_problems:\n{successful_problems}")
successful_problems

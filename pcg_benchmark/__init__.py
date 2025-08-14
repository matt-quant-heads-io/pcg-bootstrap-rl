from pcg_benchmark.pcg_env import PCGEnv
from pcg_benchmark.gym_pcg_env import GymPCGEnv
from pcg_benchmark.probs import PROBLEMS
from difflib import SequenceMatcher
import json

"""
Register a new problem that is not part of the probs folder to the system

Parameters:
    name (string): the name of the problem, usually the basic name is {name}-{version} 
    for example zelda-v0 for a modified version of the problem it follows {name}-{modification}-{version} 
    for example zelda-enemies-v0.
    problemClass (class): the class of the problem and it has to be a subclass of pcg_benchmark.probs.problem.Problem.
    problemArgs (dict[string,any]): the parameters the constructor of the problemClass needs for example 
    width and height for level generation.
"""
def register(name, problemClass, problemArgs={}):
    if name in PROBLEMS:
        raise AttributeError(f'This problem name ({name}) is already been defined')
    PROBLEMS[name] = (problemClass, problemArgs)

"""
Get all the registered environments in the pcg_benchmark either using register function or 
from exisiting in probs folder.

Returns:
    string[]: an array of all the names of the registered environments in pcg_benchmark.
"""
def list():
    names = []
    for name in PROBLEMS:
       names.append(name)
    return names

"""
create and initialize an environment from the pcg_benchmark using its name.

Parameters:
    name (string): the name of the environment that you want to initialize.

Returns:
    PCGEnv: an environment of that problem where you can test quality, diversity, 
    and controllability of your generator.
"""
# def make(name):
#     if not (name in PROBLEMS):
#         prob_names = PROBLEMS.keys()
#         max_sim = 0
#         sim_name = ""
#         for n in prob_names:
#             sim = SequenceMatcher(None, n, name).ratio()
#             if sim > max_sim:
#                 max_sim = sim
#                 sim_name = n
#         raise NotImplementedError(f'This problem ({name}) is not implemented. Did you mean to write ({sim_name}) instead.')
#     problemClass = PROBLEMS[name]
#     problemArgs = {}
#     if hasattr(PROBLEMS[name], "__len__"):
#         problemClass = PROBLEMS[name][0]
#         if len(PROBLEMS[name]) > 1:
#             problemArgs = PROBLEMS[name][1]
#     return PCGEnv(name, problemClass(**problemArgs))


def make(name):
    # import pdb; pdb.set_trace()
    if not (name in PROBLEMS):
        prob_names = PROBLEMS.keys()
        max_sim = 0
        sim_name = ""
        for n in prob_names:
            sim = SequenceMatcher(None, n, name).ratio()
            if sim > max_sim:
                max_sim = sim
                sim_name = n
        raise NotImplementedError(f'This problem ({name}) is not implemented. Did you mean to write ({sim_name}) instead.')
    problemClass = PROBLEMS[name]
    problemArgs = {}
    if hasattr(PROBLEMS[name], "__len__"):
        problemClass = PROBLEMS[name][0]
        if len(PROBLEMS[name]) > 1:
            problemArgs = PROBLEMS[name][1]
    return GymPCGEnv(name, problemClass(**problemArgs))


from gymnasium.vector import SyncVectorEnv
def make_vectorized_pcg_env(name: str, num_envs: int, asynchronous: bool = False):
    """
    Create a vectorized PCG environment using Gymnasium's built-in vectorization
    
    Parameters:
        name: Environment name
        problem: Problem instance
        num_envs: Number of parallel environments
        asynchronous: Whether to use AsyncVectorEnv (True) or SyncVectorEnv (False)
    
    Returns:
        VectorEnv: Vectorized environment
    """
    env_fns = [make_gym_env(name) for _ in range(num_envs)]
    
    if asynchronous:
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)



def append_to_json_file(filename, new_data):
    """
    Appends new_data to a JSON file containing a list of objects.
    If the file is empty or doesn't exist, it initializes it with a list.
    """
    from typing import List
    try:
        with open(filename, 'r+') as file:
            # Load existing data
            try:
                file_data = json.load(file)
            except json.JSONDecodeError: # Handle empty or invalid JSON
                file_data = []

            # Append new data
            if isinstance(file_data, List):
                file_data.append(new_data)
            else:
                print("Warning: JSON file does not contain a list. Overwriting with new data.")
                file_data = [new_data]

            # Go to the beginning of the file and write the updated data
            file.seek(0)
            json.dump(file_data, file, indent=4)
            file.truncate() # Remove any remaining old content if the new content is smaller

    except FileNotFoundError:
        # If the file doesn't exist, create it with the new data in a list
        with open(filename, 'w') as file:
            json.dump([new_data], file, indent=4)
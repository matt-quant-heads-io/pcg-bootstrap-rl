import os
import time
import argparse
import numpy as np
import gymnasium as gym
import torch
# import wandb
import logging
from collections import deque
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import utils
from config import Config
from algos.on_policy_ppo import OnPolicyPPOAgent, OnPolicyPPOTrainer
from algos.off_policy_ppo import OffPolicyPPOAgent, OffPolicyPPOTrainer
from algos.sl import SLTrainer
# from wrappers import PCGRLWrapper
import torch.nn.functional as F
import json
import pcg_benchmark



    

def create_env(env_name: str):
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except:
        raise ValueError(f"Invalid environment name: {env_name}")
    return env


def get_env_dims(env, config):
    """
    Gets observation and action dimensions from environment.
    
    Args:
        env: Gym environment
        config: Configuration object
    
    Returns:
        Tuple[int, int]: Observation dimension, Action dimension
    """
    return (config.obs_dim, config.obs_dim, config.action_dim), config.action_dim


def get_trainer(config: Config):
    if config.algo == "on_policy_ppo":
        trainer = OnPolicyPPOTrainer(
            config
        )
    elif config.algo == "off_policy_ppo":
        trainer = OffPolicyPPOTrainer(
            config
        )
    elif config.algo == "sl":
        trainer = SLTrainer(
            config
        )
    else:
        raise ValueError(f"Algo must be on_policy_ppo, off_policy_ppo, or SL!")

    return trainer



def run_training(
    env, 
    config: Config, 
    device: str
):
    start_time = time.time()

    # NOTE FIX THIS
    # obs_dim, action_dim = get_env_dims(env, config)

    
    trainer = get_trainer(config)
    _, metrics = trainer.train(max_total_training_steps=config.max_total_training_steps, max_steps=config.max_steps)

    # wandb.finish()    # close wandb logging
    print(f"Training time: {(time.time()-start_time) / 60.0:.2f} mins")
    
    return metrics

def run_evaluation(
        env, 
        config: Config, 
        device: str,
        save_video: bool = False,
        verbose: bool = True
    ):
    """
    Evaluates a trained agent on the given environment
    """

    obs_dim, action_dim = get_env_dims(env, config)

    metrics = {
        'eps_rewards': [],
        'eps_lengths': [],
        'mean_reward': 0,
        'std_reward': 0,
        'min_reward': float('inf'),
        'max_reward': float('-inf'),
        'mean_eps_length': 0,
        'total_steps': 0,
        'eval_dt': 0,
    }

    # create agent
    ppo_agent = make_agent(obs_dim, action_dim, config, device)

    # load model
    ckpt_path = os.path.join(config.ckpt_dir, config.eval_ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # print(ckpt.keys())
        ppo_agent.policy.load_state_dict(ckpt)
        logging.info(f"Successfully loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    
    # set policy to evaluation mode
    ppo_agent.policy.eval()
    recent_rewards = deque(maxlen=10)    # Track recent rewards for early stopping
    start_time = time.time()

    with torch.no_grad():
        for eps_so_far in range(config.num_eval_eps):
            obs, _ = env.reset()
            eps_reward = 0
            eps_length = 0

            # start episode
            for step in range(config.max_eps_steps):
                if config.render_mode:
                    env.render()

                action, logprob, value = ppo_agent.policy.select_action(obs)
                next_obs, reward, done, _, info = env.step(action.item())

                eps_reward += reward
                eps_length += 1

                obs = next_obs
                if done:
                    break
            
            # update metrics
            metrics['eps_rewards'].append(eps_reward)
            metrics['eps_lengths'].append(eps_length)
            metrics['min_reward'] = min(metrics['min_reward'], eps_reward)
            metrics['max_reward'] = max(metrics['max_reward'], eps_reward)
            metrics['total_steps'] += eps_length
            recent_rewards.append(eps_reward)

            # Early stopping check
            if len(recent_rewards) == recent_rewards.maxlen:
                if np.std(recent_rewards) < 0.1 * np.mean(recent_rewards):
                    logging.info(f"Early stopping as rewards have converged")
                    break

    metrics['eval_dt'] = time.time() - start_time
    metrics['mean_reward'] = np.mean(metrics['eps_rewards'])
    metrics['std_reward'] = np.std(metrics['eps_rewards'])
    metrics['mean_eps_length'] = np.mean(metrics['eps_lengths'])

    if verbose:
        logging.info(f"\nEvaluation Summary:")
        logging.info(f"Mean Reward: {metrics['mean_reward']:.2f} +- {metrics['std_reward']:.2f}")
        logging.info(f"Min/Max Reward: {metrics['min_reward']:.2f}/{metrics['max_reward']:.2f}")
        logging.info(f"Mean Episode Length: {metrics['mean_eps_length']:.2f}")
        logging.info(f"Total Steps: {metrics['total_steps']}")
        logging.info(f"Eval Time: {metrics['eval_dt']:.2f} s")

    env.close()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--config_path', type=str, required=True, 
                        help='path to config file')
    parser.add_argument('--wandb_dir', type=str, default=None, 
                        help='path to wandb directory')
    parser.add_argument('--verbose', action='store_true', 
                        help='enable verbose logging')

    # Allow overriding any config parameter from command line
    parser.add_argument('--override', nargs='*', default=[], 
                        help='override parameters, format: key=value')
    parser.add_argument('--problem_name', type=str, default="zelda-v0", 
                        help='problem name')
    args = parser.parse_args()
    return args

def set_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')    # for Apple Macbook GPUs
    else:
        device = torch.device('cpu')

    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)
    return device

def main(config, problem_name):
    device = set_device()
    logging.info(f'using device: {device}')
    # logging.info(config)

    # create environment
    # logging.info(f"Environment: {config.env_name}")
    # env = create_env(config.env_name)
    # from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

    env = pcg_benchmark.make_gym_env(problem_name) 

    # if config.random_seed:
    #     utils.set_random_seed(config.random_seed)

    config.log_dir = os.path.join(config.log_dir, problem_name, config.experiment_name)
    config.ckpt_dir = os.path.join(config.log_dir, "checkpoints")

    
    os.makedirs(config.ckpt_dir, exist_ok=True)

    config.logpath = os.path.join(config.log_dir, f'log.txt')
    logging.info(f"Logs saved at: {config.logpath}")
    with open(config.logpath, 'w') as f:
        pass

    metrics = run_training(env, config, device)

    print("Done!")
    env.close()


if __name__ == "__main__":

    # parse command-line arguments
    args = parse_args()

    if args.wandb_dir is None:
        args.wandb_dir = os.path.join(os.getcwd(), '../')
    # os.makedirs(args.wandb_dir, exist_ok=True)

    # load configuration
    config = Config.load_config(args.config_path)

    # override with command-line arguments
    overrides = {}
    for override in args.override:
        key, value = override.split('=')
        try:
            value = eval(value)
        except:
            pass
        overrides[key] = value
    config = Config.update_config(config, overrides)

    # Set up logging
    # logger = utils.setup_logging(log_dir=config.log_dir, verbose=args.verbose)

    main(config, args.problem_name)
import yaml
import os
from typing import Dict, Any
from types import SimpleNamespace

def load_config(problem_name: str) -> Dict[str, Any]:
    """
    Load configuration for a given problem from YAML file.
    
    Parameters:
        problem_name (str): Name of the problem (e.g., 'zelda-v0')
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_dir = os.path.dirname(__file__)
    config_file = os.path.join(config_dir, f"{problem_name}.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_problem_params(problem_name: str) -> Dict[str, Any]:
    """
    Extract problem parameters from config file for use with pcg_benchmark.make().
    
    Parameters:
        problem_name (str): Name of the problem
        
    Returns:
        Dict[str, Any]: Parameters dictionary suitable for problem constructor
    """
    config = load_config(problem_name)
    
    # Extract parameters from different sections
    params = {}
    
    # Level parameters
    if 'level' in config:
        params.update(config['level'])
    
    # Game parameters
    if 'game' in config:
        params.update(config['game'])
    
    return params

def list_available_configs() -> list:
    """
    List all available configuration files.
    
    Returns:
        list: List of available problem configuration names
    """
    config_dir = os.path.dirname(__file__)
    configs = []
    
    for file in os.listdir(config_dir):
        if file.endswith('.yaml') and not file.startswith('_'):
            configs.append(file[:-5])  # Remove .yaml extension
    
    return sorted(configs)


class Config:
    """Configuration class to load, save and update configuration"""

    @staticmethod
    def _convert_dict_to_obj(config_dict: Dict[str, Any]) -> SimpleNamespace:
        """Convert dictionary to an object with dot notation access"""
        namespace = SimpleNamespace()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(namespace, key, Config._convert_dict_to_obj(value))    # Recursively convert nested dicts
            else:
                setattr(namespace, key, value)
        return namespace


    @staticmethod
    def _convert_obj_to_dict(ns: SimpleNamespace) -> Dict:
        """Convert a Config object back to a dictionary"""
        output = {}
        for key, value in ns.__dict__.items():
            if isinstance(value, SimpleNamespace):
                output[key] = Config._convert_object_to_dict(value)
            else:
                output[key] = value
        return output


    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config._convert_dict_to_obj(config_dict)


    @staticmethod
    def save_config(config, save_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = Config._convert_obj_to_dict(config)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f)


    @staticmethod
    def update_config(config: SimpleNamespace, updates: Dict) -> SimpleNamespace:
        """Update configuration with new parameters"""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates like 'training.learning_rate'
                keys = key.split('.')
                conf = config
                for k in keys[:-1]:
                    if not hasattr(conf, k):
                        setattr(conf, k, SimpleNamespace())
                    conf = getattr(conf, k)
                setattr(conf, keys[-1], value)
            else:
                setattr(config, key, value)
        return config
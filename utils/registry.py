from typing import Dict, Type

MODEL_REGISTRY: Dict[str, type] = {}
AGENT_REGISTRY: Dict[str, type] = {}

def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco

def register_agent(name: str):
    def deco(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return deco

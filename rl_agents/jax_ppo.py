# jax_ppo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from jax_models import CNNPolicy


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    total_steps: int = 200_000
    rollout_length: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    seed: int = 0
    save_every_steps: int = 50_000  # checkpoint cadence


def make_ppo_train_state(rng, obs_shape: Tuple[int,int,int], n_actions: int, lr: float) -> TrainState:
    model = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions)
    dummy = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
    params = model.init(rng, dummy)['params']
    tx = optax.adam(lr)
    return TrainState(apply_fn=model.apply, params=params, tx=tx, opt_state=tx.init(params))


def _eval_logprob(logits: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(logp, actions[..., None], axis=-1).squeeze(-1)


def _entropy(logits: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.softmax(logits)
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(p * logp, axis=-1)


def ppo_loss(params: FrozenDict,
             apply_fn,
             obs: jnp.ndarray,          # [B,C,H,W]
             actions: jnp.ndarray,      # [B]
             old_logprobs: jnp.ndarray, # [B]
             advantages: jnp.ndarray,   # [B]
             returns: jnp.ndarray,      # [B]
             clip_coef: float,
             vf_coef: float,
             ent_coef: float):
    logits, values = apply_fn({'params': params}, obs)
    new_logprobs = _eval_logprob(logits, actions)
    ratio = jnp.exp(new_logprobs - old_logprobs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
    v_loss = jnp.mean((values - returns) ** 2)
    ent = jnp.mean(_entropy(logits))
    loss = pg_loss + vf_coef * v_loss - ent_coef * ent
    metrics = dict(pg_loss=pg_loss, v_loss=v_loss, entropy=ent, loss=loss)
    return loss, metrics


@jax.jit
def ppo_update(train_state: TrainState,
               obs, actions, old_logprobs, advantages, returns,
               clip_coef, vf_coef, ent_coef):
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(train_state.params, train_state.apply_fn,
                                     obs, actions, old_logprobs, advantages, returns,
                                     clip_coef, vf_coef, ent_coef)
    new_state = train_state.apply_gradients(grads=grads)
    return new_state, metrics


@jax.jit
def forward_logits_value(params, apply_fn, obs: jnp.ndarray):
    """JITed forward used during rollouts (env itself stays in Python)."""
    return apply_fn({'params': params}, obs)


def gae_scan(rewards: jnp.ndarray,
             values: jnp.ndarray,
             dones: jnp.ndarray,
             last_value: jnp.ndarray,
             gamma: float,
             lam: float):
    """
    JAX GAE with reverse scan.
    All inputs shape [T], returns (adv [T], ret [T]).
    """
    def step(carry, t):
        lastgaelam = carry
        nextnonterminal = 1.0 - dones[t]
        nextvalue = jnp.where(t == rewards.shape[0]-1, last_value, values[t+1])
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        return lastgaelam, lastgaelam

    # scan reversed indices
    init = jnp.array(0.0, dtype=jnp.float32)
    _, adv_rev = jax.lax.scan(step, init, jnp.arange(rewards.shape[0]-1, -1, -1))
    adv = jnp.flip(adv_rev, axis=0)
    returns = adv + values
    return adv, returns

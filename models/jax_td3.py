# jax_td3.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import optax


# -------------------- Models --------------------

def _pool_nchw(x, window, strides, avg=False):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = nn.avg_pool(x_nhwc, window, strides, 'VALID') if avg \
        else nn.max_pool(x_nhwc, window, strides, 'VALID')
    return jnp.transpose(y, (0, 3, 1, 2))

class CNNFeat(nn.Module):
    in_channels: int
    @nn.compact
    def __call__(self, x):
        def conv(ch): return nn.Conv(ch, (3,3), padding='SAME')
        x = conv(32)(x); x = nn.relu(x)
        x = _pool_nchw(x, (2,2), (2,2), avg=False)
        x = conv(64)(x); x = nn.relu(x)
        x = _pool_nchw(x, (2,2), (2,2), avg=False)
        x = conv(64)(x); x = nn.relu(x)

        # adaptive avg pool to 10x10
        _, _, H, W = x.shape
        sh = max(1, H // 10); sw = max(1, W // 10)
        kh = max(1, H - 9 * sh); kw = max(1, W - 9 * sw)
        x = _pool_nchw(x, (kh, kw), (sh, sw), avg=True)
        return x.reshape((x.shape[0], -1))


class TD3Actor(nn.Module):
    obs_shape: Tuple[int,int,int]
    n_actions: int
    @nn.compact
    def __call__(self, x):
        z = CNNFeat(self.obs_shape[0])(x)
        h = nn.relu(nn.Dense(256)(z))
        out = nn.Dense(self.n_actions)(h)
        return jnp.tanh(out)  # actions in [-1,1]


class TD3Critic(nn.Module):
    obs_shape: Tuple[int,int,int]
    n_actions: int
    @nn.compact
    def __call__(self, x, a):
        z = CNNFeat(self.obs_shape[0])(x)
        qa = jnp.concatenate([z, a], axis=-1)
        h = nn.relu(nn.Dense(256)(qa))
        q = nn.Dense(1)(h)
        return jnp.squeeze(q, axis=-1)


# -------------------- Config --------------------

@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    batch_size: int = 256
    seed: int = 0


# -------------------- TrainStates & Update --------------------

def make_td3_states(rng, obs_shape, n_actions, cfg: TD3Config):
    dummy_obs = jnp.zeros((1, *obs_shape), jnp.float32)
    dummy_act = jnp.zeros((1, n_actions), jnp.float32)
    actor = TD3Actor(obs_shape, n_actions)
    critic = TD3Critic(obs_shape, n_actions)
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    aparams = actor.init(k1, dummy_obs)['params']
    q1params = critic.init(k2, dummy_obs, dummy_act)['params']
    q2params = critic.init(k3, dummy_obs, dummy_act)['params']
    actor_state = TrainState.create(apply_fn=actor.apply, params=aparams, tx=optax.adam(cfg.lr))
    critic1_state = TrainState.create(apply_fn=critic.apply, params=q1params, tx=optax.adam(cfg.lr))
    critic2_state = TrainState.create(apply_fn=critic.apply, params=q2params, tx=optax.adam(cfg.lr))
    # targets
    target_actor_params = aparams
    target_q1_params = q1params
    target_q2_params = q2params
    return rng, actor_state, critic1_state, critic2_state, target_actor_params, target_q1_params, target_q2_params


@jax.jit
def td3_critic_update(actor_state, critic1_state, critic2_state,
                      target_actor_params, target_q1_params, target_q2_params,
                      batch, cfg: TD3Config, rng):
    # target action with smoothing noise
    rng, nk = jax.random.split(rng)
    next_a = actor_state.apply_fn({'params': target_actor_params}, batch['obs_next'])
    noise = jax.random.normal(nk, next_a.shape) * cfg.policy_noise
    noise = jnp.clip(noise, -cfg.noise_clip, cfg.noise_clip)
    next_a = jnp.clip(next_a + noise, -1.0, 1.0)

    q1_t = critic1_state.apply_fn({'params': target_q1_params}, batch['obs_next'], next_a)
    q2_t = critic2_state.apply_fn({'params': target_q2_params}, batch['obs_next'], next_a)
    tq = jnp.minimum(q1_t, q2_t)
    target = batch['rew'] + cfg.gamma * (1.0 - batch['done']) * tq

    def loss_fn(p1, p2):
        q1 = critic1_state.apply_fn({'params': p1}, batch['obs'], batch['act'])
        q2 = critic2_state.apply_fn({'params': p2}, batch['obs'], batch['act'])
        l1 = jnp.mean((q1 - target)**2)
        l2 = jnp.mean((q2 - target)**2)
        return l1 + l2, (l1, l2)

    (loss, (l1, l2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic1_state.params, critic2_state.params)
    g1, g2 = grads
    new_c1 = critic1_state.apply_gradients(grads=g1)
    new_c2 = critic2_state.apply_gradients(grads=g2)
    return rng, new_c1, new_c2, {'critic_loss': loss, 'q1_loss': l1, 'q2_loss': l2}


@jax.jit
def td3_actor_update(actor_state, critic1_state, batch):
    def loss_fn(p):
        a = actor_state.apply_fn({'params': p}, batch['obs'])
        q = critic1_state.apply_fn({'params': critic1_state.params}, batch['obs'], a)
        return -jnp.mean(q)
    loss, grads = jax.value_and_grad(loss_fn)(actor_state.params)
    new_actor = actor_state.apply_gradients(grads=grads)
    return new_actor, {'actor_loss': loss}


def soft_update(p, tp, tau):
    return jax.tree_util.tree_map(lambda a,b: (1.0 - tau)*b + tau*a, p, tp)

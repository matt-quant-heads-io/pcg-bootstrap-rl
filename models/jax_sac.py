# jax_sac.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
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
        tgt_h, tgt_w = 10, 10
        sh = max(1, H // tgt_h); sw = max(1, W // tgt_w)
        kh = max(1, H - (tgt_h - 1) * sh); kw = max(1, W - (tgt_w - 1) * sw)
        x = _pool_nchw(x, (kh, kw), (sh, sw), avg=True)
        return x.reshape((x.shape[0], -1))


class SACActor(nn.Module):
    obs_shape: Tuple[int,int,int]  # (C,H,W)
    n_actions: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x, rng=None, sample=True):
        z = CNNFeat(self.obs_shape[0])(x)
        h = nn.relu(nn.Dense(256)(z))
        mu = nn.Dense(self.n_actions)(h)
        log_std = nn.Dense(self.n_actions)(h)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)

        if sample:
            assert rng is not None
            eps = jax.random.normal(rng, mu.shape)
            pre_tanh = mu + std * eps
        else:
            pre_tanh = mu

        a = jnp.tanh(pre_tanh)
        # log prob with tanh correction
        log_prob = None
        if sample:
            log_prob = -0.5 * (((pre_tanh - mu) / (std + 1e-8))**2 + 2*log_std + jnp.log(2*jnp.pi))
            log_prob = jnp.sum(log_prob, axis=-1)
            # Tanh correction: log(1 - tanh(x)^2) = log(1 - a^2)
            log_prob -= jnp.sum(jnp.log(jnp.clip(1 - a**2, 1e-6, 1.0)), axis=-1)
        return a, log_prob, mu, log_std


class SACCritic(nn.Module):
    obs_shape: Tuple[int,int,int]
    n_actions: int
    @nn.compact
    def __call__(self, x, a):
        # x: [B,C,H,W], a: [B, A] in [-1,1]
        z = CNNFeat(self.obs_shape[0])(x)
        qa = jnp.concatenate([z, a], axis=-1)
        h = nn.relu(nn.Dense(256)(qa))
        q = nn.Dense(1)(h)
        return jnp.squeeze(q, axis=-1)


# -------------------- Config --------------------

@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005           # target smoothing
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy_scale: float = 0.98  # target_entropy = -scale * action_dim
    batch_size: int = 256
    seed: int = 0


# -------------------- TrainStates --------------------

def make_sac_states(rng, obs_shape: Tuple[int,int,int], n_actions: int, cfg: SACConfig):
    dummy_obs = jnp.zeros((1, *obs_shape), jnp.float32)
    dummy_act = jnp.zeros((1, n_actions), jnp.float32)

    actor = SACActor(obs_shape, n_actions)
    critic1 = SACCritic(obs_shape, n_actions)
    critic2 = SACCritic(obs_shape, n_actions)

    rng, akey = jax.random.split(rng)
    aparams = actor.init(akey, dummy_obs, rng=akey, sample=True)['params']
    c1params = critic1.init(rng, dummy_obs, dummy_act)['params']
    c2params = critic2.init(rng, dummy_obs, dummy_act)['params']

    actor_state  = TrainState.create(apply_fn=actor.apply, params=aparams, tx=optax.adam(cfg.lr))
    critic1_state= TrainState.create(apply_fn=critic1.apply, params=c1params, tx=optax.adam(cfg.lr))
    critic2_state= TrainState.create(apply_fn=critic2.apply, params=c2params, tx=optax.adam(cfg.lr))

    # targets
    target_c1_params = c1params
    target_c2_params = c2params

    # entropy temperature
    target_entropy = -cfg.target_entropy_scale * float(n_actions)
    log_alpha = jnp.array(0.0)
    alpha_opt = optax.adam(cfg.alpha_lr)
    alpha_opt_state = alpha_opt.init(log_alpha)

    return (rng, actor_state, critic1_state, critic2_state,
            target_c1_params, target_c2_params, log_alpha, alpha_opt_state, target_entropy, alpha_opt)


# -------------------- SAC update (JIT) --------------------

@jax.jit
def sac_update(rng,
               actor_state: TrainState,
               critic1_state: TrainState,
               critic2_state: TrainState,
               target_c1_params: FrozenDict,
               target_c2_params: FrozenDict,
               log_alpha: jnp.ndarray,
               alpha_opt_state,
               target_entropy: float,
               alpha_opt,
               batch: Dict[str, jnp.ndarray],
               cfg: SACConfig):

    alpha = jnp.exp(log_alpha)

    def critic_loss_fn(c1_params, c2_params):
        # Compute target Q
        rng_pi, sub = jax.random.split(rng)
        new_a, logp, _, _ = actor_state.apply_fn({'params': actor_state.params}, batch['obs_next'], rng=rng_pi, sample=True)
        q1_t = critic1_state.apply_fn({'params': target_c1_params}, batch['obs_next'], new_a)
        q2_t = critic2_state.apply_fn({'params': target_c2_params}, batch['obs_next'], new_a)
        tq = jnp.minimum(q1_t, q2_t) - alpha * logp
        target = batch['rew'] + cfg.gamma * (1.0 - batch['done']) * tq

        q1 = critic1_state.apply_fn({'params': c1_params}, batch['obs'], batch['act'])
        q2 = critic2_state.apply_fn({'params': c2_params}, batch['obs'], batch['act'])
        loss1 = jnp.mean((q1 - target)**2)
        loss2 = jnp.mean((q2 - target)**2)
        return loss1 + loss2, {'critic_loss': loss1 + loss2}

    grads, aux = jax.grad(lambda p1, p2: critic_loss_fn(p1, p2), has_aux=True)(critic1_state.params, critic2_state.params)
    # split the grads for c1/c2:
    grads_c1, grads_c2 = grads
    new_c1 = critic1_state.apply_gradients(grads=grads_c1)
    new_c2 = critic2_state.apply_gradients(grads=grads_c2)

    # Actor + alpha
    def actor_loss_fn(a_params, log_alpha):
        rng_pi, _ = jax.random.split(rng)
        a, logp, _, _ = actor_state.apply_fn({'params': a_params}, batch['obs'], rng=rng_pi, sample=True)
        q1 = new_c1.apply_fn({'params': new_c1.params}, batch['obs'], a)
        q2 = new_c2.apply_fn({'params': new_c2.params}, batch['obs'], a)
        q = jnp.minimum(q1, q2)
        actor_loss = jnp.mean(alpha * logp - q)

        # temperature loss
        alpha_loss = jnp.mean(-log_alpha * (logp + target_entropy).detach() if hasattr(logp, "detach")
                              else -log_alpha * (logp + target_entropy))
        return actor_loss, (actor_loss, alpha_loss, logp)

    # compute grads
    (actor_loss, (actor_loss_val, alpha_loss_val, logp_samp)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, log_alpha)

    # update actor
    new_actor = actor_state.apply_gradients(grads=actor_grads)

    # update alpha
    alpha_grads = jax.grad(lambda la: actor_loss_fn(new_actor.params, la)[1][1])(log_alpha)
    new_alpha_opt_state = alpha_opt.update(alpha_grads, alpha_opt_state)[0]
    new_log_alpha = optax.apply_updates(log_alpha, alpha_opt.update(alpha_grads, alpha_opt_state).updates)

    # soft update targets
    def soft(p, tp): return jax.tree_util.tree_map(lambda a,b: (1.0 - cfg.tau)*b + cfg.tau*a, p, tp)
    new_t1 = soft(new_c1.params, target_c1_params)
    new_t2 = soft(new_c2.params, target_c2_params)

    metrics = {
        'critic_loss': aux['critic_loss'],
        'actor_loss': actor_loss_val,
        'alpha': jnp.exp(new_log_alpha),
        'entropy': -jnp.mean(logp_samp)
    }

    return (rng, new_actor, new_c1, new_c2, new_t1, new_t2,
            new_log_alpha, new_alpha_opt_state, metrics)

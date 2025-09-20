# rl_agents/hooks.py
from typing import Dict, Any
import numpy as np
import torch

def hook_fit_obs_norm(agent, ctx: Dict[str, Any]):
    """
    Roll out K steps with random policy to fit observation normalization stats.
    ctx: {"steps": 5000}
    """
    steps = int(ctx.get("steps", 5000))
    if not hasattr(agent, "obs_normalizer"):
        print("[fit_obs_norm] agent has no obs_normalizer; skipping")
        return
    obs, info = agent.env.reset()
    for _ in range(steps):
        state = agent.preprocess_observation(obs)
        agent.obs_normalizer.observe(state)  # your normalizer should update running stats
        a = np.random.randint(agent.action_dim)  # or env.action_space.n
        obs, r, term, trunc, info = agent.env.step(int(a))
        if term or trunc:
            obs, info = agent.env.reset()
    agent.obs_normalizer.freeze()


def hook_replay_warmup(agent, ctx: Dict[str, Any]):
    """
    Fill SAC/TD3 replay buffer with random policy or a heuristic policy.
    ctx: {"steps": 50_000, "policy": "random"}  # or provide callable in ctx["policy_fn"]
    """
    steps = int(ctx.get("steps", 50_000))
    policy = ctx.get("policy", "random")
    policy_fn = ctx.get("policy_fn", None)

    if not hasattr(agent, "replay"):
        print("[replay_warmup] no replay buffer on agent; skipping")
        return

    obs, info = agent.env.reset()
    for _ in range(steps):
        if policy_fn is not None:
            a = policy_fn(agent, obs)
        elif policy == "random":
            a = np.random.randint(agent.action_dim)
        else:
            # simple epsilon-greedy on current model if available
            epsilon = float(ctx.get("epsilon", 0.2))
            if np.random.rand() < epsilon:
                a = np.random.randint(agent.action_dim)
            else:
                state = agent.preprocess_observation(obs)
                a = agent.act_eval(state)
        next_obs, r, term, trunc, info2 = agent.env.step(int(a))
        done = term or trunc
        agent.replay.add(obs, a, r, next_obs, done)  # implement .add in your replay
        obs = next_obs if not done else agent.env.reset()[0]


def hook_behavior_cloning(agent, ctx: Dict[str, Any]):
    """
    Pretrain policy with behavior cloning (cross-entropy on expert actions).
    ctx: {"dataset": "/path/to/demos.npz", "epochs": 5, "batch_size": 256, "lr": 3e-4}
    Expect dataset with keys: "obs" (N,C,H,W) or (N,H,W,C), "act" (N,)
    """
    import os
    ds_path = ctx.get("dataset", "")
    if not ds_path or not os.path.exists(ds_path):
        print(f"[bc] dataset not found: {ds_path}; skipping")
        return

    epochs = int(ctx.get("epochs", 5))
    bs = int(ctx.get("batch_size", 256))
    lr = float(ctx.get("lr", 3e-4))

    data = np.load(ds_path)
    obs_np = data["obs"]
    act_np = data["act"].astype(np.int64)

    # simple torch loop for PPO policy; adapt for JAX if needed
    agent.model.train()
    opt = torch.optim.Adam(agent.model.parameters(), lr=lr)
    n = obs_np.shape[0]
    idxs = np.arange(n)
    ce = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        np.random.shuffle(idxs)
        total = 0.0
        for i in range(0, n, bs):
            mb = idxs[i:i+bs]
            x = torch.tensor(obs_np[mb], dtype=torch.float32, device=agent.device)
            y = torch.tensor(act_np[mb], dtype=torch.long, device=agent.device)
            logits, _ = agent.model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(mb)
        print(f"[bc] epoch {ep+1}/{epochs} loss={total/n:.4f}")
    agent.model.eval()


def hook_load_pretrained_backbone(agent, ctx: Dict[str, Any]):
    """
    Load encoder/backbone weights then (optionally) freeze for k steps.
    ctx: {"checkpoint": "path/to.pt", "freeze_steps": 0}
    """
    import os
    ckpt = ctx.get("checkpoint", "")
    if not ckpt or not os.path.exists(ckpt):
        print(f"[load_backbone] checkpoint not found: {ckpt}; skipping")
        return
    freeze_steps = int(ctx.get("freeze_steps", 0))

    sd = torch.load(ckpt, map_location=agent.device)
    # expect agent.model.backbone exists; adapt to your model layout
    missing, unexpected = agent.model.backbone.load_state_dict(sd, strict=False)
    print(f"[load_backbone] loaded; missing={missing}, unexpected={unexpected}")

    if freeze_steps > 0:
        for p in agent.model.backbone.parameters():
            p.requires_grad = False
        agent._unfreeze_backbone_at_step = getattr(agent, "_unfreeze_backbone_at_step", None)
        # install a small gate that you check in your train loop
        agent._unfreeze_backbone_at_step = freeze_steps
        print(f"[load_backbone] backbone frozen for first {freeze_steps} steps")


def hook_eval_sanity(agent, ctx: Dict[str, Any]):
    """
    Quick evaluation before training, to ensure metrics/plots infra works.
    ctx: {"episodes": 3}
    """
    episodes = int(ctx.get("episodes", 3))
    res = agent._evaluate_for_checkpoint(episodes, save_images=True, step=0)
    # also write per-ckpt metrics/plots to ckpt_00000000
    agent._write_eval_json(0, res, extra={"checkpoint_path": None})
    agent._plot_metrics_by_checkpoint(step=0)


# Central registry so you can refer to hooks by "type" in YAML
_HOOK_LIBRARY = {
    "fit_obs_norm": hook_fit_obs_norm,
    "replay_warmup": hook_replay_warmup,
    "behavior_cloning": hook_behavior_cloning,
    "load_pretrained_backbone": hook_load_pretrained_backbone,
    "eval_sanity": hook_eval_sanity,
}

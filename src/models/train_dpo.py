#!/usr/bin/env python3
# train_dpo.py
# Run from project root (so `src.models` imports work).

import argparse, os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# ── Boilerplate to import your src/ package ──────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── ActorNetwork imports (discrete only) ────────────────────────────────────
from PPO_discrete import ActorNetwork as ActorCartpole
from PPO_mountaincar import ActorNetwork as ActorMountainCar

# ── Args ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(
    description="DPO fine-tuning sweep on CartPole-v1 or MountainCar-v0"
)
p.add_argument(
    "--env", choices=["cartpole", "mountaincar"], required=True,
    help="Which environment to run"
)
p.add_argument(
    "--rollout_dir", type=str, required=True,
    help="Folder with full/partial trajectories + preference_pairs.csv"
)
p.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to your half-policy .pth checkpoint"
)
p.add_argument(
    "--output", default="dpo_actor",
    help="Base name for saved DPO policies"
)
p.add_argument(
    "--epochs", type=int, default=5, help="DPO training epochs"
)
p.add_argument(
    "--batch_size", type=int, default=16, help="Minibatch size"
)
p.add_argument(
    "--beta", type=float, default=0.5, help="Inverse-temperature β"
)
p.add_argument(
    "--eval_episodes", type=int, default=50,
    help="Episodes per policy for evaluation"
)
p.add_argument(
    "--train_Ks", nargs="+", type=int, default=None,
    help="List of preference-dataset sizes K to sweep"
)
p.add_argument(
    "--seeds", nargs="+", type=int, default=[0,1,2],
    help="Random seeds for averaging"
)
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using device:", device)
os.makedirs(args.rollout_dir, exist_ok=True)

# ── Make a plots directory ──────────────────────────────────────────────────
plots_dir = os.path.join(project_root, "plots", "dpo", args.env)
os.makedirs(plots_dir, exist_ok=True)

# ── Load trajectories ────────────────────────────────────────────────────────
def load_trajs(path):
    df = pd.read_csv(path)
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    act_cols = [c for c in df.columns if c.startswith("action_") or c == "action"]
    trajs = {}
    for ep, g in df.groupby("episode_id"):
        g = g.sort_values("step")
        trajs[int(ep)] = [
            ([row[c] for c in obs_cols], [row[c] for c in act_cols])
            for _, row in g.iterrows()
        ]
    return trajs

full_csv    = os.path.join(args.rollout_dir, "full_model_trajectories.csv")
partial_csv = os.path.join(args.rollout_dir, "partial_model_trajectories.csv")
pairs_csv   = os.path.join(args.rollout_dir, "preference_pairs.csv")

print("Loading trajectories from:\n ", full_csv, "\n ", partial_csv)
full_trajs    = load_trajs(full_csv)
partial_trajs = load_trajs(partial_csv)

# ── Load preference pairs ────────────────────────────────────────────────────
all_pairs = pd.read_csv(pairs_csv).to_dict("records")
N_pairs   = len(all_pairs)
print(f"Total preference pairs: {N_pairs}")
train_Ks = sorted(args.train_Ks) if args.train_Ks else [N_pairs]

# prepare to collect losses for aggregation
losses_all = {K: [] for K in train_Ks}
results    = []

# ── Env & Actor setup ────────────────────────────────────────────────────────
if args.env == "cartpole":
    env_id    = "CartPole-v1"
    ActorNet  = ActorCartpole
elif args.env == "mountaincar":
    env_id    = "MountainCar-v0"
    ActorNet  = ActorMountainCar
else:
    raise ValueError("Unsupported env")

env         = gym.make(env_id)
obs_shape   = env.observation_space.shape
n_actions   = env.action_space.n
actor_args  = (n_actions, obs_shape)

# ── Helper: log-prob of a trajectory ─────────────────────────────────────────
def logprob_traj(actor, traj):
    lp = 0.0
    for state, action in traj:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist = actor(s)  # always Categorical for these two envs
        a = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(0)
        lp += dist.log_prob(a).sum()
    return lp

# ── Main sweep: seeds × K ────────────────────────────────────────────────────
for seed in args.seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for K in train_Ks:
        use_K = min(K, N_pairs)
        if K > N_pairs:
            print(f"Warning: requested K={K} but only {N_pairs} available → using {N_pairs}")
        prefs = random.sample(all_pairs, use_K)

        # instantiate & freeze reference
        actor_ref   = ActorNet(*actor_args).to(device)
        actor_train = ActorNet(*actor_args).to(device)
        actor_ref.load_checkpoint(args.checkpoint)
        actor_train.load_checkpoint(args.checkpoint)
        for p in actor_ref.parameters():
            p.requires_grad = False

        optimizer   = AdamW(actor_train.parameters(), lr=3e-5)

        # DPO fine-tuning
        dpo_losses = []
        for ep in range(1, args.epochs+1):
            torch.manual_seed(seed + ep)
            perm  = torch.randperm(use_K)
            e_loss = 0.0

            for i in range(0, use_K, args.batch_size):
                batch_idx  = perm[i : i + args.batch_size]
                optimizer.zero_grad()
                b_loss = 0.0

                for idx in batch_idx.tolist():
                    p = prefs[idx]
                    pos = (full_trajs[p["traj1_id"]] 
                           if p["chosen_one"] == 0 
                           else partial_trajs[p["traj2_id"]])
                    neg = (partial_trajs[p["traj2_id"]] 
                           if p["chosen_one"] == 0 
                           else full_trajs[p["traj1_id"]])

                    dθ = logprob_traj(actor_train, pos) - logprob_traj(actor_train, neg)
                    with torch.no_grad():
                        dr = (logprob_traj(actor_ref, pos) 
                              - logprob_traj(actor_ref, neg))

                    b_loss += -F.logsigmoid(args.beta * (dθ - dr))

                b_loss = b_loss / len(batch_idx)
                b_loss.backward()
                optimizer.step()
                e_loss += b_loss.item()

            avg_l = e_loss / (use_K / args.batch_size)
            dpo_losses.append(avg_l)
            print(f"[env={args.env} seed={seed} K={K}] Epoch {ep}/{args.epochs} — DPO loss {avg_l:.4f}")

        losses_all[K].append(dpo_losses)

        # save fine-tuned policy
        out_name = f"{args.output}_{args.env}_s{seed}_K{K}.pth"
        torch.save(actor_train.state_dict(), out_name)
        print("→ Saved", out_name)

        # evaluate returns
        def evaluate(actor):
            rs = []
            for _ in range(args.eval_episodes):
                obs, _ = env.reset()
                done   = False
                tot    = 0.0
                while not done:
                    s   = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    a   = Categorical(actor(s).probs).sample().cpu().item()
                    obs, r, done, _, _ = env.step(a)
                    tot += r
                rs.append(tot)
            return np.array(rs)

        ref_R   = evaluate(actor_ref)
        dpo_R   = evaluate(actor_train)
        win_rate = sum(
            int(
                logprob_traj(actor_train, 
                             full_trajs[p["traj1_id"]] 
                             if p["chosen_one"] == 0 
                             else partial_trajs[p["traj2_id"]])
              > logprob_traj(actor_train,
                             partial_trajs[p["traj2_id"]] 
                             if p["chosen_one"] == 0 
                             else full_trajs[p["traj1_id"]])
            )
            for p in prefs
        ) / use_K

        results.append({
            "env":        args.env,
            "seed":       seed,
            "K":          K,
            "ref_mean":   ref_R.mean(),
            "ref_std":    ref_R.std(),
            "train_mean": dpo_R.mean(),
            "train_std":  dpo_R.std(),
            "win_rate":   win_rate
        })

# ── Save summary ───────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
out_csv = os.path.join(plots_dir, "dpo_summary.csv")
df.to_csv(out_csv, index=False)
print("Sweep done. Summary →", out_csv)



# ---- Example Command to Run ----
# python src/models/train_dpo.py \
#   --env cartpole \
#   --rollout_dir ppo_cartpole_rollouts \
#   --checkpoint checkpoints_cartpole/half_actor_model_CP \
#   --train_Ks 5 \
#   --seeds 0 1 \
#   --epochs 5 \
#   --batch_size 16 \
#   --beta 0.5
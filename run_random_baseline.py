# run_random_baseline.py
"""
Random baseline for the RecSim interest_exploration environment.

This script:
- Uses the same overridden RecSim environment (via ./overrides).
- Runs a purely random policy for N episodes.
- Saves episodic returns under runs/<run_name>/eval_random/returns_random,
  so analyze_returns.py can read it.
"""

import argparse
import pathlib
import sys
from typing import List

import numpy as np


def make_env(seed: int = 0):
    """Construct the interest_exploration environment with a dict config."""
    root = pathlib.Path(__file__).parent.resolve()
    overrides_dir = root / "overrides"

    # Make our local overrides/recsim/... shadow site-packages (use our choice_model).
    if overrides_dir.exists():
        sys.path.insert(0, str(overrides_dir))

    from recsim.environments import interest_exploration  # imported after tweaking sys.path

    env_config = {
        "slate_size": 5,
        "num_candidates": 20,
        "seed": seed,
        "resample_documents": True,  
    }
    return interest_exploration.create_environment(env_config)


def run_random(num_episodes: int, max_steps: int, seed: int = 0) -> List[float]:
    """Run a random policy and return episodic returns."""
    env = make_env(seed=seed)

    # Seed numpy + env if possible
    np.random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)

    returns: List[float] = []

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            # Sample a random action from the env's action space.
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            if done:
                break

        returns.append(total_reward)

    return returns


def save_returns(run_name: str, returns: List[float]) -> pathlib.Path:
    """Save returns in a simple text format."""
    root = pathlib.Path(__file__).parent.resolve()
    run_dir = root / "runs" / run_name
    eval_dir = run_dir / "eval_random"
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_path = eval_dir / "returns_random"
    with out_path.open("w") as f:
        for r in returns:
            f.write(f"{r}\n")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Random baseline for RecSim interest_exploration."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="random_baseline",
        help="Directory name under ./runs/ to save results.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=50,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    returns = run_random(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps_per_episode,
        seed=args.seed,
    )

    out_path = save_returns(args.run_name, returns)

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"Saved random baseline returns to: {out_path}")
    print(f"Num episodes: {len(returns)}")
    print(f"Mean return : {mean_ret:.3f}  (std: {std_ret:.3f})")


if __name__ == "__main__":
    main()

"""
Bandit-style baseline for the RecSim interest_exploration environment.

Idea:
- Treat each slate (action) we have tried as an arm.
- Keep running averages of reward per action.
- With epsilon-greedy, pick either:
  - a random action (explore), or
  - the best-mean-reward action seen so far (exploit).

This is a simple non-RL baseline that's still smarter than pure random.
"""

import argparse
import pathlib
from typing import Dict, List, Tuple
import copy
import numpy as np

from recsim.environments import interest_exploration


def make_env(seed: int = 0):
    """Construct the interest_exploration environment with default config."""
    # interest_exploration.create_environment can be called with None or a dict.
    # Using None gets the default config (same as RecSim examples).
    env = interest_exploration.create_environment(None)
    # Seed if supported
    if hasattr(env, "seed"):
        env.seed(seed)
    return env


def action_key(action) -> Tuple:
    """Turn an action into a hashable key (tuple of numbers)."""
    arr = np.asarray(action)
    return tuple(arr.ravel().tolist())


def run_bandit(
    num_episodes: int,
    max_steps: int,
    epsilon: float = 0.1,
    seed: int = 0,
) -> List[float]:
    """Run epsilon-greedy bandit baseline and return episodic returns."""
    np.random.seed(seed)
    env = make_env(seed=seed)

    # stats[key] = [count, total_reward]
    stats: Dict[Tuple, List[float]] = {}
    # store the actual action object we can replay
    actions_by_key: Dict[Tuple, object] = {}

    returns: List[float] = []

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            # Decide: explore or exploit
            if (len(stats) == 0) or (np.random.rand() < epsilon):
                # Explore: random action
                action = env.action_space.sample()
            else:
                # Exploit: pick action with highest mean reward so far
                best_key = max(
                    stats.keys(),
                    key=lambda k: stats[k][1] / stats[k][0],
                )
                action = copy.deepcopy(actions_by_key[best_key])

            key = action_key(action)
            # Keep a representative action object for this key
            if key not in actions_by_key:
                actions_by_key[key] = copy.deepcopy(action)

            obs, reward, done, info = env.step(action)
            r = float(reward)
            total_reward += r

            # Update bandit stats
            if key not in stats:
                stats[key] = [0.0, 0.0]
            stats[key][0] += 1.0          # count
            stats[key][1] += r            # total_reward

            if done:
                break

        returns.append(total_reward)

    return returns


def save_returns(run_name: str, returns: List[float]) -> pathlib.Path:
    """Save returns to runs/<run_name>/eval_bandit/returns_bandit."""
    root = pathlib.Path(__file__).parent.resolve()
    run_dir = root / "runs" / run_name
    eval_dir = run_dir / "eval_bandit"
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_path = eval_dir / "returns_bandit"
    with out_path.open("w") as f:
        for r in returns:
            f.write(f"{r}\n")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Bandit-style baseline for RecSim interest_exploration."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="bandit_baseline",
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
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate (probability of random action).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

"""
Bandit-style baseline for the RecSim interest_exploration environment.

Idea:
- Treat each slate (action) we have tried as an arm.
- Keep running averages of reward per action.
- With epsilon-greedy, pick either:
  - a random action (explore), or
  - the best-mean-reward action seen so far (exploit).

This is a simple non-RL baseline that's still smarter than pure random.
"""

import argparse
import pathlib
from typing import Dict, List, Tuple
import copy
import numpy as np

from recsim.environments import interest_exploration


def make_env(seed: int = 0):
    """Construct the interest_exploration environment with default config."""
    # interest_exploration.create_environment can be called with None or a dict.
    # Using None gets the default config (same as RecSim examples).
    env = interest_exploration.create_environment(None)
    # Seed if supported
    if hasattr(env, "seed"):
        env.seed(seed)
    return env


def action_key(action) -> Tuple:
    """Turn an action into a hashable key (tuple of numbers)."""
    arr = np.asarray(action)
    return tuple(arr.ravel().tolist())


def run_bandit(
    num_episodes: int,
    max_steps: int,
    epsilon: float = 0.1,
    seed: int = 0,
) -> List[float]:
    """Run epsilon-greedy bandit baseline and return episodic returns."""
    np.random.seed(seed)
    env = make_env(seed=seed)

    # stats[key] = [count, total_reward]
    stats: Dict[Tuple, List[float]] = {}
    # store the actual action object we can replay
    actions_by_key: Dict[Tuple, object] = {}

    returns: List[float] = []

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            # Decide: explore or exploit
            if (len(stats) == 0) or (np.random.rand() < epsilon):
                # Explore: random action
                action = env.action_space.sample()
            else:
                # Exploit: pick action with highest mean reward so far
                best_key = max(
                    stats.keys(),
                    key=lambda k: stats[k][1] / stats[k][0],
                )
                action = copy.deepcopy(actions_by_key[best_key])

            key = action_key(action)
            # Keep a representative action object for this key
            if key not in actions_by_key:
                actions_by_key[key] = copy.deepcopy(action)

            obs, reward, done, info = env.step(action)
            r = float(reward)
            total_reward += r

            # Update bandit stats
            if key not in stats:
                stats[key] = [0.0, 0.0]
            stats[key][0] += 1.0          # count
            stats[key][1] += r            # total_reward

            if done:
                break

        returns.append(total_reward)

    return returns


def save_returns(run_name: str, returns: List[float]) -> pathlib.Path:
    """Save returns to runs/<run_name>/eval_bandit/returns_bandit."""
    root = pathlib.Path(__file__).parent.resolve()
    run_dir = root / "runs" / run_name
    eval_dir = run_dir / "eval_bandit"
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_path = eval_dir / "returns_bandit"
    with out_path.open("w") as f:
        for r in returns:
            f.write(f"{r}\n")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Bandit-style baseline for RecSim interest_exploration."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="bandit_baseline",
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
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate (probability of random action).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    returns = run_bandit(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps-per-episode if hasattr(args, "max_steps-per-episode") else args.max_steps_per_episode,  # just in case
        epsilon=args.epsilon,
        seed=args.seed,
    )

    out_path = save_returns(args.run_name, returns)
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    print(f"Saved bandit baseline returns to: {out_path}")
    print(f"Num episodes: {len(returns)}")
    print(f"Mean return : {mean_ret:.3f}  (std: {std_ret:.3f})")


if __name__ == "__main__":
    main()


    out_path = save_returns(args.run_name, returns)
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    print(f"Saved bandit baseline returns to: {out_path}")
    print(f"Num episodes: {len(returns)}")
    print(f"Mean return : {mean_ret:.3f}  (std: {std_ret:.3f})")


if __name__ == "__main__":
    main()

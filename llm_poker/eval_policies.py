"""Evaluation helpers for LLM poker policies on top of `PokerTextEnv`.

This module does NOT talk to any specific LLM or RL framework. Instead,
it expects a simple `policy_fn(prompt: str) -> str` function, which
returns an action token like "A0", and then runs many episodes to
estimate average reward and win-rate proxies.

You can plug in:
    - A Qwen policy accessed via Tinker,
    - The base (unfine-tuned) Qwen policy,
    - Or any other policy implementation you like.
"""

from dataclasses import dataclass
from typing import Callable, List, Dict

from llm_poker.poker_env import PokerTextEnv, StepResult


PolicyFn = Callable[[str], str]


@dataclass
class EvalStats:
    """Aggregate statistics for evaluation."""

    num_episodes: int
    mean_total_reward: float
    rewards: List[float]


def evaluate_policy(
    policy: PolicyFn,
    opponent_type: str = "always_call",
    num_episodes: int = 1000,
    max_steps: int = 64,
) -> EvalStats:
    """Evaluate a policy in `PokerTextEnv` against a chosen opponent.

    Args:
        policy: Function mapping prompt -> action token.
        opponent_type: 'self', 'random', 'always_call', or 'baseline'.
        num_episodes: Number of hands to simulate.
        max_steps: Safety cap on steps per episode.

    Returns:
        EvalStats with per-episode rewards and mean reward.
    """
    env = PokerTextEnv(opponent_type=opponent_type)
    rewards: List[float] = []

    for _ in range(num_episodes):
        prompt = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < max_steps:
            steps += 1
            action_token = policy(prompt)
            result: StepResult = env.step(action_token)
            total_reward += result.reward
            done = result.done
            prompt = result.prompt

        rewards.append(total_reward)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return EvalStats(num_episodes=num_episodes, mean_total_reward=mean_reward, rewards=rewards)


def summarize_eval(eval_stats: EvalStats) -> Dict[str, float]:
    """Return a simple dict summary of evaluation statistics."""
    if not eval_stats.rewards:
        return {"num_episodes": 0, "mean_total_reward": 0.0}

    import numpy as np

    arr = np.array(eval_stats.rewards, dtype=float)
    return {
        "num_episodes": float(eval_stats.num_episodes),
        "mean_total_reward": float(arr.mean()),
        "std_total_reward": float(arr.std()),
        "p25_total_reward": float(np.percentile(arr, 25)),
        "p50_total_reward": float(np.percentile(arr, 50)),
        "p75_total_reward": float(np.percentile(arr, 75)),
    }



"""Example RL-style training loop for PokerTextEnv.

This module does NOT depend on Tinker directly. Instead, it shows how to:
    - Instantiate `PokerTextEnv`
    - Run episodes by querying a policy function on the text prompt
    - Accumulate rewards for basic policy-gradient style training

You can use this structure as a reference when wiring the environment
into Tinker's RL API as described in the Tinker docs:
https://tinker-docs.thinkingmachines.ai/rl/rl-basic
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

from llm_poker.poker_env import PokerTextEnv, StepResult


PolicyFn = Callable[[str], str]


@dataclass
class Episode:
    prompts: List[str]
    actions: List[str]
    rewards: List[float]
    total_reward: float


def run_episode(env: PokerTextEnv, policy: PolicyFn, max_steps: int = 64) -> Episode:
    """Run a single episode using the given policy function.

    Args:
        env: `PokerTextEnv` instance.
        policy: Function mapping a prompt string -> action token (e.g. "A0").
        max_steps: Safety cap on number of decisions per episode.

    Returns:
        Episode object containing the prompts, actions, per-step rewards, and
        total reward for the episode.
    """
    prompts: List[str] = []
    actions: List[str] = []
    rewards: List[float] = []

    prompt = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        steps += 1
        prompts.append(prompt)

        action_token = policy(prompt)
        actions.append(action_token)

        result: StepResult = env.step(action_token)
        rewards.append(result.reward)
        done = result.done
        prompt = result.prompt

    return Episode(
        prompts=prompts,
        actions=actions,
        rewards=rewards,
        total_reward=sum(rewards),
    )


def random_policy(prompt: str) -> str:
    """A trivial random policy that chooses uniformly among listed tokens.

    This is meant only for local sanity checks of the environment.
    In practice, Tinker will replace this with a Qwen-based policy that
    samples from the model given the prompt.
    """
    # This implementation relies on the fact that `format_state_prompt`
    # includes a line like "Valid tokens: A0, A1, A2".
    lines = prompt.splitlines()
    token_line = ""
    for line in lines[::-1]:
        if line.startswith("Valid tokens:"):
            token_line = line
            break
    if not token_line:
        return "A0"

    # Extract tokens after the colon
    tokens_part = token_line.split(":", 1)[-1].strip()
    tokens = [t.strip() for t in tokens_part.split(",") if t.strip()]
    if not tokens:
        return "A0"

    import random

    return random.choice(tokens)


def run_random_baseline(num_episodes: int = 10) -> List[Episode]:
    """Run a few random-policy episodes for quick environment sanity checks."""
    env = PokerTextEnv(opponent_type="always_call")
    episodes: List[Episode] = []
    for _ in range(num_episodes):
        ep = run_episode(env, random_policy)
        episodes.append(ep)
    return episodes



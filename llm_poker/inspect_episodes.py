"""Inspection helpers for LLM-driven poker episodes.

These utilities mirror what we did for NN-based agents (`scripts/inspect_hands.py`),
but operate on `PokerTextEnv` and a generic `policy_fn(prompt) -> token`.

You can use them to print out a small number of hands to qualitatively
check whether the RL-tuned Qwen agent is making sensible decisions.
"""

from typing import Callable

from llm_poker.poker_env import PokerTextEnv, StepResult


PolicyFn = Callable[[str], str]


def inspect_policy(
    policy: PolicyFn,
    opponent_type: str = "always_call",
    num_hands: int = 5,
    max_steps: int = 64,
) -> None:
    """Print a few hands played by `policy` vs the chosen opponent.

    This is intended for interactive debugging in a notebook or REPL.
    """
    env = PokerTextEnv(opponent_type=opponent_type)

    for hand_idx in range(num_hands):
        prompt = env.reset()
        done = False
        steps = 0

        print(f"=== Hand {hand_idx + 1} vs {opponent_type} ===")
        print("Initial prompt:")
        print(prompt)
        print("------")

        while not done and steps < max_steps:
            steps += 1
            action_token = policy(prompt)
            print(f"[Agent] action token: {action_token!r}")

            result: StepResult = env.step(action_token)
            done = result.done
            prompt = result.prompt

            if done:
                print(f"[Env] done=True, reward={result.reward}, info={result.info}")
            else:
                print(f"[Env] reward={result.reward}")
                print("Next prompt:")
                print(prompt)
                print("------")

        print()



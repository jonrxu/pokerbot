#!/usr/bin/env python3
"""
Interactive inspection of Tinker-driven poker hands.

This script is similar to `tinker_poker_eval.py` but prints out detailed
information per decision so you can see what the model is doing:

  - The prompt given to the model (PokerTextEnv state)
  - The raw model text
  - The parsed action token (e.g. "A0")
  - Per-step reward and final payoffs

It plays a small number of hands for debugging, not for large-scale eval.
"""

import argparse
import logging
from typing import Optional

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from llm_poker.poker_env import PokerTextEnv, StepResult


logger = logging.getLogger(__name__)


def parse_first_token(text: str) -> str:
    """Extract the first whitespace-delimited token from a model response."""
    text = text.strip()
    if not text:
        return "A0"
    return text.split()[0]


def inspect_poker_hands(
    model_name: str,
    num_hands: int,
    opponent_type: str,
    max_tokens: int = 32,
    base_url: Optional[str] = None,
) -> None:
    """Play a few hands and print detailed logs for each."""
    # Set up tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Tinker clients
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    env = PokerTextEnv(opponent_type=opponent_type)

    for hand_idx in range(num_hands):
        prompt = env.reset()
        done = False
        episode_reward = 0.0

        print("=" * 80)
        print(f"Hand {hand_idx + 1}/{num_hands} vs {opponent_type}")
        print("- Initial prompt:")
        print(prompt)
        print("-" * 80)

        step_num = 0
        while not done:
            step_num += 1

            # Wrap prompt in a simple conversation for the renderer
            convo = [
                {"role": "user", "content": prompt},
            ]
            model_input = renderer.build_generation_prompt(convo)

            # Sample model response
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sample_result = future.result()
            seq = sample_result.sequences[0]

            parsed_message, _ = renderer.parse_response(seq.tokens)
            raw_text = parsed_message["content"]
            action_token = parse_first_token(raw_text)

            print(f"[Step {step_num}] Model raw text:")
            print(raw_text)
            print(f"[Step {step_num}] Parsed action token: {action_token!r}")

            # Apply action to env
            step_result: StepResult = env.step(action_token)
            episode_reward += step_result.reward

            # Update loop condition
            done = step_result.done

            if done:
                payoffs = step_result.info.get("payoffs")
                print(f"[Step {step_num}] Episode done.")
                print(f"  Step reward: {step_result.reward:.4f}")
                print(f"  Total normalized episode reward: {episode_reward:.4f}")
                if payoffs is not None:
                    print(f"  Raw payoffs: {payoffs}")
            else:
                print(f"[Step {step_num}] Step reward: {step_result.reward:.4f}")
                print("- Next prompt:")
                print(step_result.prompt)
                print("-" * 80)

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Tinker-driven poker hands in detail.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name as understood by Tinker (e.g. Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=3,
        help="Number of poker hands to inspect.",
    )
    parser.add_argument(
        "--opponent-type",
        type=str,
        default="always_call",
        choices=["self", "random", "always_call", "baseline"],
        help="Opponent behavior in PokerTextEnv.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens to sample per decision.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional custom Tinker base URL (usually leave as default).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    inspect_poker_hands(
        model_name=args.model_name,
        num_hands=args.num_hands,
        opponent_type=args.opponent_type,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()



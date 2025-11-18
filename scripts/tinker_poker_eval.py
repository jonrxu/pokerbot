#!/usr/bin/env python3
"""
Evaluate a base LLM (via Tinker) in the poker environment.

This script:
  - Uses Tinker + tinker-cookbook to call a hosted model (e.g. Qwen)
  - Wraps your existing simplified heads-up NLHE game via `PokerTextEnv`
  - Lets the model choose actions by reading a prompt and emitting an action token
  - Reports average reward over many hands

This is intentionally evaluation-only (no training updates) but follows the
same patterns as the rl_loop / rl_basic examples in the Tinker Cookbook:
https://tinker-docs.thinkingmachines.ai/install
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


def run_poker_eval(
    model_name: str,
    num_hands: int,
    opponent_type: str,
    max_tokens: int = 32,
    base_url: Optional[str] = None,
) -> None:
    """Run `num_hands` episodes of PokerTextEnv using a base model via Tinker."""
    # Set up tokenizer and renderer (chat-style wrapper for prompts)
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

    # Poker environment (uses your existing simplified NLHE logic)
    env = PokerTextEnv(opponent_type=opponent_type)

    total_rewards = 0.0

    for hand_idx in range(num_hands):
        prompt = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # We wrap the text prompt into a single-user-message conversation
            convo = [
                {"role": "user", "content": prompt},
            ]
            model_input = renderer.build_generation_prompt(convo)

            # Sample one response from the model
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sample_result = future.result()
            seq = sample_result.sequences[0]

            # Parse model output text and convert to an action token (e.g. "A0")
            parsed_message, _ = renderer.parse_response(seq.tokens)
            raw_text = parsed_message["content"]
            action_token = parse_first_token(raw_text)

            # Step the poker environment
            step_result: StepResult = env.step(action_token)
            episode_reward += step_result.reward
            done = step_result.done
            prompt = step_result.prompt

        total_rewards += episode_reward
        logger.info(f"Hand {hand_idx + 1}/{num_hands}: episode_reward={episode_reward:.4f}")

    avg_reward = total_rewards / num_hands if num_hands > 0 else 0.0
    print("======================================")
    print(f"Model:        {model_name}")
    print(f"Opponent:     {opponent_type}")
    print(f"Hands:        {num_hands}")
    print(f"Avg reward:   {avg_reward:.6f} (normalized by starting stack)")
    print("Note: positive reward means winning chips on average as Player 0.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Tinker model in PokerTextEnv.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name as understood by Tinker (e.g. a Qwen or Llama model).",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=200,
        help="Number of poker hands to play for evaluation.",
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

    run_poker_eval(
        model_name=args.model_name,
        num_hands=args.num_hands,
        opponent_type=args.opponent_type,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Small RL run for Qwen via Tinker in the PokerTextEnv.

This script:
  - Uses a Tinker TrainingClient (LoRA) on top of a base model (e.g. Qwen3-4B-Instruct-2507)
  - Plays episodes in PokerTextEnv vs a fixed opponent (e.g. always_call)
  - Treats each LLM decision as a bandit-style rollout with a trajectory-level reward
  - Builds importance-sampling Datums and calls `forward_backward` + `optim_step`

It is intentionally minimal and meant for small experiments, not long runs.
You should run this in the `poker_bot_tinker` env with TINKER_API_KEY set.
"""

import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from llm_poker.poker_env import PokerTextEnv, StepResult


logger = logging.getLogger(__name__)


def parse_first_token(text: str) -> str:
    """Extract the first whitespace-delimited token from a model response."""
    text = text.strip()
    if not text:
        return "A0"
    return text.split()[0]


def collect_batch_episodes(
    env: PokerTextEnv,
    sampling_client: tinker.SamplingClient,
    renderer,
    batch_size: int,
    max_tokens: int,
) -> Tuple[List[types.Datum], List[float]]:
    """Collect a small batch of episodes and convert them into Tinker Datums.

    For simplicity, we:
      - Treat each **LLM decision** as a separate trajectory
      - Use the final normalized episode return as the reward for all its decisions
      - Use importance_sampling loss with one sample per decision
    """
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    all_datums: List[types.Datum] = []
    episode_rewards: List[float] = []

    for _ in range(batch_size):
        prompt = env.reset()
        done = False
        episode_reward = 0.0

        # For each episode, we store (prompt_tokens, sampled_tokens, sampled_logprobs)
        decision_records: List[Tuple[List[int], List[int], List[float]]] = []

        while not done:
            convo = [{"role": "user", "content": prompt}]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            # Sample a response
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sample_result = future.result()
            seq = sample_result.sequences[0]
            sampled_tokens = seq.tokens
            sampled_logprobs = seq.logprobs
            if sampled_logprobs is None:
                # If logprobs are missing, skip this decision
                sampled_logprobs = [0.0] * len(sampled_tokens)

            # Parse token and step env
            parsed_message, _ = renderer.parse_response(sampled_tokens)
            raw_text = parsed_message["content"]
            action_token = parse_first_token(raw_text)

            step_result: StepResult = env.step(action_token)
            episode_reward += step_result.reward
            done = step_result.done
            prompt = step_result.prompt

            # Record decision
            decision_records.append((prompt_tokens, sampled_tokens, sampled_logprobs))

        # At this point, episode_reward includes formatting bonuses and final EV;
        # we use the final reward as the RL signal.
        episode_rewards.append(episode_reward)

        # Convert decisions into Datums with advantages = episode_reward
        for prompt_tokens, sampled_tokens, sampled_logprobs in decision_records:
            tokens = prompt_tokens + sampled_tokens
            ob_len = len(prompt_tokens) - 1

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            # Align logprobs to input_tokens: pad observation tokens with 0.0
            all_logprobs = [0.0] * ob_len + sampled_logprobs
            # Advantages: 0.0 for prompt tokens, episode_reward for action tokens
            all_advantages = [0.0] * ob_len + [episode_reward] * (len(input_tokens) - ob_len)

            assert (
                len(input_tokens)
                == len(target_tokens)
                == len(all_logprobs)
                == len(all_advantages)
            ), "Token/logprob/advantage lengths must match"

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                },
            )
            all_datums.append(datum)

    return all_datums, episode_rewards


def run_poker_rl(
    model_name: str,
    log_path: str,
    num_batches: int,
    episodes_per_batch: int,
    opponent_type: str,
    learning_rate: float,
    max_tokens: int,
    lora_rank: int,
    base_url: str | None,
) -> None:
    """Run a small RL loop on PokerTextEnv."""
    os.makedirs(log_path, exist_ok=True)

    # Logging
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=None,
        wandb_name=None,
        config={
            "model_name": model_name,
            "episodes_per_batch": episodes_per_batch,
            "num_batches": num_batches,
            "opponent_type": opponent_type,
            "learning_rate": learning_rate,
        },
        do_configure_logging_module=True,
    )

    # Tokenizer + renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Tinker clients
    service_client = tinker.ServiceClient(base_url=base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["loop_state"].get("batch", 0)
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
        )
        start_batch = 0
        logger.info("Starting new LoRA training client")

    sampling_path = (
        training_client.save_weights_for_sampler(name="init")
        .result()
        .path
    )
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    adam_params = types.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Poker env
    env = PokerTextEnv(opponent_type=opponent_type)

    for batch_idx in range(start_batch, num_batches):
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "progress/done_frac": (batch_idx + 1) / num_batches,
            "optim/lr": learning_rate,
        }

        logger.info(f"Collecting batch {batch_idx+1}/{num_batches} episodes...")
        datums, episode_rewards = collect_batch_episodes(
            env=env,
            sampling_client=sampling_client,
            renderer=renderer,
            batch_size=episodes_per_batch,
            max_tokens=max_tokens,
        )

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        metrics["reward/mean"] = avg_reward
        logger.info(f"Batch {batch_idx}: avg episode reward = {avg_reward:.4f}")

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            datums,
            loss_fn="importance_sampling",
        )
        optim_step_future = training_client.optim_step(adam_params)
        _ = fwd_bwd_future.result()
        _ = optim_step_future.result()

        # Save checkpoint and refresh sampling weights periodically
        if (batch_idx + 1) % 2 == 0 or batch_idx == num_batches - 1:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=log_path,
                kind="state",
                loop_state={"batch": batch_idx + 1},
            )
            sampling_path = (
                training_client.save_weights_for_sampler(
                    name=f"{batch_idx:06d}"
                )
                .result()
                .path
            )
            sampling_client = service_client.create_sampling_client(
                model_path=sampling_path
            )

        ml_logger.log_metrics(metrics, step=batch_idx)

    ml_logger.close()
    logger.info("Poker RL training completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Small RL run for Qwen in PokerTextEnv via Tinker.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model to fine-tune via LoRA.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="/tmp/tinker-examples/rl_poker_llm",
        help="Directory for Tinker logs and checkpoints.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of RL batches (iterations) to run.",
    )
    parser.add_argument(
        "--episodes-per-batch",
        type=int,
        default=8,
        help="Number of episodes per batch.",
    )
    parser.add_argument(
        "--opponent-type",
        type=str,
        default="always_call",
        choices=["self", "random", "always_call", "baseline"],
        help="Opponent behavior in PokerTextEnv during RL.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="Learning rate for LoRA optimizer.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens to sample per decision.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank for Tinker training client.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional custom Tinker base URL.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_poker_rl(
        model_name=args.model_name,
        log_path=args.log_path,
        num_batches=args.num_batches,
        episodes_per_batch=args.episodes_per_batch,
        opponent_type=args.opponent_type,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        lora_rank=args.lora_rank,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()



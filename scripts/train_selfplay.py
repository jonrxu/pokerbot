#!/usr/bin/env python3
"""
Self-Play RL run for Qwen via Tinker in the PokerTextEnv.

This script:
  - Uses a Tinker TrainingClient (LoRA) on top of a base model.
  - Runs full self-play games where the model controls both Player 0 and Player 1.
  - Collects trajectories for both players.
  - Assigns rewards correctly based on final payoffs.
  - Updates the model using importance sampling.

Run this in the `poker_bot_tinker` env with TINKER_API_KEY set.
"""

import argparse
import logging
import os
from typing import List, Tuple, Dict

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


def collect_selfplay_batch_episodes(
    env: PokerTextEnv,
    sampling_client: tinker.SamplingClient,
    renderer,
    batch_size: int,
    max_tokens: int,
    payoff_scale: float,
) -> Tuple[List[types.Datum], List[float]]:
    """Collect a batch of self-play episodes and convert them into Tinker Datums.
    
    In self-play, the model generates actions for both players.
    We record decisions for both, and assign rewards based on their respective payoffs.
    """
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    all_datums: List[types.Datum] = []
    episode_payoffs: List[float] = []  # Track avg payoff for P0 (to log performance)

    for _ in range(batch_size):
        prompt = env.reset()
        done = False
        
        # Track decisions: list of (player_idx, prompt_tokens, sampled_tokens, sampled_logprobs, accumulated_formatting_reward)
        # Note: formatting reward is immediate, but we usually sum it into return. 
        # Here we'll simplify: Return = Final Payoff + Sum of Formatting Bonuses.
        decision_records: List[Dict] = []
        
        # We also need to track accumulated formatting bonuses per player to add to their final return
        player_formatting_rewards = {0: 0.0, 1: 0.0}

        while not done:
            current_player = env.current_player
            
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
                sampled_logprobs = [0.0] * len(sampled_tokens)

            # Parse token and step env
            parsed_message, _ = renderer.parse_response(sampled_tokens)
            raw_text = parsed_message["content"]
            action_token = parse_first_token(raw_text)

            step_result: StepResult = env.step(action_token)
            
            # step_result.reward contains formatting bonus (if not done) or final reward (if done)
            # We separate formatting bonus from payoff.
            # If done=False, reward is formatting bonus.
            # If done=True, reward is formatting bonus (if any) + payoff? 
            # Actually PokerTextEnv logic: 
            #   If invalid: reward = penalty, done=True.
            #   If valid & terminal: reward = normalized payoff P0 (we ignore this here and use info['payoffs']).
            #   If valid & not terminal: reward = formatting bonus.
            
            immediate_reward = 0.0
            
            if step_result.done and "error" in step_result.info:
                # Invalid action penalty
                immediate_reward = step_result.reward
            elif not step_result.done:
                # Formatting bonus
                immediate_reward = step_result.reward
            
            player_formatting_rewards[current_player] += immediate_reward
            
            decision_records.append({
                "player": current_player,
                "prompt_tokens": prompt_tokens,
                "sampled_tokens": sampled_tokens,
                "sampled_logprobs": sampled_logprobs,
            })

            done = step_result.done
            prompt = step_result.prompt
            
            if done:
                # Get final payoffs
                payoffs = step_result.info.get("payoffs", [0.0, 0.0])
                # Normalize payoffs
                norm_payoffs = [
                    max(-payoff_scale, min(payoff_scale, p)) / payoff_scale 
                    for p in payoffs
                ]
                
                # If "error" in info, the payoffs might be 0, but we applied penalty above.
                # If valid terminal, we have payoffs.
                
                # Distribute rewards to trajectories
                for record in decision_records:
                    p_idx = record["player"]
                    # Total Return = Normalized Payoff + Accumulated Formatting Bonuses
                    # Note: strictly, return should be sum of future rewards.
                    # Since formatting bonus is immediate and payoff is terminal, 
                    # Return_t = Payoff + sum_{k=t}^{T} formatting_bonus_k.
                    # But for simplicity in this bandit-like formulation (one decision = one sample), 
                    # we often just give the full episode return.
                    # Let's just assign (Payoff_p + Total_Formatting_p).
                    
                    final_value = norm_payoffs[p_idx] + player_formatting_rewards[p_idx]
                    
                    # Create Datum
                    prompt_tokens = record["prompt_tokens"]
                    sampled_tokens = record["sampled_tokens"]
                    sampled_logprobs = record["sampled_logprobs"]
                    
                    tokens = prompt_tokens + sampled_tokens
                    ob_len = len(prompt_tokens) - 1
                    input_tokens = tokens[:-1]
                    target_tokens = tokens[1:]
                    
                    all_logprobs = [0.0] * ob_len + sampled_logprobs
                    all_advantages = [0.0] * ob_len + [final_value] * (len(input_tokens) - ob_len)
                    
                    assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages)
                    
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                        },
                    )
                    all_datums.append(datum)
                
                episode_payoffs.append(payoffs[0])

    return all_datums, episode_payoffs


def run_selfplay_rl(
    model_name: str,
    log_path: str,
    num_batches: int,
    episodes_per_batch: int,
    learning_rate: float,
    max_tokens: int,
    lora_rank: int,
    base_url: str | None,
) -> None:
    """Run the self-play RL loop."""
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
            "type": "self_play",
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

    # Initial weights for sampling
    sampling_path = (
        training_client.save_weights_for_sampler(name="init")
        .result()
        .path
    )
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    adam_params = types.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Self-play env
    env = PokerTextEnv(opponent_type="self")

    for batch_idx in range(start_batch, num_batches):
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "progress/done_frac": (batch_idx + 1) / num_batches,
            "optim/lr": learning_rate,
        }

        logger.info(f"Collecting batch {batch_idx+1}/{num_batches} episodes (Self-Play)...")
        datums, episode_payoffs = collect_selfplay_batch_episodes(
            env=env,
            sampling_client=sampling_client,
            renderer=renderer,
            batch_size=episodes_per_batch,
            max_tokens=max_tokens,
            payoff_scale=env.payoff_scale,
        )

        avg_payoff = float(np.mean(episode_payoffs)) if episode_payoffs else 0.0
        # In self-play, avg payoff for P0 should hover around 0 if balanced.
        # Positive means P0 > P1 (which is the same agent, so likely noise or position imbalance).
        metrics["reward/mean_payoff_p0"] = avg_payoff
        logger.info(f"Batch {batch_idx}: avg P0 payoff = {avg_payoff:.2f}")

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            datums,
            loss_fn="importance_sampling",
        )
        optim_step_future = training_client.optim_step(adam_params)
        _ = fwd_bwd_future.result()
        _ = optim_step_future.result()

        # Save checkpoint and refresh sampling weights
        # In self-play, it's crucial to update the sampling weights frequently 
        # so the agent plays against its improving self.
        if True: # Update every batch for self-play to ensure "latest vs latest"
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
    logger.info("Poker Self-Play RL training completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Play RL run for Qwen in PokerTextEnv via Tinker.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model to fine-tune via LoRA.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="/tmp/tinker-examples/rl_poker_selfplay",
        help="Directory for Tinker logs and checkpoints.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of RL batches (iterations) to run.",
    )
    parser.add_argument(
        "--episodes-per-batch",
        type=int,
        default=16,
        help="Number of episodes per batch.",
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

    run_selfplay_rl(
        model_name=args.model_name,
        log_path=args.log_path,
        num_batches=args.num_batches,
        episodes_per_batch=args.episodes_per_batch,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        lora_rank=args.lora_rank,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()


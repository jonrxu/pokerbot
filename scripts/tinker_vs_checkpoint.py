#!/usr/bin/env python3
"""
Evaluate a Tinker-hosted LLM (e.g., Qwen3-4B-Instruct-2507) against a
checkpointed NN agent (e.g., iteration 195) in the same PokerGame.

This mirrors the logic of `scripts/evaluate_benchmarks.py` but replaces
one of the NN agents with a Tinker-driven policy that acts via the
text-based interface (prompts + action tokens).
"""

import argparse
import logging
import os
from typing import Optional, List, Tuple

import numpy as np
import torch
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from poker_game.game import PokerGame, GameState, Action
from poker_game.state_encoder import StateEncoder
from models.value_policy_net import ValuePolicyNet
from llm_poker.text_interface import (
    build_action_options,
    format_state_prompt,
    parse_action_token,
    ActionOption,
)


logger = logging.getLogger(__name__)


class NNAgent:
    """Wrapper around a ValuePolicyNet for evaluation."""

    def __init__(self, policy_net: ValuePolicyNet, state_encoder: StateEncoder, device: str = "cpu"):
        self.policy_net = policy_net.to(device)
        self.policy_net.eval()
        self.state_encoder = state_encoder
        self.device = device

    def get_action(self, state: GameState, legal_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        if not legal_actions:
            return (Action.FOLD, 0)

        player = state.current_player
        encoding = self.state_encoder.encode(state, player)
        state_tensor = torch.tensor(encoding, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, policy_logits = self.policy_net(state_tensor)
            action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        num_legal = len(legal_actions)
        legal_probs = action_probs[:num_legal]
        if legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            legal_probs = np.ones(num_legal) / num_legal

        idx = np.random.choice(num_legal, p=legal_probs)
        return legal_actions[idx]


class TinkerLLMAgent:
    """Agent that uses a Tinker-hosted LLM via the text interface."""

    def __init__(
        self,
        model_name: str,
        game: PokerGame,
        tokenizer: Optional[object] = None,
        renderer: Optional[object] = None,
        sampling_client: Optional[tinker.SamplingClient] = None,
        sampling_params: Optional[types.SamplingParams] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 32,
    ):
        self.game = game

        # Set up tokenizer and renderer if not provided
        if tokenizer is None:
            tokenizer = get_tokenizer(model_name)
        if renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(model_name)
            renderer = renderers.get_renderer(renderer_name, tokenizer)
            logger.info(f"TinkerLLMAgent using renderer: {renderer_name}")
        self.renderer = renderer

        # Set up Tinker sampling client if not provided
        if sampling_client is None:
            service_client = tinker.ServiceClient(base_url=base_url)
            sampling_client = service_client.create_sampling_client(base_model=model_name)
        self.sampling_client = sampling_client

        if sampling_params is None:
            sampling_params = types.SamplingParams(
                max_tokens=max_tokens,
                stop=self.renderer.get_stop_sequences(),
            )
        self.sampling_params = sampling_params

    def _policy_token(self, prompt: str) -> str:
        """Call the LLM and extract the first token (e.g. 'A2')."""
        convo = [{"role": "user", "content": prompt}]
        model_input = self.renderer.build_generation_prompt(convo)

        future = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self.sampling_params,
        )
        sample_result = future.result()
        seq = sample_result.sequences[0]
        parsed_message, _ = self.renderer.parse_response(seq.tokens)
        raw_text = parsed_message["content"].strip()
        if not raw_text:
            return "A0"
        return raw_text.split()[0]

    def get_action(self, state: GameState, legal_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        if not legal_actions:
            return (Action.FOLD, 0)

        action_options: List[ActionOption] = build_action_options(self.game, state)
        prompt = format_state_prompt(self.game, state, action_options, current_player=state.current_player)
        token = self._policy_token(prompt)

        parsed = parse_action_token(token, action_options)
        if parsed is None:
            # Fallback: pick a safe action (call/check if available, else fold)
            for act, amt in legal_actions:
                if act in (Action.CALL, Action.CHECK):
                    return act, amt
            return (Action.FOLD, 0)
        return parsed


def load_checkpoint_agent(iteration: int, device: str = "cpu") -> NNAgent:
    """Load NN policy from /checkpoints/checkpoint_iter_{iteration}.pt."""
    # Resolve checkpoint path relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    checkpoint_path = os.path.join(
        project_root, "checkpoints", f"checkpoint_iter_{iteration}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_encoder = StateEncoder()
    input_dim = state_encoder.feature_dim
    policy_net = ValuePolicyNet(input_dim=input_dim)

    if "policy_net_state" not in checkpoint:
        raise ValueError(f"Checkpoint missing policy_net_state: {checkpoint_path}")
    policy_net.load_state_dict(checkpoint["policy_net_state"])

    return NNAgent(policy_net=policy_net, state_encoder=state_encoder, device=device)


def play_match(
    model_name: str,
    checkpoint_iter: int,
    num_games: int,
    model_path: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 32,
) -> None:
    """Run a head-to-head match between a Tinker LLM and an NN checkpoint."""
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False, starting_stack=20000)

    # Agents
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    service_client = tinker.ServiceClient(base_url=base_url)
    if model_path:
        logger.info(f"Using RL-tuned weights from: {model_path}")
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    llm_agent = TinkerLLMAgent(
        model_name=model_name,
        game=game,
        tokenizer=tokenizer,
        renderer=renderer,
        sampling_client=sampling_client,
        sampling_params=sampling_params,
    )
    nn_agent = load_checkpoint_agent(checkpoint_iter, device="cpu")

    llm_payoff = 0.0
    nn_payoff = 0.0
    llm_wins = 0
    nn_wins = 0

    for game_num in range(num_games):
        state = game.reset()

        # Randomize seating: LLM may be Player 0 or 1
        if np.random.rand() < 0.5:
            agents = [llm_agent, nn_agent]
            llm_is_player0 = True
        else:
            agents = [nn_agent, llm_agent]
            llm_is_player0 = False

        while not state.is_terminal:
            player = state.current_player
            legal_actions = game.get_legal_actions(state)
            if not legal_actions:
                break

            agent = agents[player]
            action, amount = agent.get_action(state, legal_actions)
            state = game.apply_action(state, action, amount)

        payoffs = game.get_payoff(state)

        if llm_is_player0:
            llm_payoff += payoffs[0]
            nn_payoff += payoffs[1]
            if payoffs[0] > payoffs[1]:
                llm_wins += 1
            elif payoffs[1] > payoffs[0]:
                nn_wins += 1
        else:
            llm_payoff += payoffs[1]
            nn_payoff += payoffs[0]
            if payoffs[1] > payoffs[0]:
                llm_wins += 1
            elif payoffs[0] > payoffs[1]:
                nn_wins += 1

    llm_win_rate = llm_wins / num_games
    nn_win_rate = nn_wins / num_games
    llm_avg = llm_payoff / num_games
    nn_avg = nn_payoff / num_games

    print("======================================")
    print(f"Tinker model:   {model_name}")
    print(f"NN checkpoint:  iter {checkpoint_iter}")
    print(f"Games played:   {num_games}")
    print()
    print(f"LLM win rate:   {llm_win_rate:.3f}")
    print(f"NN  win rate:   {nn_win_rate:.3f}")
    print(f"LLM avg payoff: {llm_avg:.2f} chips/game")
    print(f"NN  avg payoff: {nn_avg:.2f} chips/game")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pit a Tinker-hosted LLM against a local NN checkpoint in PokerGame."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Tinker model name (e.g., Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--checkpoint-iter",
        type=int,
        default=195,
        help="Iteration of NN checkpoint to load (default: 195).",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=500,
        help="Number of head-to-head games to play.",
    )
    parser.add_argument(
        "--rl-log-path",
        type=str,
        default="",
        help="Optional Tinker RL log path; if set and a checkpoint exists there, "
             "we evaluate the RL-tuned model instead of the base model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens to sample per LLM decision.",
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

    # If an RL log path is provided, try to load the latest state_path from there
    model_path: Optional[str] = None
    if args.rl_log_path:
        resume_info = checkpoint_utils.get_last_checkpoint(args.rl_log_path)
        if resume_info and "state_path" in resume_info:
            state_path = resume_info["state_path"]
            logger.info(f"Found RL training state at {state_path}, exporting sampler weights...")

            # Create a TrainingClient from the saved state, then export sampler weights
            service_client = tinker.ServiceClient(base_url=args.base_url)
            training_client = service_client.create_training_client_from_state(state_path)
            sampler_info = training_client.save_weights_for_sampler(name="eval").result()
            model_path = sampler_info.path
            logger.info(f"Using RL sampler weights from {args.rl_log_path}: {model_path}")
        else:
            logger.info(
                f"No RL checkpoint found at {args.rl_log_path}, falling back to base model."
            )

    play_match(
        model_name=args.model_name,
        checkpoint_iter=args.checkpoint_iter,
        num_games=args.num_games,
        model_path=model_path,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()



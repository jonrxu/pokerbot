"""Policy gradient training for poker LLM (no Tinker required)."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Episode:
    """A single episode of poker play."""
    prompts: List[str]
    actions: List[str]
    rewards: List[float]
    total_reward: float


class PolicyGradientTrainer:
    """REINFORCE-style policy gradient trainer for poker LLM.

    This implements a simple policy gradient algorithm:
    1. Collect episodes with current policy
    2. Calculate returns (cumulative rewards)
    3. Update policy to increase probability of high-reward actions
    """

    def __init__(
        self,
        agent,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,  # Discount factor
        clip_grad_norm: float = 1.0,
    ):
        """Initialize trainer.

        Args:
            agent: PokerLLMAgent instance
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            clip_grad_norm: Gradient clipping value
        """
        self.agent = agent
        self.gamma = gamma
        self.clip_grad_norm = clip_grad_norm

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            agent.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns for an episode.

        Args:
            rewards: List of rewards for each step

        Returns:
            List of discounted returns (G_t)
        """
        returns = []
        G = 0.0

        # Compute returns backwards
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return returns

    def train_step(self, episodes: List[Episode]) -> Dict[str, float]:
        """Perform a single training step on a batch of episodes.

        Args:
            episodes: List of Episode objects

        Returns:
            Dictionary of training metrics
        """
        self.agent.model.train()

        total_loss = 0.0
        num_steps = 0

        for episode in episodes:
            # Compute returns
            returns = self.compute_returns(episode.rewards)

            # Normalize returns (helps stability)
            if len(returns) > 1:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)
                returns = ((returns_tensor - returns_tensor.mean()) /
                          (returns_tensor.std() + 1e-8)).tolist()

            # Process each step in the episode
            for prompt, action, G in zip(episode.prompts, episode.actions, returns):
                # Get log probability of the action
                log_prob = self._get_action_log_prob(prompt, action)

                # Policy gradient loss: -log_prob * return
                # (We want to increase prob of high-return actions)
                loss = -log_prob * G

                total_loss += loss.item()
                num_steps += 1

                # Backward pass
                loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(),
            self.clip_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.agent.model.eval()

        # Return metrics
        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        avg_return = sum(ep.total_reward for ep in episodes) / len(episodes)

        return {
            "loss": avg_loss,
            "avg_return": avg_return,
            "num_episodes": len(episodes),
            "num_steps": num_steps,
        }

    def _get_action_log_prob(self, prompt: str, action: str) -> torch.Tensor:
        """Get log probability of an action given a prompt.

        Args:
            prompt: Game state prompt
            action: Action token (e.g., "A0")

        Returns:
            Log probability tensor
        """
        # Format prompt
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.agent.tokenizer, 'apply_chat_template'):
            text = self.agent.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        # Tokenize prompt + action
        full_text = text + action
        inputs = self.agent.tokenizer(text, return_tensors="pt").to(self.agent.device)
        full_inputs = self.agent.tokenizer(full_text, return_tensors="pt").to(self.agent.device)

        # Get model outputs
        with torch.set_grad_enabled(True):
            outputs = self.agent.model(**full_inputs)
            logits = outputs.logits

            # Get log probs for the action tokens
            prompt_length = inputs['input_ids'].shape[1]
            action_ids = full_inputs['input_ids'][0, prompt_length:]

            # Calculate log probability
            log_probs = F.log_softmax(logits[0, prompt_length-1:-1], dim=-1)
            action_log_probs = log_probs[range(len(action_ids)), action_ids]

            # Sum log probs for the full action
            total_log_prob = action_log_probs.sum()

        return total_log_prob

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.agent.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Tuple of (epoch, metrics)
        """
        checkpoint = torch.load(path, map_location=self.agent.device)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['metrics']

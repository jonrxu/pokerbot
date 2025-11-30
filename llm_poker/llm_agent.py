"""LLM agent for poker using transformers (no Tinker required)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import re


class PokerLLMAgent:
    """LLM poker agent using HuggingFace transformers.

    This agent uses a causal language model to generate poker actions
    based on text prompts describing the game state.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 10,
        temperature: float = 0.7,
    ):
        """Initialize the LLM agent.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()
        print(f"Model loaded successfully!")

    def get_action_token(self, prompt: str) -> str:
        """Generate an action token from a game state prompt.

        Args:
            prompt: Text description of game state

        Returns:
            Action token string (e.g., "A0", "A1", "A2")
        """
        # Format as chat message
        messages = [{"role": "user", "content": prompt}]

        # Tokenize
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Extract first token that looks like an action (A0, A1, etc.)
        action_token = self._parse_action_token(generated_text)
        return action_token

    def _parse_action_token(self, text: str) -> str:
        """Extract action token from generated text.

        Args:
            text: Generated text from model

        Returns:
            Action token (defaults to "A0" if not found)
        """
        # Look for pattern like A0, A1, A2, etc.
        match = re.search(r'\bA\d+\b', text)
        if match:
            return match.group(0)

        # Fallback: try to find just the first token
        tokens = text.strip().split()
        if tokens and tokens[0].startswith('A'):
            return tokens[0]

        # Default to first action
        return "A0"

    def get_trainable_parameters(self):
        """Get trainable parameters for fine-tuning."""
        return self.model.parameters()

    def save(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str = None):
        """Load model checkpoint."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        agent = cls.__new__(cls)
        agent.device = device
        agent.max_new_tokens = 10
        agent.temperature = 0.7

        print(f"Loading checkpoint from {path}...")
        agent.tokenizer = AutoTokenizer.from_pretrained(path)
        agent.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        agent.model.eval()

        return agent

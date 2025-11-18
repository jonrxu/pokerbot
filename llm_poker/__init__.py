"""LLM-based poker interfaces and environments.

This package provides:
- Text-based representations of `PokerGame` states suitable for LLM prompts.
- Utilities to map legal actions to stable action tokens and back.
- A simple RL-style environment wrapper (see `poker_env.py`) that can be
  integrated with external RL frameworks such as Tinker.
"""



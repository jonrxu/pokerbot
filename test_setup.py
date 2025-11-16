"""Test script to verify all components work."""

import sys
sys.path.insert(0, '.')

print("Testing imports...")

# Test game logic
from poker_game.game import PokerGame, GameState, Action
print("✓ Game logic imported")

from poker_game.state_encoder import StateEncoder
print("✓ State encoder imported")

from poker_game.information_set import InformationSet
print("✓ Information set imported")

# Test models
from models.value_policy_net import ValuePolicyNet
print("✓ Value-policy network imported")

# Test game functionality
game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
state = game.reset()
print(f"✓ Game initialized: Pot={state.pot}, Stacks={state.stacks}")

# Test state encoding
encoder = StateEncoder()
state_encoding = encoder.encode(state, 0)
print(f"✓ State encoded: Shape={state_encoding.shape}, Dim={encoder.feature_dim}")

# Test network creation
input_dim = encoder.feature_dim
value_net = ValuePolicyNet(input_dim=input_dim)
policy_net = ValuePolicyNet(input_dim=input_dim)
print(f"✓ Networks created: Input dim={input_dim}")

# Test a simple game play
legal_actions = game.get_legal_actions(state)
print(f"✓ Legal actions retrieved: {len(legal_actions)} actions available")

# Test action application
if legal_actions:
    action, amount = legal_actions[0]
    new_state = game.apply_action(state, action, amount)
    print(f"✓ Action applied: {action.name}, New pot={new_state.pot}")

print("\n✅ All basic tests passed! Ready for training.")


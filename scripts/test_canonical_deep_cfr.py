
import torch
import numpy as np
from poker_game.game import PokerGame
from poker_game.state_encoder import StateEncoder
from models.advantage_net import AdvantageNet
from models.policy_net import PolicyNet
from training.deep_cfr import DeepCFR

def test_canonical_deep_cfr_traversal():
    print("Initializing Canonical Deep CFR components...")
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    advantage_net = AdvantageNet(input_dim)
    policy_net = PolicyNet(input_dim)
    
    deep_cfr = DeepCFR(
        advantage_net=advantage_net,
        policy_net=policy_net,
        state_encoder=encoder,
        game=game,
        device='cpu'  # Test on CPU
    )
    
    print("Starting traversal test...")
    # Buffers for 2-network architecture
    buffers = {'advantage': [], 'policy': []}
    state = game.reset()
    
    # Run a few traversals
    for i in range(5):
        print(f"  Traversal {i+1}...")
        # Traversal player 0, using PolicyNet for player 1 simulation
        deep_cfr.traverse_external_sampling(state, player=0, buffers=buffers)
        state = game.reset()
        
    print("Checking buffers...")
    print(f"  Advantage buffer size: {len(buffers['advantage'])}")
    print(f"  Policy buffer size: {len(buffers['policy'])}")
    
    assert len(buffers['advantage']) > 0, "Advantage buffer should not be empty"
    assert len(buffers['policy']) > 0, "Policy buffer should not be empty"
    assert 'value' not in buffers, "Value buffer should not exist in Canonical Deep CFR"
    
    # Check shapes
    adv_state, adv_vec = buffers['advantage'][0]
    pol_state, pol_vec = buffers['policy'][0]
    
    print(f"  Advantage vector shape: {adv_vec.shape}")
    print(f"  Policy vector shape: {pol_vec.shape}")
    print(f"  Sample Advantage: {adv_vec}")
    print(f"  Sample Policy: {pol_vec}")
    
    assert adv_vec.shape == (20,), f"Expected advantage dim 20, got {adv_vec.shape}"
    assert pol_vec.shape == (20,), f"Expected policy dim 20, got {pol_vec.shape}"
    
    print("\n✓ Canonical Deep CFR logic verification passed!")

if __name__ == "__main__":
    test_canonical_deep_cfr_traversal()


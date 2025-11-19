import torch
import numpy as np
import os
import modal
from modal_deploy.config import checkpoint_volume, image
from poker_game.game import PokerGame, Action, GameState
from poker_game.state_encoder import StateEncoder
from models.advantage_net import AdvantageNet
from models.policy_net import PolicyNet

app = modal.App("poker-bot-diagnostic")

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume}
)
def diagnose_policy(iteration=25):
    print(f"Diagnosing Checkpoint Iteration {iteration}...")
    
    checkpoint_path = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Initialize
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    adv_net = AdvantageNet(input_dim)
    pol_net = PolicyNet(input_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adv_net.load_state_dict(checkpoint['advantage_net_state'])
    pol_net.load_state_dict(checkpoint['policy_net_state'])
    
    adv_net.eval()
    pol_net.eval()
    
    # Scenarios to test
    scenarios = [
        ("Preflop AA (Strong)", {'hole_cards': [(12, 0), (12, 1)], 'street': 0, 'pot': 150}),
        ("Preflop 72o (Weak)", {'hole_cards': [(5, 0), (0, 1)], 'street': 0, 'pot': 150}),
        ("River Nuts (Flush)", {
            'hole_cards': [(12, 0), (11, 0)], 
            'community_cards': [(10, 0), (8, 0), (4, 0), (2, 1), (9, 2)],
            'street': 3,
            'pot': 1000
        }),
    ]
    
    for name, setup in scenarios:
        print(f"\n--- Scenario: {name} ---")
        state = game.reset()
        
        # Manually inject state
        state.hole_cards[state.current_player] = setup['hole_cards']
        if 'community_cards' in setup:
            state.community_cards = setup['community_cards']
        state.street = setup['street']
        state.pot = setup['pot']
        
        # Get Encoding
        encoding = encoder.encode(state, state.current_player)
        tensor = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0)
        
        # 1. Advantage Net Output
        with torch.no_grad():
            advantages = adv_net(tensor).numpy()[0]
        
        print("Predicted Advantages (Scaled):")
        legal_actions = game.get_legal_actions(state)
        legal_indices = [a[0].value for a in legal_actions] # Assuming Action is IntEnum
        
        # Map advantages to action names
        # Action enum: FOLD=0, CHECK=1, CALL=2, BET=3, RAISE=4
        action_names = {0: "FOLD", 1: "CHECK", 2: "CALL", 3: "BET", 4: "RAISE"}
        
        for i, adv in enumerate(advantages):
            if i in action_names:
                marker = "*" if i in [a[0].value for a in legal_actions] else " "
                print(f"  {marker} {action_names[i]:<5}: {adv:.4f}")
                
        # 2. Policy Net Output
        with torch.no_grad():
            logits = pol_net(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        print("Predicted Policy (Probs):")
        for i, prob in enumerate(probs):
            if i in action_names:
                marker = "*" if i in [a[0].value for a in legal_actions] else " "
                print(f"  {marker} {action_names[i]:<5}: {prob:.4f}")

if __name__ == "__main__":
    with app.run():
        diagnose_policy.remote(25)


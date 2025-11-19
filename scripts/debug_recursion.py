import torch
import numpy as np
from poker_game.game import PokerGame, Action
from poker_game.state_encoder import StateEncoder
from models.advantage_net import AdvantageNet
from models.policy_net import PolicyNet
from training.deep_cfr import DeepCFR

def debug_recursion():
    print("Setting up debug environment...")
    game = PokerGame()
    state_encoder = StateEncoder()
    advantage_net = AdvantageNet(state_encoder.feature_dim)
    policy_net = PolicyNet(state_encoder.feature_dim)
    
    # Initialize Deep CFR with fixed code
    deep_cfr = DeepCFR(advantage_net, policy_net, state_encoder, game)
    
    # We want to detect if depth gets high, even if it doesn't crash
    print("Running 2000 iterations to hunt for deep games...")
    buffers = {'advantage': [], 'policy': []}
    
    deep_games = 0
    max_depth_seen = 0
    
    # Monkey patch traverse to track depth
    original_traverse = deep_cfr.traverse_external_sampling
    
    def tracking_traverse(state, player, buffers, depth=0, max_depth=1000):
        nonlocal max_depth_seen, deep_games
        max_depth_seen = max(max_depth_seen, depth)
        if depth == 50: # Only count once per deep branch
            deep_games += 1
            # Log the state causing deep recursion
            print(f"\n[ALERT] Depth {depth} reached!")
            print(f"  Street: {state.street}")
            print(f"  History: {state.betting_history[-5:]}")
            print(f"  Bets: {state.current_bets}")
        
        return original_traverse(state, player, buffers, depth, max_depth)
        
    deep_cfr.traverse_external_sampling = tracking_traverse
    
    for i in range(2000):
        state = game.reset()
        try:
            # We call the tracking version which calls original (which recursively calls tracking version because we assigned it to self)
            # Wait, deep_cfr.traverse... is a bound method. Assigning to it on instance works?
            # Yes, usually.
            deep_cfr.traverse_external_sampling(state, 0, buffers)
        except RecursionError:
            print(f"RecursionError at iteration {i}!")
            break
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            
    print(f"Max depth seen: {max_depth_seen}")
    print(f"Deep games (>50 depth): {deep_games}")

if __name__ == "__main__":
    debug_recursion()

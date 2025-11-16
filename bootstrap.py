"""Bootstrap scripts for initializing training."""

import torch
import numpy as np
from typing import Dict, List
import random

from poker_game.game import PokerGame, GameState, Action
from poker_game.state_encoder import StateEncoder
from poker_game.information_set import InformationSet, get_information_set
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR


class BaselineAgent:
    """Simple baseline agent for warm-start training."""
    
    def __init__(self, game: PokerGame):
        self.game = game
    
    def get_action(self, state: GameState) -> tuple:
        """Get action using simple heuristic strategy."""
        legal_actions = self.game.get_legal_actions(state)
        if len(legal_actions) == 0:
            return (Action.FOLD, 0)
        
        player = state.current_player
        hole_cards = state.hole_cards[player]
        
        # Simple tight-aggressive strategy
        # Evaluate hand strength
        hand_strength = self._evaluate_hand_strength(hole_cards, state.community_cards)
        
        # Betting logic
        to_call = state.current_bets[1 - player] - state.current_bets[player]
        
        if hand_strength > 0.7:
            # Strong hand - bet/raise
            if to_call == 0:
                # Can bet
                bet_actions = [a for a in legal_actions if a[0] in [Action.BET, Action.RAISE]]
                if bet_actions:
                    return random.choice(bet_actions)
            else:
                # Can raise
                raise_actions = [a for a in legal_actions if a[0] == Action.RAISE]
                if raise_actions:
                    return random.choice(raise_actions)
                # Or call
                call_actions = [a for a in legal_actions if a[0] == Action.CALL]
                if call_actions:
                    return call_actions[0]
        elif hand_strength > 0.4:
            # Medium hand - call/check
            if to_call == 0:
                check_actions = [a for a in legal_actions if a[0] == Action.CHECK]
                if check_actions:
                    return check_actions[0]
            else:
                call_actions = [a for a in legal_actions if a[0] == Action.CALL]
                if call_actions and to_call < state.stacks[player] * 0.1:
                    return call_actions[0]
                else:
                    return (Action.FOLD, 0)
        else:
            # Weak hand - fold or check
            if to_call == 0:
                check_actions = [a for a in legal_actions if a[0] == Action.CHECK]
                if check_actions:
                    return check_actions[0]
            return (Action.FOLD, 0)
        
        # Default: first legal action
        return legal_actions[0]
    
    def _evaluate_hand_strength(self, hole_cards: List, community_cards: List) -> float:
        """Evaluate hand strength (0-1)."""
        # Simple heuristic based on card ranks
        ranks = [card[0] for card in hole_cards]
        
        # Pair
        if ranks[0] == ranks[1]:
            return 0.6 + min(ranks[0], 12) / 12.0 * 0.3
        
        # High cards
        max_rank = max(ranks)
        if max_rank >= 10:  # Face cards
            return 0.4 + (max_rank - 10) / 2.0 * 0.2
        
        # Suited
        if hole_cards[0][1] == hole_cards[1][1]:
            return 0.3
        
        # Low cards
        return 0.2


def generate_baseline_trajectories(game: PokerGame, num_games: int = 1000) -> List[Dict]:
    """Generate trajectories using baseline agent."""
    agent = BaselineAgent(game)
    trajectories = []
    
    for _ in range(num_games):
        state = game.reset()
        trajectory = {
            'states': [],
            'actions': [],
            'info_sets': [],
            'payoffs': None,
            'player': None
        }
        
        current_player = state.current_player
        trajectory['player'] = current_player
        
        while not state.is_terminal:
            player = state.current_player
            info_set = get_information_set(state, player)
            legal_actions = game.get_legal_actions(state)
            
            if len(legal_actions) == 0:
                break
            
            # Get action from baseline agent
            action, amount = agent.get_action(state)
            
            # Find action index
            action_idx = 0
            for i, (a, amt) in enumerate(legal_actions):
                if a == action and abs(amt - amount) < 1:
                    action_idx = i
                    break
            
            trajectory['states'].append(state)
            trajectory['actions'].append((action_idx, action, amount))
            trajectory['info_sets'].append(info_set)
            
            state = game.apply_action(state, action, amount)
        
        payoffs = game.get_payoff(state)
        trajectory['payoffs'] = payoffs
        trajectories.append(trajectory)
    
    return trajectories


def warm_start_network(network: ValuePolicyNet, 
                      state_encoder: StateEncoder,
                      game: PokerGame,
                      num_trajectories: int = 5000,
                      device: str = 'cpu'):
    """Warm-start network by training on baseline agent trajectories."""
    print(f"Warm-starting network with {num_trajectories} baseline trajectories...")
    
    # Generate baseline trajectories
    trajectories = generate_baseline_trajectories(game, num_trajectories)
    
    # Prepare training data
    states = []
    values = []
    policies = []
    
    for trajectory in trajectories:
        for i, state in enumerate(trajectory['states']):
            player = trajectory['player']
            state_encoding = state_encoder.encode(state, player)
            states.append(state_encoding)
            
            # Use final payoff as value estimate
            payoff = trajectory['payoffs'][player]
            values.append(payoff)
            
            # Use baseline action as policy target
            if i < len(trajectory['actions']):
                action_idx = trajectory['actions'][i][0]
                policy = np.zeros(network.max_actions)
                policy[action_idx] = 1.0
            else:
                policy = np.ones(network.max_actions) / network.max_actions
            policies.append(policy)
    
    # Train network
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    criterion_value = torch.nn.MSELoss()
    criterion_policy = torch.nn.CrossEntropyLoss()
    
    batch_size = 32
    num_epochs = 10
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(states))
        total_loss = 0.0
        
        for i in range(0, len(states), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = torch.tensor([states[j] for j in batch_indices], dtype=torch.float32).to(device)
            batch_values = torch.tensor([values[j] for j in batch_indices], dtype=torch.float32).to(device).unsqueeze(1)
            batch_policies = torch.tensor([policies[j] for j in batch_indices], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            pred_values, pred_policy_logits = network(batch_states)
            
            value_loss = criterion_value(pred_values, batch_values)
            policy_loss = criterion_policy(pred_policy_logits, batch_policies.argmax(dim=1))
            
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (len(states) / batch_size):.4f}")
    
    print("Warm-start complete!")


def create_multi_agent_pool(game: PokerGame,
                            state_encoder: StateEncoder,
                            num_agents: int = 5,
                            device: str = 'cpu') -> List[ValuePolicyNet]:
    """Create a pool of agents with different initializations."""
    agent_pool = []
    input_dim = state_encoder.feature_dim
    
    for i in range(num_agents):
        # Create agent with different random initialization
        agent = ValuePolicyNet(input_dim=input_dim)
        
        # Vary initialization scale
        for param in agent.parameters():
            param.data.normal_(0, 0.1 * (1 + i * 0.1))
        
        agent.to(device)
        agent_pool.append(agent)
    
    return agent_pool


if __name__ == '__main__':
    # Example: Warm-start a network
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    input_dim = state_encoder.feature_dim
    
    value_net = ValuePolicyNet(input_dim=input_dim)
    policy_net = ValuePolicyNet(input_dim=input_dim)
    
    print("Warm-starting value network...")
    warm_start_network(value_net, state_encoder, game, num_trajectories=1000)
    
    print("Warm-starting policy network...")
    warm_start_network(policy_net, state_encoder, game, num_trajectories=1000)
    
    print("Bootstrap complete! Networks are ready for Deep CFR training.")


"""Main training loop for Deep CFR."""

import torch
import numpy as np
from typing import List, Optional, Dict
from tqdm import tqdm

from poker_game.game import PokerGame
from poker_game.state_encoder import StateEncoder
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR
from training.self_play import SelfPlayGenerator


class Trainer:
    """Main trainer for Deep CFR poker bot."""
    
    def __init__(self,
                 game: PokerGame,
                 state_encoder: StateEncoder,
                 value_net: ValuePolicyNet,
                 policy_net: ValuePolicyNet,
                 device: str = 'cpu',
                 learning_rate: float = 1e-4,
                 trajectories_per_iteration: int = 1000,
                 network_update_frequency: int = 10):
        self.game = game
        self.state_encoder = state_encoder
        self.device = device
        
        # Initialize Deep CFR
        self.deep_cfr = DeepCFR(
            value_net=value_net,
            policy_net=policy_net,
            state_encoder=state_encoder,
            game=game,
            learning_rate=learning_rate,
            device=device
        )
        
        # Training parameters
        self.trajectories_per_iteration = trajectories_per_iteration
        self.network_update_frequency = network_update_frequency
        
        # Training state
        self.iteration = 0
        self.value_buffer = []
        self.policy_buffer = []
    
    def train_iteration(self):
        """Run one training iteration."""
        self.iteration += 1
        
        # Generate trajectories
        generator = SelfPlayGenerator(
            self.game,
            self.deep_cfr,
            num_trajectories=self.trajectories_per_iteration
        )
        
        trajectories = generator.generate_trajectories()
        
        # Process trajectories and update regrets
        for trajectory in trajectories:
            self._process_trajectory(trajectory)
        
        # Update networks periodically
        if self.iteration % self.network_update_frequency == 0:
            self._update_networks_from_buffers()
        
        return {
            'iteration': self.iteration,
            'num_trajectories': len(trajectories),
            'value_buffer_size': len(self.value_buffer),
            'policy_buffer_size': len(self.policy_buffer)
        }
    
    def _process_trajectory(self, trajectory: Dict):
        """Process a trajectory and update training buffers."""
        states = trajectory['states']
        info_sets = trajectory['info_sets']
        actions = trajectory['actions']
        payoffs = trajectory['payoffs']
        player = trajectory['player']
        
        # Compute counterfactual values backwards through trajectory
        for i in range(len(states) - 1, -1, -1):
            state = states[i]
            info_set = info_sets[i]
            
            # Encode state
            state_encoding = self.state_encoder.encode(state, player)
            
            # Get counterfactual value from memory or network
            cf_value = self.deep_cfr.counterfactual_values.get(info_set.key, 0.0)
            
            # If not in memory, use network prediction
            if cf_value == 0.0:
                state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    predicted_value, _ = self.deep_cfr.value_net(state_tensor)
                    cf_value = predicted_value.item()
            
            # Add to value buffer
            self.value_buffer.append((state_encoding, cf_value))
            
            # Get strategy for policy buffer
            legal_actions = self.game.get_legal_actions(state)
            strategy = self.deep_cfr.get_strategy(info_set, legal_actions)
            
            # Convert strategy to probability vector
            max_actions = self.deep_cfr.policy_net.max_actions
            action_probs = np.zeros(max_actions)
            for action_idx, prob in strategy.items():
                if action_idx < max_actions:
                    action_probs[action_idx] = prob
            
            # Normalize
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            
            self.policy_buffer.append((state_encoding, action_probs))
    
    def _update_networks_from_buffers(self, batch_size: int = 32):
        """Update networks from accumulated buffers."""
        if len(self.value_buffer) == 0 and len(self.policy_buffer) == 0:
            return
        
        # Update value network
        if len(self.value_buffer) >= batch_size:
            num_batches = min(100, len(self.value_buffer) // batch_size)  # Limit batches per update
            for _ in range(num_batches):
                indices = np.random.choice(len(self.value_buffer), batch_size, replace=False)
                batch_states = [self.value_buffer[i][0] for i in indices]
                batch_values = [self.value_buffer[i][1] for i in indices]
                
                states_tensor = torch.tensor(np.stack(batch_states), dtype=torch.float32).to(self.device)
                values_tensor = torch.tensor(batch_values, dtype=torch.float32).to(self.device).unsqueeze(1)
                
                self.deep_cfr.value_optimizer.zero_grad()
                predicted_values, _ = self.deep_cfr.value_net(states_tensor)
                value_loss = torch.nn.MSELoss()(predicted_values, values_tensor)
                value_loss.backward()
                self.deep_cfr.value_optimizer.step()
        
        # Update policy network
        if len(self.policy_buffer) >= batch_size:
            num_batches = min(100, len(self.policy_buffer) // batch_size)
            for _ in range(num_batches):
                indices = np.random.choice(len(self.policy_buffer), batch_size, replace=False)
                batch_states = [self.policy_buffer[i][0] for i in indices]
                batch_probs = [self.policy_buffer[i][1] for i in indices]
                
                states_tensor = torch.tensor(np.stack(batch_states), dtype=torch.float32).to(self.device)
                probs_tensor = torch.tensor(np.stack(batch_probs), dtype=torch.float32).to(self.device)
                
                self.deep_cfr.policy_optimizer.zero_grad()
                _, policy_logits = self.deep_cfr.policy_net(states_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1)
                
                # KL divergence loss
                kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(
                    torch.log(policy_probs + 1e-8), probs_tensor
                )
                kl_loss.backward()
                self.deep_cfr.policy_optimizer.step()
        
        # Clear buffers periodically
        if len(self.value_buffer) > 10000:
            self.value_buffer = self.value_buffer[-5000:]
        if len(self.policy_buffer) > 10000:
            self.policy_buffer = self.policy_buffer[-5000:]
    
    def train(self, num_iterations: int, checkpoint_callback: Optional[callable] = None):
        """Train for multiple iterations."""
        for iteration in tqdm(range(num_iterations), desc="Training"):
            stats = self.train_iteration()
            
            if checkpoint_callback and (iteration + 1) % 100 == 0:
                checkpoint_callback(self, iteration + 1)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: {stats}")
    
    def get_training_state(self) -> Dict:
        """Get current training state for checkpointing."""
        return {
            'iteration': self.iteration,
            'value_net_state': self.deep_cfr.value_net.state_dict(),
            'policy_net_state': self.deep_cfr.policy_net.state_dict(),
            'value_optimizer_state': self.deep_cfr.value_optimizer.state_dict(),
            'policy_optimizer_state': self.deep_cfr.policy_optimizer.state_dict(),
            'regret_memory': dict(self.deep_cfr.regret_memory),
            'strategy_memory': dict(self.deep_cfr.strategy_memory),
            'counterfactual_values': dict(self.deep_cfr.counterfactual_values)
        }
    
    def load_training_state(self, state: Dict):
        """Load training state from checkpoint."""
        self.iteration = state['iteration']
        self.deep_cfr.value_net.load_state_dict(state['value_net_state'])
        self.deep_cfr.policy_net.load_state_dict(state['policy_net_state'])
        self.deep_cfr.value_optimizer.load_state_dict(state['value_optimizer_state'])
        self.deep_cfr.policy_optimizer.load_state_dict(state['policy_optimizer_state'])
        self.deep_cfr.regret_memory = defaultdict(lambda: defaultdict(float), state['regret_memory'])
        self.deep_cfr.strategy_memory = defaultdict(lambda: defaultdict(float), state['strategy_memory'])
        self.deep_cfr.counterfactual_values = defaultdict(float, state['counterfactual_values'])


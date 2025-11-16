"""Modal functions for distributed training."""

import modal
import torch
import pickle
import os
from typing import List, Dict, Any
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modal.config import image, checkpoint_volume, GPU_CONFIG, CPU_CONFIG, TRAINING_CONFIG
from poker_game.game import PokerGame
from poker_game.state_encoder import StateEncoder
from poker_game.information_set import InformationSet, get_information_set
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR
from training.self_play import SelfPlayGenerator

app = modal.App("poker-bot-training")


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    **CPU_CONFIG,
    timeout=3600,
)
def generate_trajectories(
    num_trajectories: int,
    checkpoint_path: str = None,
    iteration: int = 0
) -> List[Dict]:
    """Generate self-play trajectories (CPU workers)."""
    # Initialize game
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    
    # Load or create networks
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        # Load checkpoint
        checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location='cpu')
        input_dim = state_encoder.feature_dim
        value_net = ValuePolicyNet(input_dim=input_dim)
        policy_net = ValuePolicyNet(input_dim=input_dim)
        value_net.load_state_dict(checkpoint['value_net_state'])
        policy_net.load_state_dict(checkpoint['policy_net_state'])
    else:
        # Create new networks
        input_dim = state_encoder.feature_dim
        value_net = ValuePolicyNet(input_dim=input_dim)
        policy_net = ValuePolicyNet(input_dim=input_dim)
    
    # Initialize Deep CFR
    deep_cfr = DeepCFR(
        value_net=value_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        device='cpu'
    )
    
    # Load regret memory if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location='cpu')
        if 'regret_memory' in checkpoint:
            deep_cfr.regret_memory = checkpoint['regret_memory']
    
    # Generate trajectories
    generator = SelfPlayGenerator(game, deep_cfr, num_trajectories=num_trajectories)
    trajectories = generator.generate_trajectories()
    
    return trajectories


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    **GPU_CONFIG,
    timeout=7200,
)
def train_networks(
    trajectories: List[Dict],
    checkpoint_path: str = None,
    iteration: int = 0,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Train networks on trajectories (GPU workers)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize components
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    input_dim = state_encoder.feature_dim
    
    # Load or create networks
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location=device)
        value_net = ValuePolicyNet(input_dim=input_dim).to(device)
        policy_net = ValuePolicyNet(input_dim=input_dim).to(device)
        value_net.load_state_dict(checkpoint['value_net_state'])
        policy_net.load_state_dict(checkpoint['policy_net_state'])
    else:
        value_net = ValuePolicyNet(input_dim=input_dim).to(device)
        policy_net = ValuePolicyNet(input_dim=input_dim).to(device)
    
    # Initialize Deep CFR
    deep_cfr = DeepCFR(
        value_net=value_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        device=device
    )
    
    # Load training state if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location=device)
        if 'value_optimizer_state' in checkpoint:
            deep_cfr.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        if 'policy_optimizer_state' in checkpoint:
            deep_cfr.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        if 'regret_memory' in checkpoint:
            deep_cfr.regret_memory = checkpoint['regret_memory']
        if 'counterfactual_values' in checkpoint:
            deep_cfr.counterfactual_values = checkpoint['counterfactual_values']
    
    # Process trajectories and update networks
    value_buffer = []
    policy_buffer = []
    
    for trajectory in trajectories:
        states = trajectory['states']
        info_sets = trajectory['info_sets']
        payoffs = trajectory['payoffs']
        player = trajectory['player']
        
        # Process trajectory backwards
        for i in range(len(states) - 1, -1, -1):
            state = states[i]
            info_set = info_sets[i]
            
            # Encode state
            state_encoding = state_encoder.encode(state, player)
            
            # Get counterfactual value
            cf_value = deep_cfr.counterfactual_values.get(info_set.key, 0.0)
            if cf_value == 0.0:
                state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted_value, _ = value_net(state_tensor)
                    cf_value = predicted_value.item()
            
            value_buffer.append((state_encoding, cf_value))
            
            # Get strategy
            legal_actions = game.get_legal_actions(state)
            strategy = deep_cfr.get_strategy(info_set, legal_actions)
            
            # Convert to probability vector
            max_actions = policy_net.max_actions
            action_probs = [0.0] * max_actions
            for action_idx, prob in strategy.items():
                if action_idx < max_actions:
                    action_probs[action_idx] = prob
            
            total = sum(action_probs)
            if total > 0:
                action_probs = [p / total for p in action_probs]
            
            policy_buffer.append((state_encoding, action_probs))
    
    # Train networks
    num_updates = min(100, len(value_buffer) // batch_size)
    value_losses = []
    policy_losses = []
    
    for _ in range(num_updates):
        # Update value network
        if len(value_buffer) >= batch_size:
            indices = torch.randint(0, len(value_buffer), (batch_size,))
            batch_states = torch.tensor([value_buffer[i][0] for i in indices], dtype=torch.float32).to(device)
            batch_values = torch.tensor([value_buffer[i][1] for i in indices], dtype=torch.float32).to(device).unsqueeze(1)
            
            deep_cfr.value_optimizer.zero_grad()
            predicted_values, _ = value_net(batch_states)
            value_loss = torch.nn.MSELoss()(predicted_values, batch_values)
            value_loss.backward()
            deep_cfr.value_optimizer.step()
            value_losses.append(value_loss.item())
        
        # Update policy network
        if len(policy_buffer) >= batch_size:
            indices = torch.randint(0, len(policy_buffer), (batch_size,))
            batch_states = torch.tensor([policy_buffer[i][0] for i in indices], dtype=torch.float32).to(device)
            batch_probs = torch.tensor([policy_buffer[i][1] for i in indices], dtype=torch.float32).to(device)
            
            deep_cfr.policy_optimizer.zero_grad()
            _, policy_logits = policy_net(batch_states)
            policy_probs = torch.softmax(policy_logits, dim=1)
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(
                torch.log(policy_probs + 1e-8), batch_probs
            )
            kl_loss.backward()
            deep_cfr.policy_optimizer.step()
            policy_losses.append(kl_loss.item())
    
    # Prepare updated checkpoint
    checkpoint = {
        'iteration': iteration,
        'value_net_state': value_net.state_dict(),
        'policy_net_state': policy_net.state_dict(),
        'value_optimizer_state': deep_cfr.value_optimizer.state_dict(),
        'policy_optimizer_state': deep_cfr.policy_optimizer.state_dict(),
        'regret_memory': dict(deep_cfr.regret_memory),
        'counterfactual_values': dict(deep_cfr.counterfactual_values),
    }
    
    # Save checkpoint
    checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    torch.save(checkpoint, checkpoint_file)
    checkpoint_volume.commit()
    
    return {
        'iteration': iteration,
        'value_loss': sum(value_losses) / len(value_losses) if value_losses else 0.0,
        'policy_loss': sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
        'checkpoint_path': checkpoint_file
    }


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    **CPU_CONFIG,
    timeout=1800,
)
def evaluate_agents(
    checkpoint_paths: List[str],
    num_games: int = 100
) -> Dict[str, Any]:
    """Evaluate agents head-to-head."""
    results = {}
    
    # Load agents
    agents = {}
    for path in checkpoint_paths:
        if os.path.exists(f"/checkpoints/{path}"):
            checkpoint = torch.load(f"/checkpoints/{path}", map_location='cpu')
            # Create agent from checkpoint
            agents[path] = checkpoint
    
    # Run head-to-head matches
    for i, path1 in enumerate(checkpoint_paths):
        for j, path2 in enumerate(checkpoint_paths):
            if i >= j:
                continue
            
            # Play games
            wins = [0, 0]
            total_payoff = [0.0, 0.0]
            
            # Simplified evaluation - would need full agent implementation
            # For now, return placeholder results
            results[f"{path1}_vs_{path2}"] = {
                'wins': wins,
                'payoffs': total_payoff,
                'win_rate': [w / num_games for w in wins]
            }
    
    return results


@app.local_entrypoint()
def main(
    num_iterations: int = 1000,
    trajectories_per_iteration: int = 10000,
    num_workers: int = 4,
    resume_from: int = None
):
    """Main training loop."""
    print(f"Starting training: {num_iterations} iterations, {trajectories_per_iteration} trajectories/iter")
    
    start_iteration = resume_from if resume_from else 0
    
    for iteration in range(start_iteration, num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # Determine checkpoint path
        checkpoint_path = None
        if iteration > 0:
            checkpoint_path = f"checkpoint_iter_{iteration - 1}.pt"
        
        # Generate trajectories in parallel
        trajectories_per_worker = trajectories_per_iteration // num_workers
        trajectory_futures = []
        for worker_id in range(num_workers):
            future = generate_trajectories.spawn(
                num_trajectories=trajectories_per_worker,
                checkpoint_path=checkpoint_path,
                iteration=iteration
            )
            trajectory_futures.append(future)
        
        # Collect trajectories
        all_trajectories = []
        for future in trajectory_futures:
            trajectories = future.get()
            all_trajectories.extend(trajectories)
        
        print(f"Generated {len(all_trajectories)} trajectories")
        
        # Train networks
        train_result = train_networks.spawn(
            trajectories=all_trajectories,
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            batch_size=TRAINING_CONFIG['batch_size']
        ).get()
        
        print(f"Training complete: value_loss={train_result['value_loss']:.4f}, "
              f"policy_loss={train_result['policy_loss']:.4f}")
        
        # Checkpoint periodically
        if (iteration + 1) % TRAINING_CONFIG['checkpoint_frequency'] == 0:
            print(f"Checkpoint saved at iteration {iteration + 1}")
    
    print("Training complete!")


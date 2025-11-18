"""Modal functions for distributed training."""

import modal
import torch
import os
from typing import List, Dict, Any, Tuple
import sys
import collections
import random
import pickle

# Add parent directory to path for local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modal_deploy.config import image, checkpoint_volume, GPU_CONFIG, CPU_CONFIG, TRAINING_CONFIG

app = modal.App("poker-bot-training")


class ReplayBuffer:
    """Experience replay buffer for stable training."""
    def __init__(self, maxlen=200000):
        self.buffer = collections.deque(maxlen=maxlen)
    
    def add(self, item):
        self.buffer.append(item)
        
    def extend(self, items):
        self.buffer.extend(items)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60,
)
def setup_metrics_dir():
    """Initialize metrics directory."""
    import os
    os.makedirs("/checkpoints/metrics", exist_ok=True)
    os.makedirs("/checkpoints/logs", exist_ok=True)
    checkpoint_volume.commit()


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
    """Generate self-play trajectories (CPU workers).

    Uses the current CFR strategy derived from regret_memory for self-play.
    Average strategy (strategy_memory) is not required here, but regret_memory
    is loaded from checkpoints so that self-play reflects accumulated learning.
    """
    from collections import defaultdict
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet
    from training.deep_cfr import DeepCFR
    from training.self_play import SelfPlayGenerator
    
    # Initialize game
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    
    # Load or create networks
    input_dim = state_encoder.feature_dim
    value_net = ValuePolicyNet(input_dim=input_dim)
    policy_net = ValuePolicyNet(input_dim=input_dim)
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location='cpu', weights_only=False)
            # Validate checkpoint has required keys
            required_keys = ['value_net_state', 'policy_net_state']
            if all(key in checkpoint for key in required_keys):
                value_net.load_state_dict(checkpoint['value_net_state'])
                policy_net.load_state_dict(checkpoint['policy_net_state'])
            else:
                import logging
                logging.warning(f"Checkpoint {checkpoint_path} missing required keys. Using new networks.")
        except Exception as e:
            import logging
            logging.warning(f"Failed to load checkpoint {checkpoint_path}: {e}. Using new networks.")
    
    # Initialize Deep CFR with reduced value learning rate for stability
    # and a slightly lower policy learning rate for smoother policy updates.
    deep_cfr = DeepCFR(
        value_net=value_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        learning_rate=5e-5,
        value_learning_rate=5e-5,  # Lower LR for value network to prevent instability
        device='cpu'
    )
    
    # Load regret / strategy memory if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location='cpu', weights_only=False)
            if 'regret_memory' in checkpoint:
                # Convert dict back to defaultdict if needed
                from collections import defaultdict
                if isinstance(checkpoint['regret_memory'], dict):
                    deep_cfr.regret_memory = defaultdict(lambda: defaultdict(float), 
                        {k: defaultdict(float, v) if isinstance(v, dict) else v 
                         for k, v in checkpoint['regret_memory'].items()})
                else:
                    deep_cfr.regret_memory = checkpoint['regret_memory']
            if 'strategy_memory' in checkpoint:
                from collections import defaultdict
                if isinstance(checkpoint['strategy_memory'], dict):
                    deep_cfr.strategy_memory = defaultdict(
                        lambda: defaultdict(float),
                        {
                            k: defaultdict(float, v) if isinstance(v, dict) else v
                            for k, v in checkpoint['strategy_memory'].items()
                        },
                    )
                else:
                    deep_cfr.strategy_memory = checkpoint['strategy_memory']
        except Exception as e:
            import logging
            logging.warning(f"Failed to load regret memory from {checkpoint_path}: {e}. Using empty memory.")
    
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
    import numpy as np
    from collections import defaultdict
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from poker_game.information_set import get_information_set
    from models.value_policy_net import ValuePolicyNet
    from training.deep_cfr import DeepCFR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Scale CFVs to keep value targets numerically sane.
    # We use starting stack (20,000 chips) as the scale, so CFVs are O(1).
    CFV_SCALE = 20000.0
    
    # Enable mixed precision training for 2x GPU speedup
    use_amp = device == 'cuda'
    scaler = None
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
    
    # Log GPU availability
    import logging
    import sys
    # Set up logging to ensure it goes to stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    if device == 'cpu' and torch.cuda.is_available() == False:
        logger.warning("GPU not available, falling back to CPU. Training will be slower.")
    elif device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if use_amp:
            logger.info("Mixed precision training (FP16) enabled for 2x speedup")
    
    # Initialize components
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    input_dim = state_encoder.feature_dim
    
    # Load or create networks
    value_net = ValuePolicyNet(input_dim=input_dim).to(device)
    policy_net = ValuePolicyNet(input_dim=input_dim).to(device)
    
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location=device, weights_only=False)
            # Validate checkpoint has required keys
            required_keys = ['value_net_state', 'policy_net_state']
            if all(key in checkpoint for key in required_keys):
                value_net.load_state_dict(checkpoint['value_net_state'])
                policy_net.load_state_dict(checkpoint['policy_net_state'])
                logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            else:
                logging.warning(f"Checkpoint {checkpoint_path} missing required keys. Using new networks.")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint {checkpoint_path}: {e}. Using new networks.")
    
    # Initialize Deep CFR with reduced value learning rate for stability
    # and a slightly lower policy learning rate for smoother policy updates.
    deep_cfr = DeepCFR(
        value_net=value_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        learning_rate=5e-5,
        value_learning_rate=5e-5,  # Lower LR for value network to prevent instability
        device=device
    )
    
    # Load training state if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location=device, weights_only=False)
            if 'value_optimizer_state' in checkpoint:
                try:
                    deep_cfr.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
                except Exception as e:
                    logging.warning(f"Failed to load value optimizer state: {e}")
            if 'policy_optimizer_state' in checkpoint:
                try:
                    deep_cfr.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
                except Exception as e:
                    logging.warning(f"Failed to load policy optimizer state: {e}")
            if 'regret_memory' in checkpoint:
                try:
                    # Convert dict back to defaultdict if needed
                    from collections import defaultdict
                    if isinstance(checkpoint['regret_memory'], dict):
                        deep_cfr.regret_memory = defaultdict(lambda: defaultdict(float), 
                            {k: defaultdict(float, v) if isinstance(v, dict) else v 
                             for k, v in checkpoint['regret_memory'].items()})
                    else:
                        deep_cfr.regret_memory = checkpoint['regret_memory']
                except Exception as e:
                    logging.warning(f"Failed to load regret memory: {e}")
            if 'counterfactual_values' in checkpoint:
                try:
                    if isinstance(checkpoint['counterfactual_values'], dict):
                        deep_cfr.counterfactual_values = defaultdict(float, checkpoint['counterfactual_values'])
                    else:
                        deep_cfr.counterfactual_values = checkpoint['counterfactual_values']
                except Exception as e:
                    logging.warning(f"Failed to load counterfactual values: {e}")
            if 'strategy_memory' in checkpoint:
                try:
                    from collections import defaultdict
                    if isinstance(checkpoint['strategy_memory'], dict):
                        deep_cfr.strategy_memory = defaultdict(
                            lambda: defaultdict(float),
                            {
                                k: defaultdict(float, v) if isinstance(v, dict) else v
                                for k, v in checkpoint['strategy_memory'].items()
                            },
                        )
                    else:
                        deep_cfr.strategy_memory = checkpoint['strategy_memory']
                except Exception as e:
                    logging.warning(f"Failed to load strategy memory: {e}")
        except Exception as e:
            logging.warning(f"Failed to load training state from {checkpoint_path}: {e}")
    
    # Process trajectories and update networks
    value_buffer = []
    policy_buffer = []
    
    # Validate trajectory structure
    def validate_trajectory(traj):
        """Validate trajectory has required structure."""
        required_keys = ['states', 'info_sets', 'payoffs', 'player']
        if not isinstance(traj, dict):
            return False
        if not all(key in traj for key in required_keys):
            return False
        if not isinstance(traj['states'], list) or len(traj['states']) == 0:
            return False
        if not isinstance(traj['info_sets'], list) or len(traj['info_sets']) != len(traj['states']):
            return False
        if not isinstance(traj['payoffs'], (list, tuple)) or len(traj['payoffs']) != 2:
            return False
        if traj['player'] not in [0, 1]:
            return False
        return True
    
    valid_trajectories = []
    invalid_count = 0
    for trajectory in trajectories:
        if validate_trajectory(trajectory):
            valid_trajectories.append(trajectory)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"Skipped {invalid_count} invalid trajectories out of {len(trajectories)}")
    
    if len(valid_trajectories) == 0:
        logger.error("No valid trajectories to process!")
        return {
            'iteration': iteration,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'checkpoint_path': '',
            'num_updates': 0,
            'error': 'no_valid_trajectories'
        }
    
    # Logging for trajectory processing
    total_decision_points = 0
    total_regrets_updated = 0
    total_cf_values_computed = 0
    
    for traj_idx, trajectory in enumerate(valid_trajectories):
        states = trajectory['states']
        info_sets = trajectory['info_sets']
        actions = trajectory['actions']  # List of (action_idx, action, amount)
        payoffs = trajectory['payoffs']
        trajectory_player = trajectory['player']
        
        if traj_idx == 0:
            logger.info(f"Processing trajectory {traj_idx + 1}/{len(valid_trajectories)}:")
            logger.info(f"  States: {len(states)}, Actions: {len(actions)}, Player: {trajectory_player}")
            logger.info(f"  Payoffs: {payoffs}")
            logger.info(f"  Last state is_terminal: {states[-1].is_terminal if len(states) > 0 else 'N/A'}")
        
        # Process trajectory backwards using Monte Carlo CFR
        # Track counterfactual values backwards through the trajectory
        cf_values_after_state = {}  # Maps state index -> counterfactual value after this state
        
        # Start from terminal state
        if len(states) > 0:
            terminal_state = states[-1]
            if terminal_state.is_terminal:
                # Terminal payoff for the trajectory player (scale to keep values O(1))
                terminal_cf_value = payoffs[trajectory_player] / CFV_SCALE
                cf_values_after_state[len(states) - 1] = terminal_cf_value
                if traj_idx == 0:
                    logger.info(f"  Terminal CF value (scaled): {terminal_cf_value:.4f}")
            else:
                # If last state isn't terminal, use payoffs directly (scaled)
                # This can happen if trajectory ended without storing terminal state
                terminal_cf_value = payoffs[trajectory_player] / CFV_SCALE
                cf_values_after_state[len(states) - 1] = terminal_cf_value
                if traj_idx < 5:  # Only log first few to avoid spam
                    logger.warning(f"Trajectory {traj_idx} last state not marked as terminal, using scaled payoffs directly")
        
        # Process backwards through trajectory
        # Skip terminal state (last state) - start from second-to-last
        start_idx = len(states) - 2 if len(states) > 1 else -1
        for i in range(start_idx, -1, -1):
            state = states[i]
            info_set = info_sets[i]
            
            # Skip terminal states - they have no legal actions
            if state.is_terminal:
                continue
            
            current_player = state.current_player
            if current_player is None:
                continue
            
            # Encode state
            state_encoding = state_encoder.encode(state, trajectory_player)
            
            # Get legal actions and strategy
            legal_actions = game.get_legal_actions(state)
            if len(legal_actions) == 0:
                continue
            
            strategy = deep_cfr.get_strategy(info_set, legal_actions)
            
            # Validate strategy indices are within bounds
            max_action_idx = len(legal_actions) - 1
            valid_strategy = {idx: prob for idx, prob in strategy.items() if 0 <= idx <= max_action_idx}
            if len(valid_strategy) == 0:
                logger.warning(f"Skipping state: no valid strategy indices (legal_actions={len(legal_actions)}, strategy_keys={list(strategy.keys())})")
                continue
            
            # Compute counterfactual value for this state
            # If this is the trajectory player's decision point
            if current_player == trajectory_player:
                total_decision_points += 1
                
                # Get the action that was actually taken
                if i < len(actions):
                    _, action_taken, amount_taken = actions[i]
                    # Find the matching action in current legal_actions
                    # This handles cases where legal_actions changed between generation and training
                    action_idx_taken = None
                    for idx, (action, amount) in enumerate(legal_actions):
                        if action == action_taken and amount == amount_taken:
                            action_idx_taken = idx
                            break
                    
                    if action_idx_taken is None:
                        # Action not found in current legal_actions - this can happen if state changed
                        # Skip this state (it's a rare edge case)
                        if traj_idx < 5:  # Only log first few to avoid spam
                            logger.warning(f"Skipping state: action ({action_taken}, {amount_taken}) not found in legal_actions (len={len(legal_actions)})")
                        continue
                else:
                    # Terminal state, use payoff (already scaled)
                    cf_value = cf_values_after_state.get(i + 1, payoffs[trajectory_player] / CFV_SCALE)
                    # Clip in scaled space
                    cf_value = max(-10000.0 / CFV_SCALE, min(10000.0 / CFV_SCALE, cf_value))
                    deep_cfr.counterfactual_values[info_set.key] = cf_value
                    value_buffer.append((state_encoding, cf_value))
                    total_cf_values_computed += 1
                    continue
                
                # Get counterfactual value after taking this action
                cf_value_after_action = cf_values_after_state.get(i + 1, 0.0)
                
                # Compute node counterfactual value (weighted sum over all actions)
                node_cf_value = 0.0
                strategy_sum = 0.0
                for action_idx, prob in valid_strategy.items():
                    strategy_sum += prob
                    if action_idx == action_idx_taken:
                        # Use actual outcome
                        node_cf_value += prob * cf_value_after_action
                    else:
                        # For other actions, estimate using network or stored value
                        # Create hypothetical next state
                        # Double-check bounds (should be safe after validation above)
                        if action_idx < 0 or action_idx >= len(legal_actions):
                            logger.warning(f"Skipping action_idx={action_idx}: out of bounds (legal_actions={len(legal_actions)})")
                            continue
                        action, amount = legal_actions[action_idx]
                        next_state = game.apply_action(state, action, amount)
                        
                        # Get counterfactual value for this action
                        if next_state.is_terminal:
                            # Terminal payoff in scaled space
                            next_payoffs = game.get_payoff(next_state)
                            other_action_cf_value = next_payoffs[trajectory_player] / CFV_SCALE
                        else:
                            # Use network prediction or stored value (already scaled)
                            next_info_set = get_information_set(next_state, trajectory_player)
                            if next_info_set.key in deep_cfr.counterfactual_values:
                                other_action_cf_value = deep_cfr.counterfactual_values[next_info_set.key]
                            else:
                                # Predict using network if not in memory
                                next_state_encoding = state_encoder.encode(next_state, trajectory_player)
                                next_state_tensor = torch.tensor(next_state_encoding, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    predicted_value, _ = value_net(next_state_tensor, clip_value=True)
                                    other_action_cf_value = predicted_value.item()
                                    # Clip in scaled space
                                    other_action_cf_value = max(-10000.0 / CFV_SCALE, min(10000.0 / CFV_SCALE, other_action_cf_value))
                        
                        node_cf_value += prob * other_action_cf_value
                
                # Validate strategy sums to 1
                if abs(strategy_sum - 1.0) > 0.01:
                    logger.warning(f"Strategy doesn't sum to 1: {strategy_sum} (info_set: {info_set.key[:30]}...)")
                
                # Update regrets for this information set
                key = info_set.key
                regret = cf_value_after_action - node_cf_value
                old_regret = deep_cfr.regret_memory[key].get(action_idx_taken, 0.0)
                deep_cfr.regret_memory[key][action_idx_taken] += regret
                new_regret = deep_cfr.regret_memory[key][action_idx_taken]
                total_regrets_updated += 1
                
                # Log first few regret updates for debugging
                if total_regrets_updated <= 5:
                    logger.info(f"  Regret update #{total_regrets_updated}:")
                    logger.info(f"    Info set: {key[:40]}...")
                    logger.info(f"    Action: {action_idx_taken}, CF after action: {cf_value_after_action:.2f}, Node CF: {node_cf_value:.2f}")
                    logger.info(f"    Regret: {regret:.2f}, Old regret: {old_regret:.2f}, New regret: {new_regret:.2f}")
                
                # Clip regrets to prevent unbounded growth
                for action_idx in deep_cfr.regret_memory[key]:
                    deep_cfr.regret_memory[key][action_idx] = max(
                        -100000.0, min(100000.0, deep_cfr.regret_memory[key][action_idx])
                    )
                
                # Store counterfactual value for this state (scaled)
                cf_value = max(-10000.0 / CFV_SCALE, min(10000.0 / CFV_SCALE, node_cf_value))
                deep_cfr.counterfactual_values[key] = cf_value
                cf_values_after_state[i] = cf_value
                total_cf_values_computed += 1
                
                # Add to value buffer
                value_buffer.append((state_encoding, cf_value))
                
                # --- NEW: accumulate strategy for average strategy computation ---
                for a_idx, prob in valid_strategy.items():
                    deep_cfr.strategy_memory[key][a_idx] += prob
                
                # Convert average strategy to probability vector for policy training
                max_actions = policy_net.max_actions
                action_probs = [0.0] * max_actions
                avg_strategy = deep_cfr.compute_average_strategy(info_set, legal_actions)
                for action_idx, prob in avg_strategy.items():
                    if action_idx < max_actions:
                        action_probs[action_idx] = prob
                
                total = sum(action_probs)
                if total > 0:
                    action_probs = [p / total for p in action_probs]
                else:
                    # Uniform if no strategy
                    num_legal = min(max_actions, len(legal_actions))
                    action_probs = [1.0 / num_legal] * num_legal + [0.0] * (max_actions - num_legal)
                
                policy_buffer.append((state_encoding, action_probs))
            else:
                # Opponent's decision point - just propagate value forward
                cf_value = cf_values_after_state.get(i + 1, 0.0)
                cf_values_after_state[i] = cf_value
    
    # Log trajectory processing summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAJECTORY PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Total trajectories: {len(valid_trajectories)}")
    logger.info(f"  Total decision points processed: {total_decision_points}")
    logger.info(f"  Total regrets updated: {total_regrets_updated}")
    logger.info(f"  Total CF values computed: {total_cf_values_computed}")
    logger.info(f"  Value buffer size: {len(value_buffer)}")
    logger.info(f"  Policy buffer size: {len(policy_buffer)}")
    logger.info(f"  Unique info sets with regrets: {len(deep_cfr.regret_memory)}")
    logger.info(f"  Unique info sets with CF values: {len(deep_cfr.counterfactual_values)}")
    
    # --- REPLAY BUFFER LOGIC ---
    # Initialize and load replay buffers to prevent catastrophic forgetting
    import pickle
    
    value_replay = ReplayBuffer(maxlen=200000)
    policy_replay = ReplayBuffer(maxlen=200000)
    
    # Load existing buffers if available
    value_replay_path = "/checkpoints/value_replay.pkl"
    policy_replay_path = "/checkpoints/policy_replay.pkl"
    
    try:
        if os.path.exists(value_replay_path):
            value_replay.load(value_replay_path)
        if os.path.exists(policy_replay_path):
            policy_replay.load(policy_replay_path)
        logger.info(f"Loaded replay buffers: Value={len(value_replay)}, Policy={len(policy_replay)}")
    except Exception as e:
        logger.warning(f"Failed to load replay buffers: {e}")
    
    # Add new data to replay buffers
    value_replay.extend(value_buffer)
    policy_replay.extend(policy_buffer)
    logger.info(f"Added new samples: Value=+{len(value_buffer)}, Policy=+{len(policy_buffer)}")
    logger.info(f"Total replay buffer size: Value={len(value_replay)}, Policy={len(policy_replay)}")
    
    # Save replay buffers
    try:
        value_replay.save(value_replay_path)
        policy_replay.save(policy_replay_path)
    except Exception as e:
        logger.error(f"Failed to save replay buffers: {e}")

    # Train networks using Replay Buffer
    # We can now perform more updates because we have stable history
    num_updates = 1000 if len(value_replay) >= batch_size else 0
    value_losses = []
    policy_losses = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"NETWORK TRAINING (with Replay Buffer)")
    logger.info(f"{'='*60}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Updates: {num_updates}")
    
    if num_updates == 0:
        logger.warning(f"Insufficient data in replay buffer ({len(value_replay)}) for batch size {batch_size}")
        return {
            'iteration': iteration,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'checkpoint_path': '',
            'num_updates': 0
        }
    
    # Gradient clipping threshold
    max_grad_norm = 1.0
    
    for update_step in range(num_updates):
        # Log progress every 100 updates
        if update_step % 100 == 0 and update_step > 0:
            avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0.0
            avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
            logger.info(f"  Update {update_step}/{num_updates}: Value loss={avg_value_loss:.4f}, Policy loss={avg_policy_loss:.4f}")
        
        # Update value network from Replay Buffer
        if len(value_replay) >= batch_size:
            # Sample from replay buffer
            batch = value_replay.sample(batch_size)
            # Unpack batch: list of (state_encoding, value)
            batch_states_np = np.array([x[0] for x in batch])
            batch_values_np = np.array([x[1] for x in batch])
            
            batch_states = torch.tensor(batch_states_np, dtype=torch.float32).to(device)
            batch_values = torch.tensor(batch_values_np, dtype=torch.float32).to(device).unsqueeze(1)
            
            deep_cfr.value_optimizer.zero_grad()
            
            # Use mixed precision for faster training
            if use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    predicted_values, _ = value_net(batch_states, clip_value=True)
                    # Clip targets in scaled space
                    batch_values_clipped = torch.clamp(batch_values, -10000.0 / CFV_SCALE, 10000.0 / CFV_SCALE)
                    value_loss = torch.nn.MSELoss()(predicted_values, batch_values_clipped)
                
                # Check for NaN/Inf
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    logger.warning(f"Skipping value network update due to NaN/Inf loss at step {update_step}")
                    continue
                
                scaler.scale(value_loss).backward()
                scaler.unscale_(deep_cfr.value_optimizer)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                scaler.step(deep_cfr.value_optimizer)
                scaler.update()
            else:
                predicted_values, _ = value_net(batch_states, clip_value=True)
                # Clip targets in scaled space
                batch_values_clipped = torch.clamp(batch_values, -10000.0 / CFV_SCALE, 10000.0 / CFV_SCALE)
                value_loss = torch.nn.MSELoss()(predicted_values, batch_values_clipped)
                
                # Check for NaN/Inf
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    logger.warning(f"Skipping value network update due to NaN/Inf loss at step {update_step}")
                    continue
                
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                deep_cfr.value_optimizer.step()
            
            value_losses.append(value_loss.item())
        
        # Update policy network from Replay Buffer
        if len(policy_replay) >= batch_size:
            # Sample from replay buffer
            batch = policy_replay.sample(batch_size)
            # Unpack batch: list of (state_encoding, probs)
            batch_states_np = np.array([x[0] for x in batch])
            batch_probs_np = np.array([x[1] for x in batch])
            
            batch_states = torch.tensor(batch_states_np, dtype=torch.float32).to(device)
            batch_probs = torch.tensor(batch_probs_np, dtype=torch.float32).to(device)
            
            deep_cfr.policy_optimizer.zero_grad()
            
            # Use mixed precision for faster training
            if use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    _, policy_logits = policy_net(batch_states)
                    policy_probs = torch.softmax(policy_logits, dim=1)
                    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(
                        torch.log(policy_probs + 1e-8), batch_probs
                    )
                
                # Check for NaN/Inf
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    logger.warning(f"Skipping policy network update due to NaN/Inf loss at step {update_step}")
                    continue
                
                scaler.scale(kl_loss).backward()
                scaler.unscale_(deep_cfr.policy_optimizer)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                scaler.step(deep_cfr.policy_optimizer)
                scaler.update()
            else:
                _, policy_logits = policy_net(batch_states)
                policy_probs = torch.softmax(policy_logits, dim=1)
                kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(
                    torch.log(policy_probs + 1e-8), batch_probs
                )
                
                # Check for NaN/Inf
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    logger.warning(f"Skipping policy network update due to NaN/Inf loss at step {update_step}")
                    continue
                
                kl_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
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
        'strategy_memory': dict(deep_cfr.strategy_memory),
    }
    
    # Save checkpoint atomically (write to temp file, then rename)
    checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    temp_checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt.tmp"
    
    try:
        # Write to temp file first
        torch.save(checkpoint, temp_checkpoint_file)
        # Atomic rename
        os.rename(temp_checkpoint_file, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_checkpoint_file):
            try:
                os.remove(temp_checkpoint_file)
            except:
                pass
        raise
    
    # Commit volume with retry logic
    max_commit_retries = 3
    for retry in range(max_commit_retries):
        try:
            checkpoint_volume.commit()
            logger.info("Checkpoint volume committed successfully")
            break
        except Exception as e:
            if retry == max_commit_retries - 1:
                logger.error(f"Failed to commit checkpoint volume after {max_commit_retries} retries: {e}")
                raise
            logger.warning(f"Volume commit failed (retry {retry + 1}/{max_commit_retries}): {e}")
            import time
            time.sleep(1)  # Wait before retry
    
    avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0.0
    avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
    
    return {
        'iteration': iteration,
        'value_loss': avg_value_loss,
        'policy_loss': avg_policy_loss,
        'checkpoint_path': checkpoint_file,
        'num_updates': num_updates,
        'value_buffer_size': len(value_buffer),
        'policy_buffer_size': len(policy_buffer)
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
            checkpoint = torch.load(f"/checkpoints/{path}", map_location='cpu', weights_only=False)
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


def small_evaluation(iteration: int, num_games: int = 200) -> Dict[str, Any]:
    """
    Run a small evaluation of the current checkpoint against simple baselines.
    This is used inside the training loop for cheap sanity checks.
    """
    import os
    import random
    import numpy as np
    import torch
    from poker_game.game import PokerGame, Action
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet

    checkpoint_path = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    if not os.path.exists(checkpoint_path):
        return {}

    # Initialize game and state encoder
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    encoder = StateEncoder()
    input_dim = encoder.feature_dim

    # Load current policy network
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "policy_net_state" not in checkpoint:
        return {}
    current_net = ValuePolicyNet(input_dim=input_dim)
    current_net.load_state_dict(checkpoint["policy_net_state"])
    current_net.eval()

    # Baseline agents
    class RandomAgent:
        def get_action(self, state, legal_actions):
            if not legal_actions:
                return (Action.FOLD, 0)
            return random.choice(legal_actions)

    class AlwaysCallAgent:
        def get_action(self, state, legal_actions):
            to_call = state.current_bets[1 - state.current_player] - state.current_bets[state.current_player]
            if to_call == 0:
                for a, amt in legal_actions:
                    if a == Action.CHECK:
                        return (a, amt)
            else:
                for a, amt in legal_actions:
                    if a == Action.CALL:
                        return (a, amt)
            return (Action.FOLD, 0)

    def play_match(opponent, num_games: int) -> Tuple[float, float]:
        wins = 0
        total_payoff = 0.0
        for _ in range(num_games):
            state = game.reset()
            # Randomly assign positions
            current_is_player0 = random.random() < 0.5

            while not state.is_terminal:
                player = state.current_player
                legal_actions = game.get_legal_actions(state)
                if not legal_actions:
                    break

                if (player == 0 and current_is_player0) or (player == 1 and not current_is_player0):
                    # Current bot's turn
                    enc = encoder.encode(state, player)
                    t = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        _, logits = current_net(t)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    num_legal = len(legal_actions)
                    legal_probs = probs[:num_legal]
                    if legal_probs.sum() > 0:
                        legal_probs /= legal_probs.sum()
                    else:
                        legal_probs = np.ones(num_legal) / num_legal
                    idx = np.random.choice(num_legal, p=legal_probs)
                    action, amount = legal_actions[idx]
                else:
                    # Opponent's turn
                    action, amount = opponent.get_action(state, legal_actions)

                state = game.apply_action(state, action, amount)

            payoffs = game.get_payoff(state)
            if current_is_player0:
                total_payoff += payoffs[0]
                if payoffs[0] > payoffs[1]:
                    wins += 1
            else:
                total_payoff += payoffs[1]
                if payoffs[1] > payoffs[0]:
                    wins += 1

        return wins / num_games, total_payoff / num_games

    random_agent = RandomAgent()
    always_call_agent = AlwaysCallAgent()

    rand_wr, rand_ev = play_match(random_agent, num_games)
    call_wr, call_ev = play_match(always_call_agent, num_games)

    return {
        "eval_random_win_rate": rand_wr,
        "eval_random_ev": rand_ev,
        "eval_always_call_win_rate": call_wr,
        "eval_always_call_ev": call_ev,
    }


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    cpu=2,
    memory=4096,
    timeout=86400,  # 24 hours
)
def training_worker(
    num_iterations: int,
    trajectories_per_iteration: int,
    num_workers: int,
    start_iteration: int = 0,
    batch_size: int = 32
):
    """Main training worker that runs asynchronously."""
    import logging
    import sys
    from modal_deploy.metrics import MetricsLogger
    
    # Set up logging to both file and stdout
    log_file = '/checkpoints/training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    # Initialize metrics directory
    import os
    os.makedirs("/checkpoints/metrics", exist_ok=True)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir="/checkpoints/metrics")
    
    logger.info(f"Starting training: {num_iterations} iterations, {trajectories_per_iteration} trajectories/iter")
    logger.info(f"Starting from iteration {start_iteration}")
    
    # Initialize iteration variable before try block for exception handling
    iteration = start_iteration
    
    # Validate worker configuration
    if num_workers > trajectories_per_iteration:
        logger.error(f"num_workers ({num_workers}) > trajectories_per_iteration ({trajectories_per_iteration}). This will result in 0 trajectories per worker!")
        raise ValueError(f"num_workers must be <= trajectories_per_iteration")
    
    if num_workers <= 0:
        logger.error(f"num_workers must be > 0, got {num_workers}")
        raise ValueError(f"num_workers must be > 0")
    
    try:
        for iteration in range(start_iteration, num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")
            
            # Determine checkpoint path
            checkpoint_path = None
            if iteration > 0:
                checkpoint_path = f"checkpoint_iter_{iteration - 1}.pt"
            
            # Generate trajectories in parallel
            trajectories_per_worker = trajectories_per_iteration // num_workers
            remainder = trajectories_per_iteration % num_workers
            
            logger.info(f"Generating {trajectories_per_iteration} trajectories using {num_workers} workers...")
            logger.info(f"Trajectories per worker: {trajectories_per_worker} (remainder: {remainder})")
            
            trajectory_futures = []
            for worker_id in range(num_workers):
                # Distribute remainder to first few workers
                worker_trajectories = trajectories_per_worker + (1 if worker_id < remainder else 0)
                future = generate_trajectories.spawn(
                    num_trajectories=worker_trajectories,
                    checkpoint_path=checkpoint_path,
                    iteration=iteration
                )
                trajectory_futures.append(future)
            
            # Collect trajectories with retry logic
            all_trajectories = []
            max_worker_retries = 2
            for i in range(num_workers):
                retry_count = 0
                worker_success = False
                worker_trajectories = trajectories_per_worker + (1 if i < remainder else 0)
                
                while retry_count < max_worker_retries and not worker_success:
                    try:
                        # Get or create future for this worker
                        if retry_count == 0:
                            future = trajectory_futures[i]
                        else:
                            # Respawn worker on retry
                            logger.warning(f"Worker {i+1} failed (attempt {retry_count}/{max_worker_retries}). Retrying...")
                            future = generate_trajectories.spawn(
                                num_trajectories=worker_trajectories,
                                checkpoint_path=checkpoint_path,
                                iteration=iteration
                            )
                        
                        trajectories = future.get(timeout=3600)  # 1 hour timeout per worker
                        all_trajectories.extend(trajectories)
                        logger.info(f"Worker {i+1}/{num_workers} completed: {len(trajectories)} trajectories")
                        worker_success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_worker_retries:
                            logger.error(f"Worker {i+1} failed after {max_worker_retries} attempts: {e}")
                            # Continue with partial results rather than failing entire iteration
                            logger.warning(f"Continuing with {len(all_trajectories)} trajectories from {i} workers")
                            break
            
            logger.info(f"Total trajectories generated: {len(all_trajectories)}")
            
            if len(all_trajectories) == 0:
                logger.error("No trajectories generated! Skipping this iteration.")
                # Log metrics for failed iteration
                metrics = {
                    "iteration": iteration + 1,
                    "trajectories_generated": 0,
                    "value_loss": 0.0,
                    "policy_loss": 0.0,
                    "checkpoint_path": '',
                    "num_updates": 0,
                    "error": "no_trajectories_generated"
                }
                try:
                    metrics_logger.log_iteration(iteration + 1, metrics)
                except Exception as e:
                    logger.error(f"Failed to log metrics: {e}")
                continue
            
            # Train networks
            logger.info("Training networks on GPU...")
            try:
                train_result = train_networks.spawn(
                    trajectories=all_trajectories,
                    checkpoint_path=checkpoint_path,
                    iteration=iteration,
                    batch_size=batch_size
                ).get(timeout=7200)  # 2 hour timeout
            except Exception as e:
                logger.error(f"Training networks failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Log error metrics
                metrics = {
                    "iteration": iteration + 1,
                    "trajectories_generated": len(all_trajectories),
                    "value_loss": 0.0,
                    "policy_loss": 0.0,
                    "checkpoint_path": '',
                    "num_updates": 0,
                    "error": str(e)
                }
                try:
                    metrics_logger.log_iteration(iteration + 1, metrics)
                except Exception as log_err:
                    logger.error(f"Failed to log error metrics: {log_err}")
                raise
            
            # Log metrics with error handling
            metrics = {
                "iteration": iteration + 1,
                "trajectories_generated": len(all_trajectories),
                "value_loss": train_result.get('value_loss', 0.0),
                "policy_loss": train_result.get('policy_loss', 0.0),
                "checkpoint_path": train_result.get('checkpoint_path', ''),
                "num_updates": train_result.get('num_updates', 0)
            }

            # Occasionally run a small evaluation against simple baselines
            if (iteration + 1) % 5 == 0 and train_result.get('checkpoint_path'):
                try:
                    eval_metrics = small_evaluation(iteration)
                    if eval_metrics:
                        metrics.update(eval_metrics)
                except Exception as e:
                    logger.warning(f"Small evaluation failed at iteration {iteration + 1}: {e}")
            
            # Validate metrics for NaN/Inf
            def is_valid_number(v):
                """Check if value is a valid number (not NaN or Inf)."""
                if not isinstance(v, (int, float)):
                    return False
                return v == v and v != float('inf') and v != float('-inf')
            
            if not is_valid_number(metrics['value_loss']) or not is_valid_number(metrics['policy_loss']):
                logger.warning(f"NaN/Inf detected in metrics: {metrics}")
                metrics['value_loss'] = 0.0 if not is_valid_number(metrics['value_loss']) else metrics['value_loss']
                metrics['policy_loss'] = 0.0 if not is_valid_number(metrics['policy_loss']) else metrics['policy_loss']
            
            try:
                metrics_logger.log_iteration(iteration + 1, metrics)
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Don't fail iteration due to metrics logging failure
            
            logger.info(f"Training complete:")
            logger.info(f"  Value loss: {metrics['value_loss']:.6f}")
            logger.info(f"  Policy loss: {metrics['policy_loss']:.6f}")
            logger.info(f"  Checkpoint: {metrics['checkpoint_path']}")
            
            # Commit volume after every iteration for crash safety (with retry)
            max_commit_retries = 3
            commit_success = False
            for retry in range(max_commit_retries):
                try:
                    checkpoint_volume.commit()
                    logger.info(f"✓ Checkpoint committed at iteration {iteration + 1}")
                    commit_success = True
                    break
                except Exception as e:
                    if retry == max_commit_retries - 1:
                        logger.error(f"Failed to commit checkpoint volume after {max_commit_retries} retries: {e}")
                        # Don't fail iteration, but log error
                    else:
                        logger.warning(f"Volume commit failed (retry {retry + 1}/{max_commit_retries}): {e}")
                        import time
                        time.sleep(1)
            
            # Checkpoint periodically (informational)
            if (iteration + 1) % TRAINING_CONFIG['checkpoint_frequency'] == 0:
                logger.info(f"✓ Major checkpoint milestone at iteration {iteration + 1}")
    
    except Exception as e:
        logger.error(f"Training interrupted at iteration {iteration}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("Attempting final checkpoint commit...")
        try:
            checkpoint_volume.commit()
            logger.info("✓ Final checkpoint committed despite error")
        except Exception as commit_err:
            logger.error(f"Failed to commit checkpoint volume: {commit_err}")
        raise
    finally:
        # Always commit at end
        logger.info("="*60)
        logger.info("Training complete!")
        logger.info("="*60)
        try:
            checkpoint_volume.commit()
            logger.info("✓ Final checkpoint committed")
        except Exception as e:
            logger.error(f"Failed to commit final checkpoint: {e}")
    
    return {
        "status": "completed",
        "total_iterations": num_iterations,
        "final_metrics": metrics_logger.get_summary()
    }


@app.local_entrypoint()
def main(
    num_iterations: int = 1000,
    trajectories_per_iteration: int = 10000,
    num_workers: int = 4,
    resume_from: int = None,
    batch_size: int = 32,
    deploy: bool = False
):
    """Main entrypoint for training.
    
    Args:
        num_iterations: Number of training iterations
        trajectories_per_iteration: Trajectories per iteration
        num_workers: Number of parallel workers for trajectory generation
        resume_from: Iteration to resume from (None = start from beginning)
        batch_size: Batch size for network training
        deploy: If True, deploy as async job. If False, run synchronously.
    """
    start_iteration = resume_from if resume_from else 0
    
    if deploy:
        # Initialize metrics directory
        print("Setting up metrics directory...")
        try:
            setup_metrics_dir.remote()
        except:
            pass  # May already exist
        
        # Deploy app for async execution
        print("\n" + "="*60)
        print("Deploying app to Modal...")
        print("="*60)
        print(f"Configuration:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Trajectories per iteration: {trajectories_per_iteration}")
        print(f"  Workers: {num_workers}")
        print(f"  Batch size: {batch_size}")
        if start_iteration > 0:
            print(f"  Resuming from iteration: {start_iteration}")
        print()
        
        # Deploy the app first to make functions available
        print("Deploying app to Modal...")
        with modal.enable_output():
            app.deploy()
        
        print("="*60)
        print("✓ App deployed successfully!")
        print("="*60)
        print()
        print("Triggering training job...")
        
        # Now spawn the training worker - app is deployed so it will run independently
        future = training_worker.spawn(
            num_iterations=num_iterations,
            trajectories_per_iteration=trajectories_per_iteration,
            num_workers=num_workers,
            start_iteration=start_iteration,
            batch_size=batch_size
        )
        
        # Give it a moment to start before exiting
        import time
        print("Waiting for function to start...")
        time.sleep(3)  # Give Modal time to allocate container
        
        print(f"✓ Training job triggered! Function Call ID: {future.object_id}")
        print()
        print("The training worker is now running independently on Modal.")
        print("You can close this terminal - the job will continue running.")
        print()
        print("Monitor progress:")
        print(f"  - Modal dashboard: https://modal.com/apps")
        print(f"  - Check status: ./scripts/check_status.sh")
        print(f"  - View metrics: python scripts/view_metrics.py")
        print()
        print("Note: Metrics and logs will appear in Modal volume after first iteration.")
        print()
        
    else:
        # Run synchronously
        print(f"Starting training: {num_iterations} iterations, {trajectories_per_iteration} trajectories/iter")
        result = training_worker.remote(
            num_iterations=num_iterations,
            trajectories_per_iteration=trajectories_per_iteration,
            num_workers=num_workers,
            start_iteration=start_iteration,
            batch_size=batch_size
        )
        print("\nTraining complete!")
        print(f"Final summary: {result.get('final_metrics', {})}")


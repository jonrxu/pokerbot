"""Modal functions for distributed training."""

import modal
import torch
import os
from typing import List, Dict, Any
import sys

# Add parent directory to path for local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modal_deploy.config import image, checkpoint_volume, GPU_CONFIG, CPU_CONFIG, TRAINING_CONFIG

app = modal.App("poker-bot-training")


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
    """Generate self-play trajectories (CPU workers)."""
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
    from models.value_policy_net import ValuePolicyNet
    from training.deep_cfr import DeepCFR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Log GPU availability
    import logging
    if device == 'cpu' and torch.cuda.is_available() == False:
        logging.warning("GPU not available, falling back to CPU. Training will be slower.")
    elif device == 'cuda':
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
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
        logging.warning(f"Skipped {invalid_count} invalid trajectories out of {len(trajectories)}")
    
    if len(valid_trajectories) == 0:
        logging.error("No valid trajectories to process!")
        return {
            'iteration': iteration,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'checkpoint_path': '',
            'num_updates': 0,
            'error': 'no_valid_trajectories'
        }
    
    for trajectory in valid_trajectories:
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
    
    if num_updates == 0:
        logging.warning(f"No updates possible: value_buffer={len(value_buffer)}, policy_buffer={len(policy_buffer)}, batch_size={batch_size}")
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
        # Update value network
        if len(value_buffer) >= batch_size:
            indices = torch.randint(0, len(value_buffer), (batch_size,))
            batch_states = torch.tensor(np.array([value_buffer[i][0] for i in indices]), dtype=torch.float32).to(device)
            batch_values = torch.tensor([value_buffer[i][1] for i in indices], dtype=torch.float32).to(device).unsqueeze(1)
            
            deep_cfr.value_optimizer.zero_grad()
            predicted_values, _ = value_net(batch_states)
            value_loss = torch.nn.MSELoss()(predicted_values, batch_values)
            
            # Check for NaN/Inf
            if torch.isnan(value_loss) or torch.isinf(value_loss):
                logging.warning(f"Skipping value network update due to NaN/Inf loss at step {update_step}")
                continue
            
            value_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            deep_cfr.value_optimizer.step()
            value_losses.append(value_loss.item())
        
        # Update policy network
        if len(policy_buffer) >= batch_size:
            indices = torch.randint(0, len(policy_buffer), (batch_size,))
            batch_states = torch.tensor(np.array([policy_buffer[i][0] for i in indices]), dtype=torch.float32).to(device)
            batch_probs = torch.tensor([policy_buffer[i][1] for i in indices], dtype=torch.float32).to(device)
            
            deep_cfr.policy_optimizer.zero_grad()
            _, policy_logits = policy_net(batch_states)
            policy_probs = torch.softmax(policy_logits, dim=1)
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(
                torch.log(policy_probs + 1e-8), batch_probs
            )
            
            # Check for NaN/Inf
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                logging.warning(f"Skipping policy network update due to NaN/Inf loss at step {update_step}")
                continue
            
            kl_loss.backward()
            # Gradient clipping
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
    }
    
    # Save checkpoint atomically (write to temp file, then rename)
    checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    temp_checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt.tmp"
    
    try:
        # Write to temp file first
        torch.save(checkpoint, temp_checkpoint_file)
        # Atomic rename
        os.rename(temp_checkpoint_file, checkpoint_file)
        logging.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
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
            logging.info("Checkpoint volume committed successfully")
            break
        except Exception as e:
            if retry == max_commit_retries - 1:
                logging.error(f"Failed to commit checkpoint volume after {max_commit_retries} retries: {e}")
                raise
            logging.warning(f"Volume commit failed (retry {retry + 1}/{max_commit_retries}): {e}")
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


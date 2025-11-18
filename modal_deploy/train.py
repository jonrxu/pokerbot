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
) -> Dict[str, List]:
    """Generate self-play trajectories (CPU workers).
    
    Returns a dictionary containing training buffers:
    {
        'advantage': [(state, advantage_vector), ...],
        'policy': [(state, policy_vector), ...]
    }
    """
    from collections import defaultdict
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from models.advantage_net import AdvantageNet
    from models.policy_net import PolicyNet
    from training.deep_cfr import DeepCFR
    
    # Initialize game
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    
    # Load or create networks
    input_dim = state_encoder.feature_dim
    advantage_net = AdvantageNet(input_dim=input_dim)
    policy_net = PolicyNet(input_dim=input_dim)
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location='cpu', weights_only=False)
            # Validate checkpoint has required keys
            required_keys = ['advantage_net_state', 'policy_net_state']
            if all(key in checkpoint for key in required_keys):
                advantage_net.load_state_dict(checkpoint['advantage_net_state'])
                policy_net.load_state_dict(checkpoint['policy_net_state'])
            else:
                import logging
                logging.warning(f"Checkpoint {checkpoint_path} missing required keys. Using new networks.")
        except Exception as e:
            import logging
            logging.warning(f"Failed to load checkpoint {checkpoint_path}: {e}. Using new networks.")
    
    # Initialize Deep CFR
    deep_cfr = DeepCFR(
        advantage_net=advantage_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        learning_rate=5e-5,
        device='cpu'
    )
    
    # Training buffers for this batch of trajectories
    buffers = {
        'advantage': [],
        'policy': []
    }
    
    # Generate trajectories via external sampling traversal
    for _ in range(num_trajectories):
        state = game.reset()
        # Randomly choose a player to traverse (train)
        # The other player will be simulated using the PolicyNet (average strategy)
        traversal_player = random.randint(0, 1)
        
        deep_cfr.traverse_external_sampling(state, traversal_player, buffers)
        
    return buffers


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    **GPU_CONFIG,
    timeout=7200,
)
def train_networks(
    worker_buffers: List[Dict[str, List]],
    checkpoint_path: str = None,
    iteration: int = 0,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Train networks on aggregated buffers (GPU workers)."""
    import numpy as np
    from collections import defaultdict
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from models.advantage_net import AdvantageNet
    from models.policy_net import PolicyNet
    from training.deep_cfr import DeepCFR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Enable mixed precision training for 2x GPU speedup
    use_amp = device == 'cuda'
    scaler = None
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
    
    # Log GPU availability
    import logging
    import sys
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
    advantage_net = AdvantageNet(input_dim=input_dim).to(device)
    policy_net = PolicyNet(input_dim=input_dim).to(device)
    
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location=device, weights_only=False)
            # Validate checkpoint has required keys
            required_keys = ['advantage_net_state', 'policy_net_state']
            if all(key in checkpoint for key in required_keys):
                advantage_net.load_state_dict(checkpoint['advantage_net_state'])
                policy_net.load_state_dict(checkpoint['policy_net_state'])
                logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            else:
                logging.warning(f"Checkpoint {checkpoint_path} missing required keys. Using new networks.")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint {checkpoint_path}: {e}. Using new networks.")
    
    # Initialize Deep CFR
    deep_cfr = DeepCFR(
        advantage_net=advantage_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        learning_rate=5e-5,
        device=device
    )
    
    # Load optimizer states if available
    if checkpoint_path and os.path.exists(f"/checkpoints/{checkpoint_path}"):
        try:
            checkpoint = torch.load(f"/checkpoints/{checkpoint_path}", map_location=device, weights_only=False)
            if 'advantage_optimizer_state' in checkpoint:
                deep_cfr.advantage_optimizer.load_state_dict(checkpoint['advantage_optimizer_state'])
            if 'policy_optimizer_state' in checkpoint:
                deep_cfr.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        except Exception as e:
            logging.warning(f"Failed to load optimizer states: {e}")

    # --- REPLAY BUFFER LOGIC ---
    advantage_replay = ReplayBuffer(maxlen=200000)
    policy_replay = ReplayBuffer(maxlen=200000)
    
    # Load existing buffers
    try:
        if os.path.exists("/checkpoints/advantage_replay.pkl"): advantage_replay.load("/checkpoints/advantage_replay.pkl")
        if os.path.exists("/checkpoints/policy_replay.pkl"): policy_replay.load("/checkpoints/policy_replay.pkl")
        logger.info(f"Loaded replay buffers: Adv={len(advantage_replay)}, Pol={len(policy_replay)}")
    except Exception as e:
        logger.warning(f"Failed to load replay buffers: {e}")

    # Aggregated new data from workers
    new_advantage_count = 0
    new_policy_count = 0
    
    for buff in worker_buffers:
        advantage_replay.extend(buff['advantage'])
        policy_replay.extend(buff['policy'])
        new_advantage_count += len(buff['advantage'])
        new_policy_count += len(buff['policy'])
        
    logger.info(f"Added new samples: Adv=+{new_advantage_count}, Pol=+{new_policy_count}")
    
    # Save updated buffers
    try:
        advantage_replay.save("/checkpoints/advantage_replay.pkl")
        policy_replay.save("/checkpoints/policy_replay.pkl")
    except Exception as e:
        logger.error(f"Failed to save replay buffers: {e}")

    # --- TRAINING LOOP ---
    # Updates proportional to buffer size
    num_updates = 1000 if len(advantage_replay) >= batch_size else 0
    
    advantage_losses = []
    policy_losses = []
    
    logger.info(f"\nTRAINING NETWORKS (Updates: {num_updates})")
    
    if num_updates == 0:
        logger.warning(f"Insufficient data in replay buffer ({len(advantage_replay)}) for batch size {batch_size}")
        return {
            'iteration': iteration,
            'advantage_loss': 0.0,
            'policy_loss': 0.0,
            'checkpoint_path': '',
            'num_updates': 0
        }
        
    max_grad_norm = 1.0
    
    for update_step in range(num_updates):
        # 1. Train Advantage Net (MSE on Advantage Vector)
        if len(advantage_replay) >= batch_size:
            batch = advantage_replay.sample(batch_size)
            states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
            targets = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device) # Vector
            
            deep_cfr.advantage_optimizer.zero_grad()
            if use_amp:
                with autocast('cuda'):
                    preds = advantage_net(states)
                    loss = torch.nn.MSELoss()(preds, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(deep_cfr.advantage_optimizer)
                torch.nn.utils.clip_grad_norm_(advantage_net.parameters(), max_grad_norm)
                scaler.step(deep_cfr.advantage_optimizer)
            else:
                preds = advantage_net(states)
                loss = torch.nn.MSELoss()(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(advantage_net.parameters(), max_grad_norm)
                deep_cfr.advantage_optimizer.step()
            advantage_losses.append(loss.item())

        # 2. Train Policy Net (Cross Entropy / KL)
        if len(policy_replay) >= batch_size:
            batch = policy_replay.sample(batch_size)
            states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
            targets = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device)
            
            deep_cfr.policy_optimizer.zero_grad()
            if use_amp:
                with autocast('cuda'):
                    logits = policy_net(states)
                    # KL Div Loss
                    probs = torch.softmax(logits, dim=1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(probs + 1e-8), targets)
                scaler.scale(loss).backward()
                scaler.unscale_(deep_cfr.policy_optimizer)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                scaler.step(deep_cfr.policy_optimizer)
            else:
                logits = policy_net(states)
                probs = torch.softmax(logits, dim=1)
                loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(probs + 1e-8), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                deep_cfr.policy_optimizer.step()
            policy_losses.append(loss.item())
            
        if use_amp:
            scaler.update()

    # Prepare updated checkpoint
    checkpoint = {
        'iteration': iteration,
        'advantage_net_state': advantage_net.state_dict(),
        'policy_net_state': policy_net.state_dict(),
        'advantage_optimizer_state': deep_cfr.advantage_optimizer.state_dict(),
        'policy_optimizer_state': deep_cfr.policy_optimizer.state_dict(),
    }
    
    # Save checkpoint atomically
    checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    temp_checkpoint_file = f"/checkpoints/checkpoint_iter_{iteration}.pt.tmp"
    
    try:
        torch.save(checkpoint, temp_checkpoint_file)
        os.rename(temp_checkpoint_file, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if os.path.exists(temp_checkpoint_file):
            os.remove(temp_checkpoint_file)
        raise
    
    # Commit volume
    try:
        checkpoint_volume.commit()
    except Exception as e:
        logger.warning(f"Volume commit failed: {e}")
        
    return {
        'iteration': iteration,
        'advantage_loss': np.mean(advantage_losses) if advantage_losses else 0.0,
        'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
        'checkpoint_path': checkpoint_file,
        'num_updates': num_updates
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
    from models.policy_net import PolicyNet

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
    current_net = PolicyNet(input_dim=input_dim)
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
                        logits = current_net(t)
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
            
            # Collect buffers with retry logic
            all_worker_buffers = []
            max_worker_retries = 2
            total_trajectories = 0
            
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
                        
                        buffers = future.get(timeout=3600)  # 1 hour timeout per worker
                        all_worker_buffers.append(buffers)
                        
                        # Count total items
                        item_count = len(buffers['advantage'])
                        total_trajectories += worker_trajectories # Approximate
                        logger.info(f"Worker {i+1}/{num_workers} completed: {item_count} samples")
                        worker_success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_worker_retries:
                            logger.error(f"Worker {i+1} failed after {max_worker_retries} attempts: {e}")
                            break
            
            logger.info(f"Total worker buffers collected: {len(all_worker_buffers)}")
            
            if len(all_worker_buffers) == 0:
                logger.error("No data generated! Skipping this iteration.")
                continue
            
            # Train networks
            logger.info("Training networks on GPU...")
            try:
                train_result = train_networks.spawn(
                    worker_buffers=all_worker_buffers,
                    checkpoint_path=checkpoint_path,
                    iteration=iteration,
                    batch_size=batch_size
                ).get(timeout=7200)  # 2 hour timeout
            except Exception as e:
                logger.error(f"Training networks failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
            # Log metrics
            metrics = {
                "iteration": iteration + 1,
                "trajectories_generated": total_trajectories,
                "advantage_loss": train_result.get('advantage_loss', 0.0),
                "policy_loss": train_result.get('policy_loss', 0.0),
                "checkpoint_path": train_result.get('checkpoint_path', ''),
                "num_updates": train_result.get('num_updates', 0)
            }

            # Occasionally run a small evaluation
            if (iteration + 1) % 5 == 0 and train_result.get('checkpoint_path'):
                try:
                    eval_metrics = small_evaluation(iteration)
                    if eval_metrics:
                        metrics.update(eval_metrics)
                except Exception as e:
                    logger.warning(f"Small evaluation failed at iteration {iteration + 1}: {e}")
            
            try:
                metrics_logger.log_iteration(iteration + 1, metrics)
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
            
            logger.info(f"Training complete:")
            logger.info(f"  Advantage loss: {metrics['advantage_loss']:.6f}")
            logger.info(f"  Policy loss: {metrics['policy_loss']:.6f}")
            logger.info(f"  Checkpoint: {metrics['checkpoint_path']}")
            
            # Commit volume
            try:
                checkpoint_volume.commit()
            except Exception as e:
                logger.warning(f"Volume commit failed: {e}")
            
            # Checkpoint frequency
            if (iteration + 1) % TRAINING_CONFIG['checkpoint_frequency'] == 0:
                logger.info(f"✓ Major checkpoint milestone at iteration {iteration + 1}")
    
    except Exception as e:
        logger.error(f"Training interrupted at iteration {iteration}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Training worker finished.")

@app.local_entrypoint()
def main(
    num_iterations: int = 1000,
    trajectories_per_iteration: int = 10000,
    num_workers: int = 4,
    resume_from: int = None,
    batch_size: int = 32,
    deploy: bool = False
):
    """Main entrypoint for training."""
    start_iteration = resume_from if resume_from else 0
    
    if deploy:
        print("Deploying app to Modal...")
        with modal.enable_output():
            app.deploy()
        
        print("Triggering training job...")
        future = training_worker.spawn(
            num_iterations=num_iterations,
            trajectories_per_iteration=trajectories_per_iteration,
            num_workers=num_workers,
            start_iteration=start_iteration,
            batch_size=batch_size
        )
        print(f"Job triggered: {future.object_id}")
        
    else:
        print(f"Starting local training...")
        training_worker.remote(
            num_iterations=num_iterations,
            trajectories_per_iteration=trajectories_per_iteration,
            num_workers=num_workers,
            start_iteration=start_iteration,
            batch_size=batch_size
        )

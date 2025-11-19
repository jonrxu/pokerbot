"""Modal functions for distributed training."""

import modal
import torch
import os
from typing import List, Dict, Any, Tuple
import sys
import collections
import random
import pickle
from datetime import datetime

# Add parent directory to path for local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modal_deploy.config import image, checkpoint_volume, GPU_CONFIG, CPU_CONFIG, TRAINING_CONFIG

app = modal.App("poker-bot-training")


class ReplayBuffer:
    """Experience replay buffer for stable training (Sliding Window)."""
    def __init__(self, maxlen=50000): 
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


class PolicyReservoirBuffer:
    """Reservoir Sampling buffer for Policy Network (Long-term Average).
    
    Approximates uniform sampling over ALL past history without infinite memory.
    Used to learn the Nash Equilibrium average strategy.
    """
    def __init__(self, capacity=2000000):
        self.capacity = capacity
        self.buffer = []
        self.total_seen = 0
        
    def add(self, item):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Reservoir logic: replace random existing item with probability k/n
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = item
                
    def extend(self, items):
        for item in items:
            self.add(item)
            
    def sample(self, batch_size):
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
        
    def __len__(self):
        return len(self.buffer)
        
    def save(self, path):
        state = {
            'buffer': self.buffer,
            'total_seen': self.total_seen,
            'capacity': self.capacity
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    state = pickle.load(f)
                    if isinstance(state, dict):
                        self.buffer = state.get('buffer', [])
                        self.total_seen = state.get('total_seen', 0)
                        self.capacity = state.get('capacity', self.capacity)
                    else:
                        # Handle legacy or format mismatch
                        pass 
                except Exception:
                    pass


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
    """Generate self-play trajectories (CPU workers)."""
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
    full_path = None
    if checkpoint_path:
        if checkpoint_path.startswith("/"):
            full_path = checkpoint_path
        else:
            full_path = f"/checkpoints/{checkpoint_path}"

    if full_path and os.path.exists(full_path):
        try:
            checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
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
    
    # Generate trajectories via Outcome Sampling traversal
    max_samples_per_trajectory = 200
    
    for traj_idx in range(num_trajectories):
        state = game.reset()
        traversal_player = random.randint(0, 1)
        
        # Linear CFR weighting: later iterations get higher weight
        iteration_weight = float(iteration + 1)
        
        # Use Outcome Sampling
        deep_cfr.traverse_outcome_sampling(
            state, 
            traversal_player, 
            buffers, 
            iteration_weight=iteration_weight,
            max_depth=40,
            sample_reach=1.0
        )
        
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
    batch_size: int = 32,
    run_id: str = "default"
) -> Dict[str, Any]:
    """Train networks on aggregated buffers (GPU workers)."""
    import numpy as np
    from collections import defaultdict
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from models.advantage_net import AdvantageNet
    from models.policy_net import PolicyNet
    from training.deep_cfr import DeepCFR
    import torch.optim as optim
    
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
    
    run_dir = f"/checkpoints/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

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
    
    # Load checkpoint if available
    full_path = None
    if checkpoint_path:
        if checkpoint_path.startswith("/"):
            full_path = checkpoint_path
        else:
            full_path = f"/checkpoints/{checkpoint_path}"

    if full_path and os.path.exists(full_path):
        try:
            checkpoint = torch.load(full_path, map_location=device, weights_only=False)
            required_keys = ['advantage_net_state', 'policy_net_state']
            if all(key in checkpoint for key in required_keys):
                advantage_net.load_state_dict(checkpoint['advantage_net_state'])
                policy_net.load_state_dict(checkpoint['policy_net_state'])
                logging.info(f"Successfully loaded checkpoint from {full_path}")
            else:
                logging.warning(f"Checkpoint {full_path} missing required keys. Using new networks.")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint {full_path}: {e}. Using new networks.")
    
    # Initialize Deep CFR
    deep_cfr = DeepCFR(
        advantage_net=advantage_net,
        policy_net=policy_net,
        state_encoder=state_encoder,
        game=game,
        learning_rate=5e-5,
        device=device
    )
    
    # Load optimizer states
    if full_path and os.path.exists(full_path):
        try:
            checkpoint = torch.load(full_path, map_location=device, weights_only=False)
            if 'advantage_optimizer_state' in checkpoint:
                deep_cfr.advantage_optimizer.load_state_dict(checkpoint['advantage_optimizer_state'])
            if 'policy_optimizer_state' in checkpoint:
                deep_cfr.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        except Exception as e:
            logging.warning(f"Failed to load optimizer states: {e}")

    # --- LR SCHEDULER ---
    # Decay LR by 0.98 every iteration
    adv_scheduler = optim.lr_scheduler.ExponentialLR(deep_cfr.advantage_optimizer, gamma=0.98)
    pol_scheduler = optim.lr_scheduler.ExponentialLR(deep_cfr.policy_optimizer, gamma=0.98)

    # --- REPLAY BUFFER LOGIC (PHASE 4) ---
    # Advantage: Short-term history (Sliding Window 200k ~ 3-4 iterations)
    advantage_replay = ReplayBuffer(maxlen=200000)
    
    # Policy: Long-term history (Reservoir Sampling 2M ~ 30+ iterations)
    # This allows converging to Nash Equilibrium (Average Strategy)
    policy_replay = PolicyReservoirBuffer(capacity=2000000)
    
    # Load existing buffers
    # Note: For Phase 4, we might want to RESET the policy buffer if starting a fresh run
    # but keep the advantage buffer if resuming?
    # Actually, if we start a new run_id, we usually start fresh buffers UNLESS resuming.
    # But user wants to "kickstart" from Iter 14.
    # If we load old buffers, we might load the small deque into the Reservoir?
    # Let's handle that.
    
    try:
        if os.path.exists("/checkpoints/advantage_replay.pkl"): 
            advantage_replay.load("/checkpoints/advantage_replay.pkl")
            # Resize if needed
            if len(advantage_replay) > advantage_replay.buffer.maxlen:
                 import collections
                 new_q = collections.deque(list(advantage_replay.buffer)[-advantage_replay.buffer.maxlen:], maxlen=advantage_replay.buffer.maxlen)
                 advantage_replay.buffer = new_q
            
        if os.path.exists("/checkpoints/policy_replay.pkl"): 
            # Attempt to load. If it's a deque (old), we should probably convert or discard.
            # Given we want to start Phase 4 properly, DISCARDING old policy data is safer
            # to avoid polluting the reservoir with non-reservoir data.
            # However, losing 14 iterations of data is sad.
            # Let's try to load and populate the reservoir.
            with open("/checkpoints/policy_replay.pkl", 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, collections.deque):
                    logger.info(f"Converting legacy Policy deque (len={len(data)}) to Reservoir...")
                    policy_replay.extend(list(data))
                elif isinstance(data, dict) and 'buffer' in data:
                    # It's already a reservoir state
                    policy_replay.buffer = data['buffer']
                    policy_replay.total_seen = data['total_seen']
                    policy_replay.capacity = data['capacity']
                 
        logger.info(f"Loaded buffers: Adv={len(advantage_replay)}, Pol={len(policy_replay)} (Total Seen: {policy_replay.total_seen})")
    except Exception as e:
        logger.warning(f"Failed to load replay buffers (starting fresh): {e}")

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
    # Updates proportional to buffer size, but capped
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

        # 2. Train Policy Net (Cross Entropy / KL) with Linear CFR weighting
        if len(policy_replay) >= batch_size:
            batch = policy_replay.sample(batch_size)
            # Support both legacy (state, target) and new (state, target, weight) formats
            states_np = np.array([x[0] for x in batch])
            targets_np = np.array([x[1] for x in batch])
            weights_np = np.array([float(x[2]) if len(x) > 2 else 1.0 for x in batch], dtype=np.float32)

            states = torch.tensor(states_np, dtype=torch.float32).to(device)
            targets = torch.tensor(targets_np, dtype=torch.float32).to(device)
            weights = torch.tensor(weights_np, dtype=torch.float32).to(device)

            # Normalize weights
            weights = weights / (weights.mean() + 1e-8)
            
            deep_cfr.policy_optimizer.zero_grad()

            def weighted_kl_loss(logits, targets, weights):
                log_probs = torch.log_softmax(logits, dim=1)
                log_targets = torch.log(targets + 1e-8)
                sample_kl = (targets * (log_targets - log_probs)).sum(dim=1)
                weighted = sample_kl * weights
                return weighted.sum() / (weights.sum() + 1e-8)

            if use_amp:
                with autocast('cuda'):
                    logits = policy_net(states)
                    loss = weighted_kl_loss(logits, targets, weights)
                scaler.scale(loss).backward()
                scaler.unscale_(deep_cfr.policy_optimizer)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                scaler.step(deep_cfr.policy_optimizer)
            else:
                logits = policy_net(states)
                loss = weighted_kl_loss(logits, targets, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                deep_cfr.policy_optimizer.step()

            policy_losses.append(loss.item())
            
        if use_amp:
            scaler.update()
            
    # Step schedulers
    adv_scheduler.step()
    pol_scheduler.step()

    # Prepare updated checkpoint
    checkpoint = {
        'iteration': iteration,
        'advantage_net_state': advantage_net.state_dict(),
        'policy_net_state': policy_net.state_dict(),
        'advantage_optimizer_state': deep_cfr.advantage_optimizer.state_dict(),
        'policy_optimizer_state': deep_cfr.policy_optimizer.state_dict(),
    }
    
    # Save checkpoint atomically to RUN DIR
    checkpoint_file = f"{run_dir}/checkpoint_iter_{iteration}.pt"
    temp_checkpoint_file = f"{run_dir}/checkpoint_iter_{iteration}.pt.tmp"
    
    try:
        torch.save(checkpoint, temp_checkpoint_file)
        os.rename(temp_checkpoint_file, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Also copy to root
        root_checkpoint = f"/checkpoints/checkpoint_iter_{iteration}.pt"
        import shutil
        shutil.copy2(checkpoint_file, root_checkpoint)
        
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
    cpu=2,
    memory=4096,
    timeout=86400,  # 24 hours
)
def training_worker(
    num_iterations: int,
    trajectories_per_iteration: int,
    num_workers: int,
    start_iteration: int = 0,
    batch_size: int = 32,
    run_id: str = "default"
):
    """Main training worker that runs asynchronously."""
    import logging
    import sys
    from modal_deploy.metrics import MetricsLogger
    
    run_dir = f"/checkpoints/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = f'{run_dir}/training.log'
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
    
    metrics_logger = MetricsLogger(log_dir=run_dir)
    
    logger.info(f"Starting training run: {run_id}")
    logger.info(f"Config: {num_iterations} iterations, {trajectories_per_iteration} trajectories/iter")
    logger.info(f"Starting from iteration {start_iteration}")
    
    iteration = start_iteration
    
    if num_workers > trajectories_per_iteration:
        logger.error(f"num_workers ({num_workers}) > trajectories_per_iteration ({trajectories_per_iteration}). This will result in 0 trajectories per worker!")
        raise ValueError(f"num_workers must be <= trajectories_per_iteration")
    
    try:
        for iteration in range(start_iteration, num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")
            
            checkpoint_path = None
            if iteration > 0:
                checkpoint_path = f"{run_id}/checkpoint_iter_{iteration - 1}.pt"
            elif iteration == 0 and os.path.exists("/checkpoints/checkpoint_iter_19.pt"):
                logger.info("Bootstrapping from /checkpoints/checkpoint_iter_19.pt")
                checkpoint_path = "checkpoint_iter_19.pt"
            elif start_iteration > 0 and iteration == start_iteration:
                # Look for legacy checkpoint if new structure missing
                legacy_path = f"checkpoint_iter_{iteration - 1}.pt"
                if os.path.exists(f"/checkpoints/{legacy_path}"):
                    checkpoint_path = legacy_path
                    logger.info(f"Resuming from legacy checkpoint: {legacy_path}")
                else:
                     logger.warning(f"Could not find resume checkpoint for iter {iteration-1}")
            
            logger.info(f"Generating {trajectories_per_iteration} trajectories using {num_workers} workers...")
            
            trajectory_futures = []
            for worker_id in range(num_workers):
                worker_trajectories = trajectories_per_iteration // num_workers + (1 if worker_id < (trajectories_per_iteration % num_workers) else 0)
                future = generate_trajectories.spawn(
                    num_trajectories=worker_trajectories,
                    checkpoint_path=checkpoint_path,
                    iteration=iteration
                )
                trajectory_futures.append(future)
            
            all_worker_buffers = []
            max_worker_retries = 2
            total_trajectories = 0
            
            for i in range(num_workers):
                retry_count = 0
                worker_success = False
                
                while retry_count < max_worker_retries and not worker_success:
                    try:
                        if retry_count == 0:
                            future = trajectory_futures[i]
                        else:
                            logger.warning(f"Worker {i+1} failed (attempt {retry_count}). Retrying...")
                            worker_trajectories = trajectories_per_iteration // num_workers + (1 if i < (trajectories_per_iteration % num_workers) else 0)
                            future = generate_trajectories.spawn(
                                num_trajectories=worker_trajectories,
                                checkpoint_path=checkpoint_path,
                                iteration=iteration
                            )
                        
                        buffers = future.get(timeout=3600)
                        all_worker_buffers.append(buffers)
                        total_trajectories += len(buffers['advantage']) # Approx
                        logger.info(f"Worker {i+1}/{num_workers} completed: {len(buffers['advantage'])} samples")
                        worker_success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_worker_retries:
                            logger.error(f"Worker {i+1} failed: {e}")
                            break
            
            logger.info(f"Total worker buffers collected: {len(all_worker_buffers)}")
            
            if len(all_worker_buffers) == 0:
                logger.error("No data generated! Skipping.")
                continue
            
            logger.info("Training networks on GPU...")
            try:
                train_result = train_networks.spawn(
                    worker_buffers=all_worker_buffers,
                    checkpoint_path=checkpoint_path,
                    iteration=iteration,
                    batch_size=batch_size,
                    run_id=run_id
                ).get(timeout=7200)
            except Exception as e:
                logger.error(f"Training networks failed: {e}")
                raise
            
            metrics = {
                "iteration": iteration + 1,
                "trajectories_generated": total_trajectories,
                "advantage_loss": train_result.get('advantage_loss', 0.0),
                "policy_loss": train_result.get('policy_loss', 0.0),
                "checkpoint_path": train_result.get('checkpoint_path', ''),
                "num_updates": train_result.get('num_updates', 0)
            }
            
            try:
                metrics_logger.log_iteration(iteration + 1, metrics)
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
            
            logger.info(f"Training complete: Adv Loss={metrics['advantage_loss']:.4f}, Pol Loss={metrics['policy_loss']:.4f}")
            
            try:
                checkpoint_volume.commit()
            except Exception as e:
                logger.warning(f"Volume commit failed: {e}")
    
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
        raise
    finally:
        logger.info("Training worker finished.")

@app.local_entrypoint()
def main(
    num_iterations: int = 1000,
    trajectories_per_iteration: int = 20000,
    num_workers: int = 4,
    resume_from: int = None,
    batch_size: int = 32,
    deploy: bool = False
):
    """Main entrypoint for training."""
    start_iteration = resume_from if resume_from else 0
    
    import datetime
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Initializing Training Run: {run_id}")
    print(f"Phase 4: Reservoir Sampling (Policy Buffer Capacity: 2,000,000)")
    
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
            batch_size=batch_size,
            run_id=run_id
        )
        print(f"Job triggered: {future.object_id}")
        print(f"Monitor logs with: modal app logs poker-bot-training")
        
    else:
        print(f"Starting local training...")
        training_worker.remote(
            num_iterations=num_iterations,
            trajectories_per_iteration=trajectories_per_iteration,
            num_workers=num_workers,
            start_iteration=start_iteration,
            batch_size=batch_size,
            run_id=run_id
        )

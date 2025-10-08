import mujoco
import mujoco.viewer
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==================== PPO NEURAL NETWORK ====================

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Learnable log std for exploration
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        features = self.shared(state)
        return features
    
    def act(self, state, deterministic=False):
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        
        if deterministic:
            return action_mean, None, None
        
        # Sample action from Gaussian distribution
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, dist.entropy().sum(dim=-1)
    
    def evaluate(self, state, action):
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        value = self.critic(features).squeeze(-1)
        
        return log_prob, value, entropy


# ==================== PPO AGENT ====================

class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, _ = self.policy.act(state_tensor, deterministic)
            features = self.policy.forward(state_tensor)
            value = self.policy.critic(features).squeeze()
        
        return action.squeeze(0).numpy(), log_prob.item() if log_prob is not None else 0, value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        rewards = self.buffer['rewards']
        values = self.buffer['values'] + [next_value]
        dones = self.buffer['dones']
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, next_value, epochs=10, batch_size=64):
        """Update policy using PPO"""
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(self.buffer['states'])
        actions = torch.FloatTensor(self.buffer['actions'])
        old_log_probs = torch.FloatTensor(self.buffer['log_probs'])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch updates
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        # Clear buffer
        for key in self.buffer:
            self.buffer[key] = []
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Model loaded from {path}")


# ==================== ENVIRONMENT ====================

class Go2BackwardEnv:
    """MuJoCo environment for Unitree Go2 backward walking"""
    
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Find leg joints
        self.leg_joints = {}
        for i in range(self.model.njnt):
            jname = self.model.joint(i).name
            for leg_name in ['FL_', 'FR_', 'RL_', 'RR_']:
                if jname.startswith(leg_name):
                    self.leg_joints.setdefault(leg_name[:-1], []).append(i)
        
        for leg in self.leg_joints:
            self.leg_joints[leg] = sorted(self.leg_joints[leg], 
                                         key=lambda j: self.model.jnt_qposadr[j])
        
        # FIXED: Changed from 35 to 33 to match actual observation size
        self.state_dim = 33  # State representation (was 35, causing the error)
        self.action_dim = 12  # 12 joint torques
        self.max_torque = 23.7  # Motor limit
        
        self.reset()
    
    def reset(self):
        """Reset to standing pose"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set standing pose
        for leg, joints in self.leg_joints.items():
            if len(joints) >= 3:
                self.data.qpos[self.model.jnt_qposadr[joints[0]]] = 0.0
                self.data.qpos[self.model.jnt_qposadr[joints[1]]] = 0.8
                self.data.qpos[self.model.jnt_qposadr[joints[2]]] = -1.5
        
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.episode_reward = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get state observation"""
        base_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]
        
        # Orientation (roll and pitch)
        roll = 2 * (base_quat[0] * base_quat[1] + base_quat[2] * base_quat[3])
        pitch = 1 - 2 * (base_quat[1]**2 + base_quat[2]**2)
        
        # Construct observation: total = 33 elements
        # 1 (height) + 2 (orientation) + 3 (lin_vel) + 3 (ang_vel) + 12 (joint_pos) + 12 (joint_vel) = 33
        obs = np.concatenate([
            [base_height],           # 1
            [roll, pitch],           # 2
            self.data.qvel[:3],      # 3 - Base linear velocity
            self.data.qvel[3:6],     # 3 - Base angular velocity
            self.data.qpos[7:],      # 12 - Joint positions
            self.data.qvel[6:],      # 12 - Joint velocities
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Clip and apply torques
        action = np.clip(action, -1.0, 1.0) * self.max_torque
        self.data.ctrl[:] = action
        
        # Simulate
        mujoco.mj_step(self.model, self.data)
        
        # Get next state
        next_obs = self._get_obs()
        
        # Compute reward
        reward, info = self._compute_reward()
        
        # Check termination
        done = self._is_done()
        
        self.step_count += 1
        self.episode_reward += reward
        
        return next_obs, reward, done, info
    
    def _compute_reward(self):
        """Reward function for backward walking"""
        # Velocity reward (negative X velocity = backward)
        vel_x = self.data.qvel[0]
        velocity_reward = -vel_x * 2.0  # Reward for moving backward
        
        # Height penalty (want to stay upright)
        target_height = 0.28
        height = self.data.qpos[2]
        height_reward = -abs(height - target_height) * 3.0
        
        # Orientation penalty
        base_quat = self.data.qpos[3:7]
        roll = 2 * (base_quat[0] * base_quat[1] + base_quat[2] * base_quat[3])
        pitch = 1 - 2 * (base_quat[1]**2 + base_quat[2]**2)
        orientation_reward = -(abs(roll) + abs(pitch - 1.0)) * 2.0
        
        # Energy penalty (encourage efficiency)
        energy_penalty = -0.001 * np.sum(np.square(self.data.ctrl))
        
        # Alive bonus
        alive_bonus = 1.0
        
        # Total reward
        reward = (velocity_reward + height_reward + orientation_reward + 
                 energy_penalty + alive_bonus)
        
        info = {
            'vel_x': vel_x,
            'height': height,
            'velocity_reward': velocity_reward,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward
        }
        
        return reward, info
    
    def _is_done(self):
        """Check if episode should terminate"""
        # Fall detection
        if self.data.qpos[2] < 0.15:
            return True
        
        # Timeout
        if self.step_count >= 1000:  # ~10 seconds at 100Hz
            return True
        
        # Extreme orientation
        base_quat = self.data.qpos[3:7]
        roll = abs(2 * (base_quat[0] * base_quat[1] + base_quat[2] * base_quat[3]))
        if roll > 0.8:
            return True
        
        return False


# ==================== TRAINING ====================

def train_ppo(model_path, total_timesteps=500000, save_path="ppo_backward.pt"):
    """Train PPO agent for backward walking"""
    
    print("="*70)
    print(" TRAINING PPO FOR BACKWARD WALKING")
    print("="*70)
    
    env = Go2BackwardEnv(model_path)
    agent = PPOAgent(env.state_dim, env.action_dim, lr=3e-4)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    update_timestep = 2048  # Update every N steps
    timesteps = 0
    episode = 0
    
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print(f"Update frequency: every {update_timestep} steps\n")
    
    while timesteps < total_timesteps:
        # Collect trajectory
        for _ in range(update_timestep):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            timesteps += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                episode += 1
                
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_rewards.append(avg_reward)
                    print(f"Episode {episode:4d} | Steps: {timesteps:7d} | "
                          f"Reward: {episode_reward:7.2f} | Avg(10): {avg_reward:7.2f} | "
                          f"Length: {episode_length:4d} | Vel: {info['vel_x']:6.3f}")
                
                state = env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Update policy
        _, _, value = agent.select_action(state)
        policy_loss, value_loss = agent.update(value)
        
        # Save checkpoint
        if timesteps % 50000 == 0:
            checkpoint_path = save_path.replace('.pt', f'_{timesteps}.pt')
            agent.save(checkpoint_path)
    
    # Save final model
    agent.save(save_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(avg_rewards) > 0:
        plt.plot(np.linspace(0, len(episode_rewards), len(avg_rewards)), 
                avg_rewards, linewidth=2, label='Avg Reward (10 eps)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_training.png', dpi=150)
    plt.close()
    print("\n✓ Training complete! Plots saved to ppo_training.png")
    
    return agent


# ==================== DEPLOYMENT ====================

def deploy_ppo(model_path, policy_path, duration=20.0, show_viewer=True):
    """Deploy trained PPO policy"""
    
    print("="*70)
    print(" DEPLOYING PPO POLICY")
    print("="*70)
    
    env = Go2BackwardEnv(model_path)
    agent = PPOAgent(env.state_dim, env.action_dim)
    agent.load(policy_path)
    
    print(f"Running for {duration}s...\n")
    
    if show_viewer:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            state = env.reset()
            last_print = 0
            
            while viewer.is_running() and env.data.time < duration:
                action, _, _ = agent.select_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                
                viewer.sync()
                
                if env.data.time - last_print > 1.0:
                    print(f"[PPO] t={env.data.time:5.1f}s | x={env.data.qpos[0]:6.2f}m | "
                          f"vel_x={info['vel_x']:6.3f}m/s | height={info['height']:.3f}m | "
                          f"reward={reward:6.2f}")
                    last_print = env.data.time
                
                if done:
                    print("Episode terminated, resetting...")
                    state = env.reset()
    else:
        state = env.reset()
        while env.data.time < duration:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
    
    print("\n✓ Deployment complete!")


# ==================== MAIN ====================

if __name__ == "__main__":
    base_path = "/Users/gagandeshad/Desktop/unitree/mujoco_menagerie"
    model_path = f"{base_path}/unitree_go2/scene.xml"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit(1)
    
    print("\n" + "="*70)
    print(" UNITREE GO2 - PPO REINFORCEMENT LEARNING (BACKWARD WALKING)")
    print("="*70)
    print("\nThis will train the robot to walk backward using PPO.")
    print("Training will take ~30-60 minutes depending on your hardware.\n")
    
    # Train PPO agent
    print("\nSTEP 1: TRAINING PPO AGENT")
    print("="*70)
    train_ppo(model_path, total_timesteps=500000, save_path="ppo_backward.pt")
    
    # Deploy trained policy
    print("\n\nSTEP 2: DEPLOYING TRAINED POLICY")
    print("="*70)
    deploy_ppo(model_path, "ppo_backward.pt", duration=20.0, show_viewer=True)
    
    print("\n" + "="*70)
    print(" ✓ ALL DONE!")
    print("="*70)

"""Improved wrapper with better reward scaling"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from RL_evaluation_sim import MinimalSimulation

class ImprovedOT2Env(gym.Env):
    
    def get_true_position(self):
        return np.array(self.sim.get_pipette_position(), dtype=np.float32)

        
    def __init__(self, max_steps=500):
        super().__init__()
        
        self.max_steps = max_steps
        self.sim = MinimalSimulation(render=False)
        
        self.bounds = {
            'x': (-0.187, 0.253),
            'y': (-0.171, 0.220),
            'z': (0.170, 0.290)
        }
        
        self.observation_space = spaces.Box(
            low=np.array([-0.3, -0.3, 0.1, -0.3, -0.3, 0.1, -1.0, -1.0, -1.0]),
            high=np.array([0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(3,), dtype=np.float32
        )
        
        self.current_step = 0
        self.target = None
        self.prev_dist = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim.close()
        self.sim = MinimalSimulation(render=False)
        
        self.target = np.array([
            np.random.uniform(*self.bounds['x']),
            np.random.uniform(*self.bounds['y']),
            np.random.uniform(*self.bounds['z'])
        ], dtype=np.float32)
        
        current_pos = np.array(self.sim.get_pipette_position(), dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.target - current_pos)
        self.current_step = 0
        
        return self._get_obs(current_pos), {}
    
    def step(self, action):
        self.current_step += 1
        
        current_pos = np.array(self.sim.run(action), dtype=np.float32)
        dist = np.linalg.norm(self.target - current_pos)
        
        reward = self._calculate_reward(dist)
        self.prev_dist = dist
        
        terminated = bool(dist < 0.001)  # 1mm
        truncated = bool(self.current_step >= self.max_steps)
        
        return self._get_obs(current_pos), reward, terminated, truncated, {'distance': dist}
    
    def _get_obs(self, current_pos):
        error = self.target - current_pos
        return np.concatenate([current_pos, self.target, error], dtype=np.float32)
    
    def _calculate_reward(self, dist):
        """IMPROVED reward function with better scaling"""
        
        # Distance in millimeters (easier to interpret)
        dist_mm = dist * 1000
        
        # Main penalty: exponential to encourage getting very close
        # -100 at 100mm, -10 at 10mm, -1 at 1mm
        distance_penalty = -dist_mm / 10
        
        # Progress reward: strong signal for improvement
        progress_mm = (self.prev_dist - dist) * 1000
        progress_reward = progress_mm * 0.5  # Scale: +5 for 10mm improvement
        
        # Time penalty: matters but doesn't dominate
        time_penalty = -0.05
        
        # Tiered success bonuses (more aggressive)
        success_bonus = 0
        if dist_mm < 10:   # Within 10mm
            success_bonus = 10
        if dist_mm < 5:    # Within 5mm
            success_bonus = 25
        if dist_mm < 1:    # Within 1mm - SUCCESS
            success_bonus = 100
        
        total = distance_penalty + progress_reward + time_penalty + success_bonus
        
        return total
    
    def close(self):
        self.sim.close()

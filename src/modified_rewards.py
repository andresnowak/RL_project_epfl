import gymnasium as gym
import numpy as np

class MountainCarContinuousRewardWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.step_number = 0
        self.prev_pos = None
        self.prev_vel = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.prev_pos = observation[0]
        self.prev_vel = observation[1]
        self.step_number = 0
    
        return observation, info

    def potential(self, position, velocity):
        # Tunable constants
        C1 = 10.0  # Weight for potential energy (height)
        C2 = 1.0  # Weight for kinetic energy (velocity^2)
        # Height approximation based on environment dynamics
        height = np.sin(3 * position)
        # Kinetic energy approximation
        kinetic_energy = 0.5 * velocity**2  # Mass implicitly handled by scaling C2
        return C1 * height + C2 * kinetic_energy

    def reward(self, reward, obs):
        # Note: `reward` argument is the original reward from the underlying env
        # We will mostly ignore it, except for the +100 goal bonus.

        position, velocity = obs
        goal_reached = position >= 0.45

        # Original reward (only the goal part matters)
        original_reward = 50.0 if goal_reached else 0.0

        # Calculate potentials for shaping
        prev_potential = self.potential(self.prev_pos, self.prev_vel)
        current_potential = self.potential(position, velocity)

        # Update history for next step
        self.prev_pos = position
        self.prev_vel = velocity

        step_reward = (1000 / self.step_number) * 15 if goal_reached else 0

        # Calculate shaped reward
        shaped_reward = (
            original_reward + step_reward + self.gamma * current_potential - prev_potential
        )

        # We removed the default action penalty implicitly by calculating from potential
        return shaped_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["original_reward"] = reward  # Store original reward

        self.step_number += 1

        reward = self.reward(reward, obs)

        return obs, reward, terminated, truncated, info

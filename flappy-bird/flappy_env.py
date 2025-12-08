import random
import pygame
import numpy as np
import gymnasium as gym

from dataclasses import dataclass
from typing import Tuple, List
from gymnasium import spaces

@dataclass
class MetaData:
    render_modes: Tuple[str, str] = ("human", "rgb_array")
    render_fps: int = 60

@dataclass
class FlappyBirdEnvConfig:
    window_width: int = 600
    window_height: int = 500
    window_size: Tuple[int, int] = (window_width, window_height)
    gravity: float = 0.25
    flap_strength: float = -2.5
    max_velocity: float = 8.0
    pipe_speed: float = 2.0
    pipe_gap_size: int = 150
    pipe_frequency: int = 1500 # ms
    pipe_dist_spawn: int = 200 # pixels between pipes
    reward_survive: float = 0.1
    reward_pass_pipe: float = 5.0
    reward_collision: float = -100.0

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        
        # Bird properties
        self.bird_x = 50
        self.bird_radius = 15
        self.bird_color = (255, 255, 0)
        
        # Pipe properties
        self.pipe_width = 50
        self.pipe_color = (0, 255, 0)
        
        # RL Spaces

        # Action: 
        # 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation: 
        # [bird_y, bird_vel] + 3 * [dist_x, dist_y] for next 3 pipes
        # Total 8 values
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset bird
        self.bird_y = FlappyBirdEnvConfig.window_height // 2
        self.bird_vel = 0
        
        # Reset pipes
        self.pipes = []
        self.last_pipe_x = 0
        self._spawn_pipe(start_offset=0)
        
        self.score = 0
        self.frames_survived = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def step(self, action):
        # 1. Update Bird
        if action == 1:
            self.bird_vel = FlappyBirdEnvConfig.flap_strength
        
        self.bird_vel += FlappyBirdEnvConfig.gravity
        # Clamp velocity
        self.bird_vel = min(self.bird_vel, FlappyBirdEnvConfig.max_velocity)
        self.bird_vel = max(self.bird_vel, -FlappyBirdEnvConfig.max_velocity)

        self.bird_y += self.bird_vel

        # 2. Update Pipes
        remove_indices = []
        
        for i, pipe in enumerate(self.pipes):
            pipe['x'] -= FlappyBirdEnvConfig.pipe_speed
            
            # Check if passed
            if not pipe['passed'] and pipe['x'] + self.pipe_width < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                
            # Mark for removal
            if pipe['x'] + self.pipe_width < 0:
                remove_indices.append(i)
                
        for i in sorted(remove_indices, reverse=True):
            del self.pipes[i]

        # Spawn new pipe if needed
        if len(self.pipes) > 0:
            last_pipe = self.pipes[-1]
            if FlappyBirdEnvConfig.window_width - last_pipe['x'] > FlappyBirdEnvConfig.pipe_dist_spawn:
                self._spawn_pipe()
        else:
            # If no pipes, spawn one
            self._spawn_pipe()

        # 3. Check Collisions
        terminated = False
        reward = FlappyBirdEnvConfig.reward_survive
        
        # Ground/Ceiling collision
        if self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > FlappyBirdEnvConfig.window_height:
            terminated = True
            
        # Pipe collision
        bird_rect = pygame.Rect(self.bird_x - self.bird_radius, self.bird_y - self.bird_radius, 
                                self.bird_radius * 2, self.bird_radius * 2)
        
        if not terminated:
            for pipe in self.pipes:
                # Top pipe rect
                top_rect = pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['gap_y'])
                # Bottom pipe rect
                bottom_rect_y = pipe['gap_y'] + FlappyBirdEnvConfig.pipe_gap_size
                bottom_rect = pygame.Rect(pipe['x'], bottom_rect_y, self.pipe_width, FlappyBirdEnvConfig.window_height - bottom_rect_y)
                
                if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                    terminated = True
                    
                    break
        
        # 4. Return
        if terminated:
            reward = FlappyBirdEnvConfig.reward_collision
        else:
            for pipe in self.pipes:
                if pipe['passed'] and not pipe.get('rewarded', False):
                    reward += FlappyBirdEnvConfig.reward_pass_pipe
                    pipe['rewarded'] = True

        self.frames_survived += 1

        truncated = False # Infinite horizon technically, but we can truncate if needed
        info = {"score": self.score}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, info

    def _spawn_pipe(self, start_offset=0):
        # Gap Y is the top of the gap.
        # Min gap y = 50, Max gap y = window_height - 50 - gap_size
        min_y = 50
        max_y = FlappyBirdEnvConfig.window_height - 50 - FlappyBirdEnvConfig.pipe_gap_size
        gap_y = random.randint(min_y, max_y)
        
        x_pos = FlappyBirdEnvConfig.window_width + start_offset
        if len(self.pipes) > 0:
             x_pos = max(x_pos, self.pipes[-1]['x'] + FlappyBirdEnvConfig.pipe_dist_spawn)
             
        self.pipes.append({
            'x': x_pos,
            'gap_y': gap_y,
            'passed': False,
            'rewarded': False
        })

    def _get_obs(self):
        # Get next 3 pipes
        pipes_obs = []
        
        # Filter pipes that are ahead of the bird (or overlapping)
        future_pipes = [p for p in self.pipes if p['x'] + self.pipe_width > self.bird_x]
        
        for i in range(3):
            if i < len(future_pipes):
                pipe = future_pipes[i]
                dist_x = pipe['x'] - self.bird_x
                gap_center_y = pipe['gap_y'] + (FlappyBirdEnvConfig.pipe_gap_size / 2)
                dist_y = gap_center_y - self.bird_y
                
                # Normalize
                pipes_obs.append(dist_x / FlappyBirdEnvConfig.window_width)
                pipes_obs.append(dist_y / FlappyBirdEnvConfig.window_height)
            else:
                # If no pipe, assume far away and zero vertical diff
                pipes_obs.append(1.0) # Max distance
                pipes_obs.append(0.0)
                
        return np.array([
            self.bird_y / FlappyBirdEnvConfig.window_height,
            self.bird_vel / FlappyBirdEnvConfig.max_velocity,
            *pipes_obs
        ], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(FlappyBirdEnvConfig.window_size)
            pygame.display.set_caption("Flappy Bird RL")
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 30)

        canvas = pygame.Surface(FlappyBirdEnvConfig.window_size)
        canvas.fill((135, 206, 235)) # Sky blue
        
        # Draw Pipes
        for pipe in self.pipes:
            # Top pipe
            pygame.draw.rect(canvas, self.pipe_color, 
                             (pipe['x'], 0, self.pipe_width, pipe['gap_y']))
            # Bottom pipe
            bottom_y = pipe['gap_y'] + FlappyBirdEnvConfig.pipe_gap_size
            pygame.draw.rect(canvas, self.pipe_color,
                             (pipe['x'], bottom_y, self.pipe_width, FlappyBirdEnvConfig.window_height - bottom_y))
            
        # Draw Bird
        pygame.draw.circle(canvas, self.bird_color, (int(self.bird_x), int(self.bird_y)), self.bird_radius)
        
        # Draw Score
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(MetaData.render_fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

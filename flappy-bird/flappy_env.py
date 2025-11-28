import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        
        # Config
        self.window_width = 600
        self.window_height = 500
        self.window_size = (self.window_width, self.window_height)
        self.gravity = 0.25
        self.flap_strength = -6.0
        self.max_velocity = 10.0
        self.pipe_speed = 2.0
        self.pipe_gap_size = 150
        self.pipe_frequency = 1500 # ms (not used directly in step, using distance)
        self.pipe_dist_spawn = 250 # pixels between pipes
        
        # Bird properties
        self.bird_x = 50
        self.bird_radius = 15
        self.bird_color = (255, 255, 0)
        
        # Pipe properties
        self.pipe_width = 50
        self.pipe_color = (0, 255, 0)
        
        # RL Spaces
        # Action: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation: [bird_y, bird_vel, dist_to_next_pipe, next_pipe_gap_y]
        # We use -inf to inf for simplicity, though values are bounded in practice
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset bird
        self.bird_y = self.window_height // 2
        self.bird_vel = 0
        
        # Reset pipes
        self.pipes = []
        self.last_pipe_x = 0
        self._spawn_pipe(start_offset=0) # First pipe right at the edge to give immediate feedback
        
        self.score = 0
        self.frames_survived = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def step(self, action):
        # 1. Update Bird
        if action == 1:
            self.bird_vel = self.flap_strength
        
        self.bird_vel += self.gravity
        # Clamp velocity
        if self.bird_vel > self.max_velocity:
            self.bird_vel = self.max_velocity
            
        self.bird_y += self.bird_vel
        
        # 2. Update Pipes
        remove_indices = []
        for i, pipe in enumerate(self.pipes):
            pipe['x'] -= self.pipe_speed
            
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
        last_pipe = self.pipes[-1]
        if self.window_width - last_pipe['x'] > self.pipe_dist_spawn:
            self._spawn_pipe()

        # 3. Check Collisions
        terminated = False
        reward = 1.0 # Survival reward
        
        # Ground/Ceiling collision
        if self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > self.window_height:
            terminated = True
            reward = -100.0
            
        # Pipe collision
        bird_rect = pygame.Rect(self.bird_x - self.bird_radius, self.bird_y - self.bird_radius, 
                                self.bird_radius * 2, self.bird_radius * 2)
        
        for pipe in self.pipes:
            # Top pipe rect
            top_rect = pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['gap_y'])
            # Bottom pipe rect
            bottom_rect_y = pipe['gap_y'] + self.pipe_gap_size
            bottom_rect = pygame.Rect(pipe['x'], bottom_rect_y, self.pipe_width, self.window_height - bottom_rect_y)
            
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                terminated = True
                reward = -100.0
                break
        
        # Bonus for passing pipe (handled in update loop, but let's add to reward here if just passed)
        # We need to track if we JUST passed a pipe this step.
        # Actually, let's check the 'passed' flag transition.
        # Re-iterating to find if we just passed one.
        # Ideally we do this in the update loop.
        # Let's refine the update loop logic slightly to add reward there.
        
        # Refined reward logic:
        # Reset reward to 0.1 (small survival reward)
        reward = 0.1
        if terminated:
            reward = -100.0
        else:
            # Check for passing pipes
            for pipe in self.pipes:
                # We use a slightly different check here to ensure we only count it once per step
                # The 'passed' flag is set in the update loop above.
                # Let's move the score update and reward addition here or track it.
                # To be safe, let's just check if we are exactly in the frame where we passed it.
                # Or better, check if pipe['passed'] was False at start of step and True now.
                # For simplicity, I'll trust the update loop above set 'passed' = True.
                # But I need to know if it happened THIS step.
                # I'll modify the update loop to return a 'passed_pipe' boolean.
                pass

        # Let's redo the pipe update part to be cleaner about rewards
        pass_reward = 0
        for pipe in self.pipes:
             if pipe['passed'] and not pipe.get('rewarded', False):
                 pass_reward = 5.0
                 pipe['rewarded'] = True
        
        if not terminated:
            reward += pass_reward

        self.frames_survived += 1
        
        # 4. Return
        truncated = False # Infinite horizon technically, but we can truncate if needed
        info = {"score": self.score}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, info

    def _spawn_pipe(self, start_offset=0):
        # Gap Y is the top of the gap.
        # Min gap y = 50, Max gap y = window_height - 50 - gap_size
        min_y = 50
        max_y = self.window_height - 50 - self.pipe_gap_size
        gap_y = random.randint(min_y, max_y)
        
        x_pos = self.window_width + start_offset
        if len(self.pipes) > 0:
             x_pos = max(x_pos, self.pipes[-1]['x'] + self.pipe_dist_spawn)
             
        self.pipes.append({
            'x': x_pos,
            'gap_y': gap_y,
            'passed': False,
            'rewarded': False
        })

    def _get_obs(self):
        # Find the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            # Should not happen if we spawn correctly, but fallback
            dist_to_pipe = self.window_width
            gap_y = self.window_height / 2
        else:
            dist_to_pipe = next_pipe['x'] - self.bird_x
            gap_y = next_pipe['gap_y'] + (self.pipe_gap_size / 2) # Center of gap
            
        # Normalize observations for better neural network convergence
        # Neural networks work best when inputs are in a similar range (e.g., 0 to 1 or -1 to 1)
        return np.array([
            self.bird_y / self.window_height,
            self.bird_vel / self.max_velocity,
            dist_to_pipe / self.window_width,
            gap_y / self.window_height
        ], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Flappy Bird RL")
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 30)

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235)) # Sky blue
        
        # Draw Pipes
        for pipe in self.pipes:
            # Top pipe
            pygame.draw.rect(canvas, self.pipe_color, 
                             (pipe['x'], 0, self.pipe_width, pipe['gap_y']))
            # Bottom pipe
            bottom_y = pipe['gap_y'] + self.pipe_gap_size
            pygame.draw.rect(canvas, self.pipe_color,
                             (pipe['x'], bottom_y, self.pipe_width, self.window_height - bottom_y))
            
        # Draw Bird
        pygame.draw.circle(canvas, self.bird_color, (int(self.bird_x), int(self.bird_y)), self.bird_radius)
        
        # Draw Score
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

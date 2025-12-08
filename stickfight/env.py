import math
import random
import pygame
import pymunk
import numpy as np
import gymnasium as gym

from stickman import Stickman
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

@dataclass
class MetaData:
    RENDER_MODES: tuple[str, str] = ("human", "rgb_array")
    RENDER_FPS: int = 60

@dataclass
class StickmanEnvConfig:
    SCREEN_WIDTH: int = 1000                # enlarged arena width
    SCREEN_HEIGHT: int = 700                # enlarged arena height
    PPM: float = 1.0                        # pixels per meter scaling (keep 1 for simplicity)
    PHYSICS_STEPS_PER_ENV_STEP: int = 6
    TIME_STEP: float = 1.0 / 120.0          # physics tick
    MAX_HP: float = 100.0
    DAMAGE_CONSTANT: float = 0.05           # scales relative velocity into damage
    KNOCKBACK_SCALE: float = 30.0
    EPISODE_LENGTH: int = 3000
    APPROACH_REWARD_SCALE: float = 0.05     # small shaping coefficient for moving limbs toward opponent vulnerable points
    DAMP_LINEAR: float = 0.995              # reduced damping to allow momentum buildup for recovery
    DAMP_ANGULAR: float = 0.98              # reduced angular damping for better joint movement
    GROUND_COLLISION_TYPE: int = 99         # Ground collision type

class StickmanFightEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None, dummy_policy: str = "random", role_swap: bool = True, opponent_policy=None):
        super().__init__()
        self.render_mode = render_mode
        self.dummy_policy = dummy_policy
        self.role_swap = role_swap  # if True randomly pick which agent is 'self' each episode
        self.self_index = 0  # 0 or 1 designating perspective agent
        self.opponent_policy = opponent_policy  # callable(obs)->action or None

        # Action space: 10 target angles in [-1, 1] for joint motors
        # 0 Neck, 
        # 1 L Shoulder, 2 R Shoulder, 
        # 3 L Elbow,    4 R Elbow,
        # 5 L Hip,      6 R Hip, 
        # 7 L Knee,     8 R Knee, 
        # 9 Spine
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        # Observation space dimension calculation:
        # Joint angles (10) + joint angular velocities (10)
        # Core: torso angle + vx + vy + feet_on_ground (4)
        # Opponent vulnerable: relative head (2) + relative torso (2) = 4
        # Opponent limbs: relative l_hand,r_hand,l_foot,r_foot (4 * 2 = 8)
        # HP bars: 2
        obs_dim = 10 + 10 + 4 + 4 + 8 + 2

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.space = pymunk.Space()
        self.space.gravity = (0, -50)

        # Ground static body (y=50 acts as floor) with improved physics properties
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(ground_body, (0, 50), (StickmanEnvConfig.SCREEN_WIDTH, 50), 5)
        ground_shape.friction = 1.5  # increased friction for better grip
        ground_shape.elasticity = 0.1  # slight bounce to help with push-off
        ground_shape.collision_type = StickmanEnvConfig.GROUND_COLLISION_TYPE
        self.space.add(ground_body, ground_shape)

        # Add vertical walls and a ceiling to enlarge & bound arena
        wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        left_wall = pymunk.Segment(wall_body, (0, 50), (0, StickmanEnvConfig.SCREEN_HEIGHT - 10), 5)
        right_wall = pymunk.Segment(wall_body, (StickmanEnvConfig.SCREEN_WIDTH, 50), (StickmanEnvConfig.SCREEN_WIDTH, StickmanEnvConfig.SCREEN_HEIGHT - 10), 5)
        ceiling = pymunk.Segment(wall_body, (0, StickmanEnvConfig.SCREEN_HEIGHT - 10), (StickmanEnvConfig.SCREEN_WIDTH, StickmanEnvConfig.SCREEN_HEIGHT - 10), 5)
        
        for s in (left_wall, right_wall, ceiling):
            s.friction = 1.5  # increased friction for better grip
            s.elasticity = 0.1  # slight bounce to help with recovery
            s.collision_type = StickmanEnvConfig.GROUND_COLLISION_TYPE  # reuse for simplicity (feet may touch walls/ceiling)
        
        self.space.add(wall_body, left_wall, right_wall, ceiling)

        self.agent_a: Optional[Stickman] = None
        self.agent_b: Optional[Stickman] = None

        self.hp_a = StickmanEnvConfig.MAX_HP
        self.hp_b = StickmanEnvConfig.MAX_HP
        self.steps = 0

        # Damage bookkeeping each step (per agent)
        self.damage_dealt = [0.0, 0.0]
        self.damage_taken = [0.0, 0.0]
        self._prev_approach_dist = 0.0

        # Pygame initialization if rendering
        self.screen = None
        self.clock = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((StickmanEnvConfig.SCREEN_WIDTH, StickmanEnvConfig.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        # Collision handlers for damage
        self._setup_collision_handlers()
        self._setup_ground_handlers()

    def _aggregate_limb_distance(self) -> float:
        """Distance between self agent's torso and opponent's torso."""
        if not (self.agent_a and self.agent_b):
            return 0.0
        
        self_agent = self.agent_a if self.self_index == 0 else self.agent_b
        opp_agent = self.agent_b if self.self_index == 0 else self.agent_a
        
        # Use torso position for approach reward
        p1 = self_agent.parts['upper_torso'].position
        p2 = opp_agent.parts['upper_torso'].position
        dist = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

        return dist

    def _setup_collision_handlers(self):
        # Use Space.on_collision per Pymunk 7.2+ API
        self.space.on_collision(1, 4, post_solve=self._make_damage_callback(attacker_index=0, victim_index=1))
        self.space.on_collision(3, 2, post_solve=self._make_damage_callback(attacker_index=1, victim_index=0))

    def _setup_ground_handlers(self):
        # All body parts can provide ground contact for reaction forces
        # Agent 0: limb=1 vulnerable=2; Agent 1: limb=3 vulnerable=4
        for collision_type in [1, 2, 3, 4]:  # all stickman parts
            self.space.on_collision(StickmanEnvConfig.GROUND_COLLISION_TYPE, collision_type, begin=self._ground_begin, separate=self._ground_separate)

    def _ground_begin(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data):
        for shape in arbiter.shapes:
            # Any body part touching ground counts (not just feet)
            if shape.collision_type in [1, 2]:  # Agent A parts
                self.agent_a.mark_ground_contact(True, shape)
            elif shape.collision_type in [3, 4]:  # Agent B parts
                self.agent_b.mark_ground_contact(True, shape)
        # In Pymunk 7+, to reject collisions set arbiter.process_collision = False
        # We keep normal collision processing.
        return None

    def _ground_separate(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data):
        for shape in arbiter.shapes:
            # Any body part leaving ground counts
            if shape.collision_type in [1, 2]:  # Agent A parts
                self.agent_a.mark_ground_contact(False, shape)
            elif shape.collision_type in [3, 4]:  # Agent B parts
                self.agent_b.mark_ground_contact(False, shape)
        return None

    def _make_damage_callback(self, attacker_index: int, victim_index: int):
        def _cb(arbiter: pymunk.Arbiter, space: pymunk.Space, data):
            # relative velocity magnitude
            s1, s2 = arbiter.shapes
            v1 = s1.body.velocity
            v2 = s2.body.velocity
            rel_v = (v1 - v2).length
            damage = rel_v * StickmanEnvConfig.DAMAGE_CONSTANT

            if damage <= 0:
                return
            
            # Apply damage & knockback
            attacker = self.agent_a if attacker_index == 0 else self.agent_b
            victim = self.agent_a if victim_index == 0 else self.agent_b
            # Update per-agent damage bookkeeping
            self.damage_dealt[attacker_index] += damage
            self.damage_taken[victim_index] += damage

            if victim_index == 0:
                self.hp_a = max(0.0, self.hp_a - damage)
            else:
                self.hp_b = max(0.0, self.hp_b - damage)

            # Knockback impulse
            dir_vec = v1 - v2

            if dir_vec.length > 0:
                impulse = dir_vec.normalized() * damage * StickmanEnvConfig.KNOCKBACK_SCALE
                victim.torso.apply_impulse_at_local_point((impulse.x, impulse.y))

        return _cb

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        # Clear space except ground
        for s in list(self.space.shapes):
            if s.collision_type != StickmanEnvConfig.GROUND_COLLISION_TYPE:
                self.space.remove(s)
        for b in list(self.space.bodies):
            if b.body_type != pymunk.Body.STATIC:
                self.space.remove(b)
        # Remove any remaining constraints (KeysView, must iterate)
        for c in list(self.space.constraints):
            self.space.remove(c)

        self.agent_a = Stickman(self.space, (StickmanEnvConfig.SCREEN_WIDTH * 0.35, 200), agent_index=0)
        self.agent_b = Stickman(self.space, (StickmanEnvConfig.SCREEN_WIDTH * 0.65, 200), agent_index=1)
        self.hp_a = StickmanEnvConfig.MAX_HP
        self.hp_b = StickmanEnvConfig.MAX_HP
        self.steps = 0
        self.damage_dealt = [0.0, 0.0]
        self.damage_taken = [0.0, 0.0]
        # Randomly choose perspective if role_swap enabled
        self.self_index = random.randint(0,1) if self.role_swap else 0
        self._prev_approach_dist = self._aggregate_limb_distance()
        obs = self._get_observation()

        return obs, {}

    def step(self, action: np.ndarray):
        # Reset damage accumulators for current step (keep cumulative if desired later)
        self.damage_dealt = [0.0, 0.0]
        self.damage_taken = [0.0, 0.0]

        # Approach shaping (record distance before action)
        prev_dist = self._aggregate_limb_distance()

        # Apply action to perspective agent
        perspective_agent = self.agent_a if self.self_index == 0 else self.agent_b
        other_agent = self.agent_b if self.self_index == 0 else self.agent_a
        action = np.clip(action, -1.0, 1.0)
        perspective_agent.apply_action(action.tolist())

        # Opponent policy selection
        if self.opponent_policy is not None:
            opp_obs = self._get_opponent_observation()
            opp_action_arr, _ = self.opponent_policy(opp_obs)
            opp_action_arr = np.clip(np.asarray(opp_action_arr, dtype=np.float32), -1.0, 1.0)
            other_agent.apply_action(opp_action_arr.tolist())
        else:
            opp_action = self._dummy_action()
            other_agent.apply_action(opp_action)

        # Physics integration decoupled
        for _ in range(StickmanEnvConfig.PHYSICS_STEPS_PER_ENV_STEP):
            # Apply damping to all dynamic bodies prior to physics step
            for body in self.space.bodies:
                if body.body_type != pymunk.Body.DYNAMIC:
                    continue
                body.velocity = body.velocity * StickmanEnvConfig.DAMP_LINEAR
                body.angular_velocity = body.angular_velocity * StickmanEnvConfig.DAMP_ANGULAR
            self.space.step(StickmanEnvConfig.TIME_STEP)

        self.steps += 1

        # Compute shaping after physics
        new_dist = self._aggregate_limb_distance()
        approach_reward = (prev_dist - new_dist) * StickmanEnvConfig.APPROACH_REWARD_SCALE  # positive if moved closer
        self._prev_approach_dist = new_dist
        reward = self._compute_reward() + approach_reward
        terminated = self._is_terminated()
        truncated = self.steps >= StickmanEnvConfig.EPISODE_LENGTH
        obs = self._get_observation()
        info = {
            'hp_a': self.hp_a,
            'hp_b': self.hp_b,
            'damage_dealt_self': self.damage_dealt[self.self_index],
            'damage_taken_self': self.damage_taken[self.self_index],
            'damage_dealt_other': self.damage_dealt[1 - self.self_index],
            'approach_reward': approach_reward,
            'approach_dist': new_dist,
            'self_index': self.self_index,
        }

        return obs, reward, terminated, truncated, info

    def _dummy_action(self) -> List[float]:
        if self.dummy_policy == "random":
            return [random.uniform(-1, 1) for _ in range(self.action_space.shape[0])]
        
        # static policy
        return [0.0] * self.action_space.shape[0]

    def _compute_reward(self) -> float:
        """Compute base reward for the perspective agent.

        New rule (per user request):
        - If both agents have equal HP, apply a small negative penalty.
        - Otherwise reward the perspective agent proportionally to the (normalized) HP advantage.

        We retain a very small standing bonus to encourage agents not to stay knocked down.
        """
        hp_self = self.hp_a if self.self_index == 0 else self.hp_b
        hp_opp = self.hp_b if self.self_index == 0 else self.hp_a

        if hp_self == hp_opp:
            advantage_reward = -0.01  # tie penalty
        else:
            # Positive if self has more HP, negative if less (scaled to 0..1 range)
            advantage_reward = (hp_self - hp_opp) / StickmanEnvConfig.MAX_HP

        # Standing incentive (scaled to encourage recovery without overpowering HP advantage)
        perspective_agent = self.agent_a if self.self_index == 0 else self.agent_b
        head_y = perspective_agent.head.position.y
        
        # Calculate average foot height
        l_foot_y = perspective_agent.parts['l_lower_leg'].position.y
        r_foot_y = perspective_agent.parts['r_lower_leg'].position.y
        feet_y = (l_foot_y + r_foot_y) / 2.0
        
        height_diff = head_y - feet_y
        
        # Progressive standing bonus based on relative height (head vs feet)
        if height_diff > 110:  # fully upright (approx 2/3 of total height)
            standing_bonus = 0.01
        elif height_diff > 80:  # partially upright  
            standing_bonus = 0.005
        elif height_diff > 60 and perspective_agent.ground_contacts <= 2:  # recovering
            standing_bonus = 0.002  
        else:  # prone or too low
            standing_bonus = -0.002  # small penalty for staying down

        return advantage_reward + standing_bonus

    def _is_terminated(self) -> bool:
        return self.hp_a <= 0 or self.hp_b <= 0

    def _normalize_pos(self, x: float, y: float) -> Tuple[float, float]:
        return x / StickmanEnvConfig.SCREEN_WIDTH * 2 - 1, y / StickmanEnvConfig.SCREEN_HEIGHT * 2 - 1

    def _get_observation(self) -> np.ndarray:
        self_agent = self.agent_a if self.self_index == 0 else self.agent_b
        opp_agent = self.agent_b if self.self_index == 0 else self.agent_a

        angles, ang_vels = self_agent.joint_states()
        torso_angle, vx, vy, feet_flag = self_agent.core_state()
        com_ax, com_ay = self_agent.center_of_mass()

        opp_vuln = opp_agent.get_vulnerable_points()
        opp_limbs = opp_agent.get_limb_points()

        rel_head = (opp_vuln['head'][0] - com_ax, opp_vuln['head'][1] - com_ay)
        rel_torso = (opp_vuln['torso'][0] - com_ax, opp_vuln['torso'][1] - com_ay)
        rel_head_n = self._normalize_pos(*rel_head)
        rel_torso_n = self._normalize_pos(*rel_torso)

        limb_rel_norms = []

        for name in ['l_hand', 'r_hand', 'l_foot', 'r_foot']:
            pt = opp_limbs[name]
            rel = (pt[0] - com_ax, pt[1] - com_ay)
            limb_rel_norms.extend(self._normalize_pos(*rel))

        hp_self_norm = (self.hp_a if self.self_index == 0 else self.hp_b) / StickmanEnvConfig.MAX_HP
        hp_opp_norm = (self.hp_b if self.self_index == 0 else self.hp_a) / StickmanEnvConfig.MAX_HP

        obs = np.array([
            *angles, *ang_vels,
            torso_angle, vx, vy, feet_flag,
            *rel_head_n, *rel_torso_n,
            *limb_rel_norms,
            hp_self_norm, hp_opp_norm
        ], dtype=np.float32)
        return obs

    def _get_opponent_observation(self) -> np.ndarray:
        # Observation from opponent perspective (without changing self_index state)
        opp_index = 1 - self.self_index
        opp_agent = self.agent_a if opp_index == 0 else self.agent_b
        self_agent = self.agent_a if self.self_index == 0 else self.agent_b

        angles, ang_vels = opp_agent.joint_states()
        torso_angle, vx, vy, feet_flag = opp_agent.core_state()
        com_ax, com_ay = opp_agent.center_of_mass()

        vuln = self_agent.get_vulnerable_points()
        limbs = self_agent.get_limb_points()
        rel_head = (vuln['head'][0] - com_ax, vuln['head'][1] - com_ay)
        rel_torso = (vuln['torso'][0] - com_ax, vuln['torso'][1] - com_ay)
        rel_head_n = self._normalize_pos(*rel_head)
        rel_torso_n = self._normalize_pos(*rel_torso)
        limb_rel_norms = []

        for name in ['l_hand', 'r_hand', 'l_foot', 'r_foot']:
            pt = limbs[name]
            rel = (pt[0] - com_ax, pt[1] - com_ay)
            limb_rel_norms.extend(self._normalize_pos(*rel))

        hp_self_norm = (self.hp_a if opp_index == 0 else self.hp_b) / StickmanEnvConfig.MAX_HP
        hp_opp_norm = (self.hp_b if opp_index == 0 else self.hp_a) / StickmanEnvConfig.MAX_HP

        return np.array([
            *angles, *ang_vels,
            torso_angle, vx, vy, feet_flag,
            *rel_head_n, *rel_torso_n,
            *limb_rel_norms,
            hp_self_norm, hp_opp_norm
        ], dtype=np.float32)

    def set_opponent_policy(self, policy_callable):
        self.opponent_policy = policy_callable

    def render(self):
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return ;
        
        self.screen.fill((30, 30, 30))

        # Pymunk's coordinates have origin at bottom-left; Pygame top-left. We'll flip y.
        def to_pygame(p):
            return int(p[0]), int(StickmanEnvConfig.SCREEN_HEIGHT - p[1])

        def draw_stickman(sm: Stickman, color):
            for shape in sm.shapes:
                if isinstance(shape, pymunk.Circle):
                    pos = to_pygame(shape.body.position)
                    pygame.draw.circle(self.screen, color, pos, int(shape.radius))
                elif isinstance(shape, pymunk.Poly):
                    verts = [to_pygame(v.rotated(shape.body.angle) + shape.body.position) for v in shape.get_vertices()]
                    pygame.draw.polygon(self.screen, color, verts, 0)
                elif isinstance(shape, pymunk.Segment):
                    a = shape.body.position + shape.a.rotated(shape.body.angle)
                    b = shape.body.position + shape.b.rotated(shape.body.angle)
                    pygame.draw.line(self.screen, color, to_pygame(a), to_pygame(b), int(shape.radius * 2))

        draw_stickman(self.agent_a, (200, 200, 240))
        draw_stickman(self.agent_b, (240, 200, 120))

        # HP bars
        bar_w = 300
        bar_h = 15
        # Agent A
        a_ratio = self.hp_a / StickmanEnvConfig.MAX_HP
        b_ratio = self.hp_b / StickmanEnvConfig.MAX_HP
        pygame.draw.rect(self.screen, (80, 80, 80), pygame.Rect(40, 20, bar_w, bar_h))
        pygame.draw.rect(self.screen, (40, 180, 60), pygame.Rect(40, 20, int(bar_w * a_ratio), bar_h))
        pygame.draw.rect(self.screen, (80, 80, 80), pygame.Rect(StickmanEnvConfig.SCREEN_WIDTH - 40 - bar_w, 20, bar_w, bar_h))
        pygame.draw.rect(self.screen, (200, 60, 40), pygame.Rect(StickmanEnvConfig.SCREEN_WIDTH - 40 - bar_w, 20, int(bar_w * b_ratio), bar_h))

        pygame.display.flip()
        self.clock.tick(MetaData.RENDER_FPS)

    def close(self):
        if self.screen:
            pygame.quit()
        self.screen = None
        self.clock = None


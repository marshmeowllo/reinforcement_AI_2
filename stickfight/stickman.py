import math
import pymunk

from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BodyPartSizes:
    # Constants for body sizing (tweakable)
    HEAD_RADIUS = 15
    
    # Torso width reduced to match limb thickness
    TORSO_WIDTH = 10

    # Split torso into two sections for improved stability
    UPPER_TORSO_HEIGHT = 40
    LOWER_TORSO_HEIGHT = 20
    TORSO_HEIGHT = UPPER_TORSO_HEIGHT + LOWER_TORSO_HEIGHT
    LIMB_WIDTH = 10
    UPPER_ARM_LENGTH = 35
    LOWER_ARM_LENGTH = 35
    UPPER_LEG_LENGTH = 45
    LOWER_LEG_LENGTH = 45

    # Refined body mass distribution for more realistic inertia
    LOWER_TORSO_MASS = 6.0  # Was 4.0
    UPPER_TORSO_MASS = 5.0  # Was 3.0
    HEAD_MASS = 1.5
    UPPER_LIMB_MASS = 1.2
    LOWER_LIMB_MASS = 1.0
    
    # Motor-based control (torque/limits)
    # Max torque values (increased for better ground recovery)
    NECK_TORQUE = 500.0
    SHOULDER_TORQUE = 2000.0
    ELBOW_TORQUE = 1500.0
    HIP_TORQUE = 6000.0     # Was 4000
    KNEE_TORQUE = 5000.0
    SPINE_TORQUE = 8000.0   # Was 6000

    # Motor controller gains
    # Rebalanced: lower P to reduce saturation, higher D for damping, more rate headroom
    BASE_P_GAIN = 35.0
    DERIVATIVE_GAIN = 2.5
    MAX_MOTOR_RATE = 30.0

class Stickman:
    """Stickman ragdoll composed of multiple Pymunk bodies and joints.

    Joints (ordered for action mapping):
    0: Neck
    1: Left Shoulder
    2: Right Shoulder
    3: Left Elbow
    4: Right Elbow
    5: Left Hip
    6: Right Hip
    7: Left Knee
    8: Right Knee
    """

    def __init__(self, space: pymunk.Space, position: Tuple[float, float], agent_index: int):
        self.space = space
        self.agent_index = agent_index
        self.position = position
        self.parts = {}  # name -> body
        self.shapes = []
        # Constraints
        self.joints = []  # pivot joints etc
        self.limit_joints = []
        self.spine_joint = None
        # Motor control
        self.motors: List[pymunk.SimpleMotor] = []
        self.joint_pairs = []  # ordered body pairs for joint state extraction & action mapping (10 joints)
        self.joint_angle_limits: List[Tuple[float, float]] = []  # per-joint (min,max) aligned with action indices
        self.limb_shapes = []  # fists/feet (for damage)
        self.vulnerable_shapes = []  # head/torso
        self.feet_shapes = []  # track ground contact
        self.ground_contacts = 0
        self.ground_contact_shapes = set()  # track which specific shapes are touching ground
        
        # Diagnostic tracking
        self.step_counter = 0
        self.saturation_history = []

        # Collision type assignment.
        # Agent 0: limb=1 vulnerable=2; Agent 1: limb=3 vulnerable=4
        if agent_index == 0:
            self.limb_collision_type = 1
            self.vulnerable_collision_type = 2
        else:
            self.limb_collision_type = 3
            self.vulnerable_collision_type = 4

        self._build_ragdoll(position)

    def _add_part(self, body: pymunk.Body, shape: pymunk.Shape, name: str, vulnerable=False, limb=False, foot=False):
        # Group shapes per agent so self-collision is disabled while retaining collisions with other agents & ground.
        # User requirement: only the head should collide with the agent's own shapes. Leave head ungrouped.
        shape.filter = pymunk.ShapeFilter(group=self.agent_index + 1)
        # Add slight restitution for recovery assistance and increase friction
        shape.elasticity = 0.1 if foot else 0.05  # feet get more bounce for push-off
        shape.friction = 1.2 if foot else 1.0     # feet get extra friction for grip
        if limb:
            shape.collision_type = self.limb_collision_type
            self.limb_shapes.append(shape)
        if vulnerable:
            shape.collision_type = self.vulnerable_collision_type
            self.vulnerable_shapes.append(shape)
        if foot:
            self.feet_shapes.append(shape)
            shape.foot = True  # flag for ground handler
        self.parts[name] = body
        self.shapes.append(shape)
        self.space.add(body, shape)

    def _make_box(self, size: Tuple[float, float], pos: Tuple[float, float], mass: float) -> Tuple[pymunk.Body, pymunk.Shape]:
        width, height = size
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Poly.create_box(body, (width, height))
        return body, shape

    def _make_segment(self, length: float, pos: Tuple[float, float], mass: float, angle: float = 0.0) -> Tuple[pymunk.Body, pymunk.Shape]:
        a = (-length / 2, 0)
        b = (length / 2, 0)
        moment = pymunk.moment_for_segment(mass, a, b, BodyPartSizes.LIMB_WIDTH / 2)
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.angle = angle
        shape = pymunk.Segment(body, a, b, BodyPartSizes.LIMB_WIDTH / 2)
        return body, shape

    def _build_ragdoll(self, pos: Tuple[float, float]):
        x, y = pos
        # Lower torso
        lower_torso_body, lower_torso_shape = self._make_box((BodyPartSizes.TORSO_WIDTH, BodyPartSizes.LOWER_TORSO_HEIGHT), (x, y - (BodyPartSizes.UPPER_TORSO_HEIGHT / 2)), BodyPartSizes.LOWER_TORSO_MASS)
        self._add_part(lower_torso_body, lower_torso_shape, "lower_torso", vulnerable=True)
        # Upper torso
        upper_torso_body, upper_torso_shape = self._make_box((BodyPartSizes.TORSO_WIDTH, BodyPartSizes.UPPER_TORSO_HEIGHT), (x, y + (BodyPartSizes.LOWER_TORSO_HEIGHT / 2)), BodyPartSizes.UPPER_TORSO_MASS)
        self._add_part(upper_torso_body, upper_torso_shape, "upper_torso", vulnerable=True)
        # Legacy alias for observation & damage code
        self.parts['torso'] = upper_torso_body
        # Head
        head_mass = BodyPartSizes.HEAD_MASS
        head_moment = pymunk.moment_for_circle(head_mass, 0, BodyPartSizes.HEAD_RADIUS)
        head_body = pymunk.Body(head_mass, head_moment)
        head_body.position = (x, y + BodyPartSizes.TORSO_HEIGHT / 2 + BodyPartSizes.HEAD_RADIUS)
        head_shape = pymunk.Circle(head_body, BodyPartSizes.HEAD_RADIUS)
        self._add_part(head_body, head_shape, "head", vulnerable=True)

        # Arms
        l_upper_arm_body, l_upper_arm_shape = self._make_segment(BodyPartSizes.UPPER_ARM_LENGTH, (x - BodyPartSizes.TORSO_WIDTH / 2 - BodyPartSizes.UPPER_ARM_LENGTH / 2, y + BodyPartSizes.TORSO_HEIGHT / 2), BodyPartSizes.UPPER_LIMB_MASS)
        self._add_part(l_upper_arm_body, l_upper_arm_shape, "l_upper_arm")
        l_lower_arm_body, l_lower_arm_shape = self._make_segment(BodyPartSizes.LOWER_ARM_LENGTH, (x - BodyPartSizes.TORSO_WIDTH / 2 - BodyPartSizes.UPPER_ARM_LENGTH - BodyPartSizes.LOWER_ARM_LENGTH / 2, y + BodyPartSizes.TORSO_HEIGHT / 2), BodyPartSizes.LOWER_LIMB_MASS)
        self._add_part(l_lower_arm_body, l_lower_arm_shape, "l_lower_arm", limb=True)

        r_upper_arm_body, r_upper_arm_shape = self._make_segment(BodyPartSizes.UPPER_ARM_LENGTH, (x + BodyPartSizes.TORSO_WIDTH / 2 + BodyPartSizes.UPPER_ARM_LENGTH / 2, y + BodyPartSizes.TORSO_HEIGHT / 2), BodyPartSizes.UPPER_LIMB_MASS)
        self._add_part(r_upper_arm_body, r_upper_arm_shape, "r_upper_arm")
        r_lower_arm_body, r_lower_arm_shape = self._make_segment(BodyPartSizes.LOWER_ARM_LENGTH, (x + BodyPartSizes.TORSO_WIDTH / 2 + BodyPartSizes.UPPER_ARM_LENGTH + BodyPartSizes.LOWER_ARM_LENGTH / 2, y + BodyPartSizes.TORSO_HEIGHT / 2), BodyPartSizes.LOWER_LIMB_MASS)
        self._add_part(r_lower_arm_body, r_lower_arm_shape, "r_lower_arm", limb=True)

        # Legs
        l_upper_leg_body, l_upper_leg_shape = self._make_segment(BodyPartSizes.UPPER_LEG_LENGTH, (x - BodyPartSizes.TORSO_WIDTH / 4, y - BodyPartSizes.TORSO_HEIGHT / 2 - BodyPartSizes.UPPER_LEG_LENGTH / 2), BodyPartSizes.UPPER_LIMB_MASS, math.pi / 2)
        self._add_part(l_upper_leg_body, l_upper_leg_shape, "l_upper_leg")
        l_lower_leg_body, l_lower_leg_shape = self._make_segment(BodyPartSizes.LOWER_LEG_LENGTH, (x - BodyPartSizes.TORSO_WIDTH / 4, y - BodyPartSizes.TORSO_HEIGHT / 2 - BodyPartSizes.UPPER_LEG_LENGTH - BodyPartSizes.LOWER_LEG_LENGTH / 2), BodyPartSizes.LOWER_LIMB_MASS, math.pi / 2)
        self._add_part(l_lower_leg_body, l_lower_leg_shape, "l_lower_leg", limb=True, foot=True)

        r_upper_leg_body, r_upper_leg_shape = self._make_segment(BodyPartSizes.UPPER_LEG_LENGTH, (x + BodyPartSizes.TORSO_WIDTH / 4, y - BodyPartSizes.TORSO_HEIGHT / 2 - BodyPartSizes.UPPER_LEG_LENGTH / 2), BodyPartSizes.UPPER_LIMB_MASS, math.pi / 2)
        self._add_part(r_upper_leg_body, r_upper_leg_shape, "r_upper_leg")
        r_lower_leg_body, r_lower_leg_shape = self._make_segment(BodyPartSizes.LOWER_LEG_LENGTH, (x + BodyPartSizes.TORSO_WIDTH / 4, y - BodyPartSizes.TORSO_HEIGHT / 2 - BodyPartSizes.UPPER_LEG_LENGTH - BodyPartSizes.LOWER_LEG_LENGTH / 2), BodyPartSizes.LOWER_LIMB_MASS, math.pi / 2)
        self._add_part(r_lower_leg_body, r_lower_leg_shape, "r_lower_leg", limb=True, foot=True)

        # Joints & motors helper
        def pivot(a: pymunk.Body, b: pymunk.Body, anchor: Tuple[float, float]):
            pj = pymunk.PivotJoint(a, b, anchor)
            pj.collide_bodies = False
            self.space.add(pj)
            self.joints.append(pj)
            return pj

        def limit(a: pymunk.Body, b: pymunk.Body, min_angle: float, max_angle: float):
            lj = pymunk.RotaryLimitJoint(a, b, min_angle, max_angle)
            self.space.add(lj)
            self.limit_joints.append(lj)
            return lj

        def register_joint_pair(a: pymunk.Body, b: pymunk.Body):
            # Keep ordered list for joint state extraction
            self.joint_pairs.append((a, b))

        # Spine (lower_torso <-> upper_torso)
        pivot(upper_torso_body, lower_torso_body, (x, y))
        # Allow spine to swing both directions
        spine_limit = limit(upper_torso_body, lower_torso_body, -1.0, 1.0)
        self.spine_joint = (upper_torso_body, lower_torso_body)
        # Neck
        pivot(head_body, upper_torso_body, (x, upper_torso_body.position.y + BodyPartSizes.UPPER_TORSO_HEIGHT / 2))
        # Allow neck to swing both directions more freely
        neck_limit = limit(head_body, upper_torso_body, -1.2, 1.2)
        register_joint_pair(head_body, upper_torso_body)
        self.joint_angle_limits.append((neck_limit.min, neck_limit.max))
        # Shoulders (anchor moved to top edge to eliminate vertical gap appearance)
        shoulder_y = upper_torso_body.position.y + BodyPartSizes.UPPER_TORSO_HEIGHT / 2 - BodyPartSizes.LIMB_WIDTH / 2
        pivot(l_upper_arm_body, upper_torso_body, (upper_torso_body.position.x - BodyPartSizes.TORSO_WIDTH / 2, shoulder_y))
        l_sh_limit = limit(l_upper_arm_body, upper_torso_body, -1.5, 1.5)
        register_joint_pair(l_upper_arm_body, upper_torso_body)
        self.joint_angle_limits.append((l_sh_limit.min, l_sh_limit.max))
        pivot(r_upper_arm_body, upper_torso_body, (upper_torso_body.position.x + BodyPartSizes.TORSO_WIDTH / 2, shoulder_y))
        r_sh_limit = limit(r_upper_arm_body, upper_torso_body, -1.5, 1.5)
        register_joint_pair(r_upper_arm_body, upper_torso_body)
        self.joint_angle_limits.append((r_sh_limit.min, r_sh_limit.max))
        # Elbows
        pivot(l_lower_arm_body, l_upper_arm_body, (l_upper_arm_body.position.x - BodyPartSizes.UPPER_ARM_LENGTH / 2, l_upper_arm_body.position.y))
        # Elbows swing back and forth symmetrically
        l_el_limit = limit(l_lower_arm_body, l_upper_arm_body, -1.6, 1.6)
        register_joint_pair(l_lower_arm_body, l_upper_arm_body)
        self.joint_angle_limits.append((l_el_limit.min, l_el_limit.max))
        pivot(r_lower_arm_body, r_upper_arm_body, (r_upper_arm_body.position.x + BodyPartSizes.UPPER_ARM_LENGTH / 2, r_upper_arm_body.position.y))
        r_el_limit = limit(r_lower_arm_body, r_upper_arm_body, -1.6, 1.6)
        register_joint_pair(r_lower_arm_body, r_upper_arm_body)
        self.joint_angle_limits.append((r_el_limit.min, r_el_limit.max))
        # Hips
        pivot(l_upper_leg_body, lower_torso_body, (x - BodyPartSizes.TORSO_WIDTH / 4, lower_torso_body.position.y - BodyPartSizes.LOWER_TORSO_HEIGHT / 2))
        # Hips swing both directions
        l_hip_limit = limit(l_upper_leg_body, lower_torso_body, -1.6, 1.6)
        register_joint_pair(l_upper_leg_body, lower_torso_body)
        self.joint_angle_limits.append((l_hip_limit.min, l_hip_limit.max))
        pivot(r_upper_leg_body, lower_torso_body, (x + BodyPartSizes.TORSO_WIDTH / 4, lower_torso_body.position.y - BodyPartSizes.LOWER_TORSO_HEIGHT / 2))
        r_hip_limit = limit(r_upper_leg_body, lower_torso_body, -1.6, 1.6)
        register_joint_pair(r_upper_leg_body, lower_torso_body)
        self.joint_angle_limits.append((r_hip_limit.min, r_hip_limit.max))
        # Knees
        pivot(l_lower_leg_body, l_upper_leg_body, (l_upper_leg_body.position.x, l_upper_leg_body.position.y - BodyPartSizes.UPPER_LEG_LENGTH / 2))
        # Knees swing both directions symmetrically
        l_knee_limit = limit(l_lower_leg_body, l_upper_leg_body, -1.4, 1.4)
        register_joint_pair(l_lower_leg_body, l_upper_leg_body)
        self.joint_angle_limits.append((l_knee_limit.min, l_knee_limit.max))
        pivot(r_lower_leg_body, r_upper_leg_body, (r_upper_leg_body.position.x, r_upper_leg_body.position.y - BodyPartSizes.UPPER_LEG_LENGTH / 2))
        r_knee_limit = limit(r_lower_leg_body, r_upper_leg_body, -1.4, 1.4)
        register_joint_pair(r_lower_leg_body, r_upper_leg_body)
        self.joint_angle_limits.append((r_knee_limit.min, r_knee_limit.max))

        # Register spine as a joint pair for state (place after limbs to keep mapping intuitive)
        self.joint_pairs.append((upper_torso_body, lower_torso_body))
        # Append spine angle limits as last action index
        self.joint_angle_limits.append((spine_limit.min, spine_limit.max))

        # Springs removed: full control via motors + PD; prevents passive bias that caused action saturation.
        self.springs: List[pymunk.DampedRotarySpring] = []

        # Create motors aligned with joint_pairs/action indices
        torque_by_index = [
            BodyPartSizes.NECK_TORQUE,          # 0 neck
            BodyPartSizes.SHOULDER_TORQUE,      # 1 L shoulder
            BodyPartSizes.SHOULDER_TORQUE,      # 2 R shoulder
            BodyPartSizes.ELBOW_TORQUE,         # 3 L elbow
            BodyPartSizes.ELBOW_TORQUE,         # 4 R elbow
            BodyPartSizes.HIP_TORQUE,           # 5 L hip
            BodyPartSizes.HIP_TORQUE,           # 6 R hip
            BodyPartSizes.KNEE_TORQUE,          # 7 L knee
            BodyPartSizes.KNEE_TORQUE,          # 8 R knee
            BodyPartSizes.SPINE_TORQUE,         # 9 spine
        ]
        self.base_motor_forces: List[float] = []
        for i, (a, b) in enumerate(self.joint_pairs):
            m = pymunk.SimpleMotor(a, b, 0.0)
            base_force = torque_by_index[i]
            m.max_force = base_force
            self.base_motor_forces.append(base_force)
            self.space.add(m)
            self.motors.append(m)

        # For compatibility: expose torso as upper_torso
        self.torso = upper_torso_body
        self.lower_torso = lower_torso_body
        self.head = head_body

    def apply_action(self, action: List[float]):
        # Motor control: expect 10 target angle commands in [-1, 1]
        expected = 10
        if len(action) != expected:
            raise ValueError(f"Action length {len(action)} invalid, expected {expected}")
        
        self.step_counter += 1
        saturation_count = 0
        diagnostic_data = []

        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2 * math.pi) - math.pi

        # Sign factors so positive input means flex/raise symmetrically across sides
        control_signs = [
            1,  # neck
            -1, # L shoulder (invert)
            1,  # R shoulder
            -1, # L elbow (invert)
            1,  # R elbow
            -1, # L hip (invert)
            1,  # R hip
            -1, # L knee (invert)
            1,  # R knee
            1,  # spine
        ]

        for i, (a_body, b_body) in enumerate(self.joint_pairs):
            min_ang, max_ang = self.joint_angle_limits[i]
            mid = 0.5 * (min_ang + max_ang)
            half = 0.5 * (max_ang - min_ang)
            target = float(action[i])
            target = max(-1.0, min(1.0, target))
            signed_target = target * control_signs[i]
            target_angle = mid + half * signed_target
            curr_rel = wrap_pi(a_body.angle - b_body.angle)
            # Limit target to physical range
            if target_angle < min_ang:
                target_angle = min_ang
            elif target_angle > max_ang:
                target_angle = max_ang
            error = wrap_pi(target_angle - curr_rel)
            rel_ang_vel = a_body.angular_velocity - b_body.angular_velocity
            # PD controller: command relative angular velocity (motor.rate) proportional to angle error and
            # damped by relative angular velocity to curb overshoot.
            rate = BodyPartSizes.BASE_P_GAIN * error - BodyPartSizes.DERIVATIVE_GAIN * rel_ang_vel

            recovery_boost = 1.0
            if self.ground_contacts > 0 and i in [5, 6, 7, 8, 9]:
                # Adaptive boost based on posture and magnitude of error (reduced to prevent over-saturation)
                if self.is_prone():
                    recovery_boost = 2.0
                elif abs(error) > 0.3:
                    recovery_boost = 1.6
                else:
                    recovery_boost = 1.3
            rate *= recovery_boost
            max_rate = BodyPartSizes.MAX_MOTOR_RATE * recovery_boost
            
            # Check saturation BEFORE clamping
            is_saturated = abs(rate) >= max_rate * 0.98
            if is_saturated:
                saturation_count += 1
            
            if rate > max_rate:
                rate = max_rate
            elif rate < -max_rate:
                rate = -max_rate
            motor = self.motors[i]
            motor.rate = rate
            # Adaptive max_force: higher when far from target, lower near target to reduce jitter.
            span = max_ang - min_ang
            norm_err = abs(error) / span if span > 1e-6 else 0.0
            # Shape scaling curve (smooth ramp): base 0.4 -> 1.0 as error grows; extra boost if prone recovery.
            force_scale = 0.4 + 0.6 * min(1.0, norm_err / 0.5)
            if self.ground_contacts > 0 and i in [5,6,7,8,9] and self.is_prone():
                force_scale = min(1.4, force_scale * 1.3)
            motor.max_force = self.base_motor_forces[i] * force_scale

    def joint_states(self) -> Tuple[List[float], List[float]]:
        angles = []
        ang_vels = []
        for a, b in self.joint_pairs:
            angle = (a.angle - b.angle)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            angles.append(angle / math.pi)
            ang_vels.append((a.angular_velocity - b.angular_velocity) / 10.0)
        return angles, ang_vels

    def core_state(self) -> Tuple[float, float, float, float]:
        torso_angle = (self.torso.angle + math.pi) % (2 * math.pi) - math.pi
        torso_angle_norm = torso_angle / math.pi
        vx, vy = self.torso.velocity
        vx_norm = vx / 400.0
        vy_norm = vy / 400.0
        feet_on_ground = 1.0 if self.ground_contacts > 0 else 0.0
        return torso_angle_norm, vx_norm, vy_norm, feet_on_ground

    def get_point(self, name: str) -> Tuple[float, float]:
        body = self.parts[name]
        return body.position.x, body.position.y

    def center_of_mass(self) -> Tuple[float, float]:
        total_mass = 0.0
        cx = 0.0
        cy = 0.0
        for body in self.parts.values():
            m = body.mass
            total_mass += m
            cx += body.position.x * m
            cy += body.position.y * m
        return cx / total_mass, cy / total_mass

    def get_limb_points(self) -> dict:
        return {
            'l_hand': self.get_point('l_lower_arm'),
            'r_hand': self.get_point('r_lower_arm'),
            'l_foot': self.get_point('l_lower_leg'),
            'r_foot': self.get_point('r_lower_leg'),
        }

    def get_vulnerable_points(self) -> dict:
        return {
            'head': self.get_point('head'),
            'torso': self.get_point('torso'),
        }

    def mark_ground_contact(self, entering: bool, shape=None):
        if entering:
            self.ground_contacts += 1
            if shape:
                self.ground_contact_shapes.add(shape)
        else:
            self.ground_contacts = max(0, self.ground_contacts - 1)
            if shape and shape in self.ground_contact_shapes:
                self.ground_contact_shapes.remove(shape)
    
    def is_prone(self) -> bool:
        """Check if stickman is lying down and needs recovery assistance."""
        # If head or torso is low and touching ground, consider prone
        head_y = self.head.position.y
        torso_y = self.torso.position.y
        
        # Check if core body parts are too low (near ground level of 50)
        prone_threshold = 90  # below this height is considered prone
        return (head_y < prone_threshold or torso_y < prone_threshold) and self.ground_contacts > 0

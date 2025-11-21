import math
from typing import List, Tuple
import pymunk

# Constants for body sizing (tweakable)
HEAD_RADIUS = 15
TORSO_WIDTH = 25
# Split torso into two sections for improved stability
UPPER_TORSO_HEIGHT = 20
LOWER_TORSO_HEIGHT = 40
TORSO_HEIGHT = UPPER_TORSO_HEIGHT + LOWER_TORSO_HEIGHT
LIMB_WIDTH = 10
UPPER_ARM_LENGTH = 35
LOWER_ARM_LENGTH = 35
UPPER_LEG_LENGTH = 45
LOWER_LEG_LENGTH = 45
MASS_PER_PART = 1.0  # increased mass to reduce excessive acceleration

# Motor-based control (torque/limits)
# Max torque values (tune as needed)
NECK_TORQUE = 600.0
SHOULDER_TORQUE = 2000.0
ELBOW_TORQUE = 1500.0
HIP_TORQUE = 4000.0
KNEE_TORQUE = 4000.0
SPINE_TORQUE = 5000.0

# Proportional gain mapping angle error -> motor target angular velocity (rad/s)
# Keep conservative to avoid oscillations; can be increased after testing.
BASE_P_GAIN = 6.0
MAX_MOTOR_RATE = 12.0  # clamp motor target angular velocity

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
        shape.filter = pymunk.ShapeFilter()
        shape.elasticity = 0.0
        shape.friction = 0.9
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

    def _make_box(self, size: Tuple[float, float], pos: Tuple[float, float]) -> Tuple[pymunk.Body, pymunk.Shape]:
        mass = MASS_PER_PART
        width, height = size
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Poly.create_box(body, (width, height))
        return body, shape

    def _make_segment(self, length: float, pos: Tuple[float, float], angle: float = 0.0) -> Tuple[pymunk.Body, pymunk.Shape]:
        mass = MASS_PER_PART
        a = (-length / 2, 0)
        b = (length / 2, 0)
        moment = pymunk.moment_for_segment(mass, a, b, LIMB_WIDTH / 2)
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.angle = angle
        shape = pymunk.Segment(body, a, b, LIMB_WIDTH / 2)
        return body, shape

    def _build_ragdoll(self, pos: Tuple[float, float]):
        x, y = pos
        # Lower torso
        lower_torso_body, lower_torso_shape = self._make_box((TORSO_WIDTH, LOWER_TORSO_HEIGHT), (x, y - (UPPER_TORSO_HEIGHT / 2)))
        self._add_part(lower_torso_body, lower_torso_shape, "lower_torso", vulnerable=True)
        # Upper torso
        upper_torso_body, upper_torso_shape = self._make_box((TORSO_WIDTH, UPPER_TORSO_HEIGHT), (x, y + (LOWER_TORSO_HEIGHT / 2)))
        self._add_part(upper_torso_body, upper_torso_shape, "upper_torso", vulnerable=True)
        # Legacy alias for observation & damage code
        self.parts['torso'] = upper_torso_body
        # Head
        head_mass = MASS_PER_PART
        head_moment = pymunk.moment_for_circle(head_mass, 0, HEAD_RADIUS)
        head_body = pymunk.Body(head_mass, head_moment)
        head_body.position = (x, y + LOWER_TORSO_HEIGHT / 2 + UPPER_TORSO_HEIGHT + HEAD_RADIUS)
        head_shape = pymunk.Circle(head_body, HEAD_RADIUS)
        self._add_part(head_body, head_shape, "head", vulnerable=True)

        # Arms
        l_upper_arm_body, l_upper_arm_shape = self._make_segment(UPPER_ARM_LENGTH, (x - TORSO_WIDTH / 2 - UPPER_ARM_LENGTH / 2, y + LOWER_TORSO_HEIGHT / 2))
        self._add_part(l_upper_arm_body, l_upper_arm_shape, "l_upper_arm")
        l_lower_arm_body, l_lower_arm_shape = self._make_segment(LOWER_ARM_LENGTH, (x - TORSO_WIDTH / 2 - UPPER_ARM_LENGTH - LOWER_ARM_LENGTH / 2, y + LOWER_TORSO_HEIGHT / 2))
        self._add_part(l_lower_arm_body, l_lower_arm_shape, "l_lower_arm", limb=True)

        r_upper_arm_body, r_upper_arm_shape = self._make_segment(UPPER_ARM_LENGTH, (x + TORSO_WIDTH / 2 + UPPER_ARM_LENGTH / 2, y + LOWER_TORSO_HEIGHT / 2))
        self._add_part(r_upper_arm_body, r_upper_arm_shape, "r_upper_arm")
        r_lower_arm_body, r_lower_arm_shape = self._make_segment(LOWER_ARM_LENGTH, (x + TORSO_WIDTH / 2 + UPPER_ARM_LENGTH + LOWER_ARM_LENGTH / 2, y + LOWER_TORSO_HEIGHT / 2))
        self._add_part(r_lower_arm_body, r_lower_arm_shape, "r_lower_arm", limb=True)

        # Legs
        l_upper_leg_body, l_upper_leg_shape = self._make_segment(UPPER_LEG_LENGTH, (x - TORSO_WIDTH / 4, y - LOWER_TORSO_HEIGHT / 2 - UPPER_LEG_LENGTH / 2), math.pi / 2)
        self._add_part(l_upper_leg_body, l_upper_leg_shape, "l_upper_leg")
        l_lower_leg_body, l_lower_leg_shape = self._make_segment(LOWER_LEG_LENGTH, (x - TORSO_WIDTH / 4, y - LOWER_TORSO_HEIGHT / 2 - UPPER_LEG_LENGTH - LOWER_LEG_LENGTH / 2), math.pi / 2)
        self._add_part(l_lower_leg_body, l_lower_leg_shape, "l_lower_leg", limb=True, foot=True)

        r_upper_leg_body, r_upper_leg_shape = self._make_segment(UPPER_LEG_LENGTH, (x + TORSO_WIDTH / 4, y - LOWER_TORSO_HEIGHT / 2 - UPPER_LEG_LENGTH / 2), math.pi / 2)
        self._add_part(r_upper_leg_body, r_upper_leg_shape, "r_upper_leg")
        r_lower_leg_body, r_lower_leg_shape = self._make_segment(LOWER_LEG_LENGTH, (x + TORSO_WIDTH / 4, y - LOWER_TORSO_HEIGHT / 2 - UPPER_LEG_LENGTH - LOWER_LEG_LENGTH / 2), math.pi / 2)
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
        pivot(head_body, upper_torso_body, (x, upper_torso_body.position.y + UPPER_TORSO_HEIGHT / 2))
        # Allow neck to swing both directions more freely
        neck_limit = limit(head_body, upper_torso_body, -1.2, 1.2)
        register_joint_pair(head_body, upper_torso_body)
        self.joint_angle_limits.append((neck_limit.min, neck_limit.max))
        # Shoulders
        pivot(l_upper_arm_body, upper_torso_body, (upper_torso_body.position.x - TORSO_WIDTH / 2, upper_torso_body.position.y))
        l_sh_limit = limit(l_upper_arm_body, upper_torso_body, -1.5, 1.5)
        register_joint_pair(l_upper_arm_body, upper_torso_body)
        self.joint_angle_limits.append((l_sh_limit.min, l_sh_limit.max))
        pivot(r_upper_arm_body, upper_torso_body, (upper_torso_body.position.x + TORSO_WIDTH / 2, upper_torso_body.position.y))
        r_sh_limit = limit(r_upper_arm_body, upper_torso_body, -1.5, 1.5)
        register_joint_pair(r_upper_arm_body, upper_torso_body)
        self.joint_angle_limits.append((r_sh_limit.min, r_sh_limit.max))
        # Elbows
        pivot(l_lower_arm_body, l_upper_arm_body, (l_upper_arm_body.position.x - UPPER_ARM_LENGTH / 2, l_upper_arm_body.position.y))
        # Elbows swing back and forth symmetrically
        l_el_limit = limit(l_lower_arm_body, l_upper_arm_body, -1.2, 1.2)
        register_joint_pair(l_lower_arm_body, l_upper_arm_body)
        self.joint_angle_limits.append((l_el_limit.min, l_el_limit.max))
        pivot(r_lower_arm_body, r_upper_arm_body, (r_upper_arm_body.position.x + UPPER_ARM_LENGTH / 2, r_upper_arm_body.position.y))
        r_el_limit = limit(r_lower_arm_body, r_upper_arm_body, -1.2, 1.2)
        register_joint_pair(r_lower_arm_body, r_upper_arm_body)
        self.joint_angle_limits.append((r_el_limit.min, r_el_limit.max))
        # Hips
        pivot(l_upper_leg_body, lower_torso_body, (x - TORSO_WIDTH / 4, lower_torso_body.position.y - LOWER_TORSO_HEIGHT / 2))
        # Hips swing both directions
        l_hip_limit = limit(l_upper_leg_body, lower_torso_body, -1.2, 1.2)
        register_joint_pair(l_upper_leg_body, lower_torso_body)
        self.joint_angle_limits.append((l_hip_limit.min, l_hip_limit.max))
        pivot(r_upper_leg_body, lower_torso_body, (x + TORSO_WIDTH / 4, lower_torso_body.position.y - LOWER_TORSO_HEIGHT / 2))
        r_hip_limit = limit(r_upper_leg_body, lower_torso_body, -1.2, 1.2)
        register_joint_pair(r_upper_leg_body, lower_torso_body)
        self.joint_angle_limits.append((r_hip_limit.min, r_hip_limit.max))
        # Knees
        pivot(l_lower_leg_body, l_upper_leg_body, (l_upper_leg_body.position.x, l_upper_leg_body.position.y - UPPER_LEG_LENGTH / 2))
        # Knees swing both directions symmetrically
        l_knee_limit = limit(l_lower_leg_body, l_upper_leg_body, -1.4, 1.4)
        register_joint_pair(l_lower_leg_body, l_upper_leg_body)
        self.joint_angle_limits.append((l_knee_limit.min, l_knee_limit.max))
        pivot(r_lower_leg_body, r_upper_leg_body, (r_upper_leg_body.position.x, r_upper_leg_body.position.y - UPPER_LEG_LENGTH / 2))
        r_knee_limit = limit(r_lower_leg_body, r_upper_leg_body, -1.4, 1.4)
        register_joint_pair(r_lower_leg_body, r_upper_leg_body)
        self.joint_angle_limits.append((r_knee_limit.min, r_knee_limit.max))

        # Register spine as a joint pair for state (place after limbs to keep mapping intuitive)
        self.joint_pairs.append((upper_torso_body, lower_torso_body))
        # Append spine angle limits as last action index
        self.joint_angle_limits.append((spine_limit.min, spine_limit.max))

        # Create motors aligned with joint_pairs/action indices
        torque_by_index = [
            NECK_TORQUE,          # 0 neck
            SHOULDER_TORQUE,      # 1 L shoulder
            SHOULDER_TORQUE,      # 2 R shoulder
            ELBOW_TORQUE,         # 3 L elbow
            ELBOW_TORQUE,         # 4 R elbow
            HIP_TORQUE,           # 5 L hip
            HIP_TORQUE,           # 6 R hip
            KNEE_TORQUE,          # 7 L knee
            KNEE_TORQUE,          # 8 R knee
            SPINE_TORQUE,         # 9 spine
        ]
        for i, (a, b) in enumerate(self.joint_pairs):
            m = pymunk.SimpleMotor(a, b, 0.0)
            m.max_force = torque_by_index[i]
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

        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2 * math.pi) - math.pi

        for i, (a_body, b_body) in enumerate(self.joint_pairs):
            # Desired target within physical joint limits
            min_ang, max_ang = self.joint_angle_limits[i]
            mid = 0.5 * (min_ang + max_ang)
            half = 0.5 * (max_ang - min_ang)
            target = float(action[i])
            target = max(-1.0, min(1.0, target))
            target_angle = mid + half * target
            # Current relative angle (a - b) wrapped
            curr_rel = wrap_pi(a_body.angle - b_body.angle)
            # Clamp desired to limits (safety)
            if target_angle < min_ang:
                target_angle = min_ang
            elif target_angle > max_ang:
                target_angle = max_ang
            # Proportional control -> target relative angular velocity
            error = wrap_pi(target_angle - curr_rel)
            rate = BASE_P_GAIN * error
            # Clamp to avoid instability
            if rate > MAX_MOTOR_RATE:
                rate = MAX_MOTOR_RATE
            elif rate < -MAX_MOTOR_RATE:
                rate = -MAX_MOTOR_RATE
            self.motors[i].rate = rate

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

    def mark_ground_contact(self, entering: bool):
        if entering:
            self.ground_contacts += 1
        else:
            self.ground_contacts = max(0, self.ground_contacts - 1)

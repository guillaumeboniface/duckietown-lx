from typing import List, Tuple
from dt_protocols import PlanStep
from aido_schemas import FriendlyPose
import numpy as np
from dt_protocols import PlacedPrimitive, Circle, Rectangle
import itertools


def plan_to_destination_dt(plan_step: PlanStep, start: Tuple[int, int, int], dt: float=0.5, t_start: float=0.0) -> List[Tuple[int, int, int]]:
    destinations = []
    t_start = t_start % dt - dt if t_start > 0 else 0.0 # We only care about the offset to dt
    for t in np.arange(t_start, plan_step.duration, dt):
        segment_duration = min(t + dt, plan_step.duration)
        rotation = plan_step.angular_velocity_deg_s * segment_duration
        if abs(rotation) < 1e-12:
            chord = plan_step.velocity_x_m_s * segment_duration
            theta = np.deg2rad(start[2])
        else:
            theta = np.deg2rad(start[2] + rotation / 2)
            r = plan_step.velocity_x_m_s * segment_duration / np.deg2rad(rotation)
            chord = 2 * r * np.sin(np.deg2rad(rotation / 2))
        x = start[0] + chord * np.cos(theta) * 100
        y = start[1] + chord * np.sin(theta) * 100
        heading = (start[2] + rotation) % 360
        destinations.append((round(x), round(y), round(heading)))
    return destinations


def float_to_node_index(t: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (round(t[0] * 100), round(t[1] * 100), round(t[2]))


def pose_to_node(pose: FriendlyPose) -> Tuple[float]:
    return float_to_node_index((
        pose.x,
        pose.y,
        pose.theta_deg
    ))

def node_to_pose(node: Tuple[int, int, int]) -> FriendlyPose:
    return FriendlyPose(x=node[0] / 100, y=node[1] / 100, theta_deg=node[2])

def rectangle_center(r: PlacedPrimitive) -> FriendlyPose:
    pose = r.pose
    rect = r.primitive
    return FriendlyPose(pose.x + rect.xmin + (rect.xmax - rect.xmin) / 2, pose.y + rect.ymin + (rect.ymax - rect.ymin) / 2, pose.theta_deg)

def distance_pose(a: FriendlyPose, b: FriendlyPose) -> float:
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))

def rect_diagonal(r: Rectangle) -> float:
    return np.linalg.norm(np.array([r.xmax - r.xmin, r.ymax - r.ymin]))

def check_encompass_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    a_radius = a.primitive.radius if isinstance(a.primitive, Circle) else rect_diagonal(a.primitive)
    b_radius = b.primitive.radius if isinstance(b.primitive, Circle) else rect_diagonal(b.primitive)
    a_center = rectangle_center(a) if isinstance(a.primitive, Rectangle) else a.pose
    b_center = rectangle_center(b) if isinstance(b.primitive, Rectangle) else b.pose
    if distance_pose(a_center, b_center) + b_radius <= a_radius:
        return True
    return False

def clean_environment(environment: List[PlacedPrimitive]) -> List[PlacedPrimitive]:
    env_w_motion = filter(lambda x: x.motion is not None, environment)
    env_wo_motion = list(filter(lambda x: x.motion is None, environment))
    to_remove = set()
    for a, b in itertools.product(env_wo_motion, env_wo_motion):
        if a == b or b in to_remove:
            continue
        if check_encompass_shape(a, b):
            to_remove.add(a)
    return [e for e in env_wo_motion if e not in to_remove] + list(env_w_motion)
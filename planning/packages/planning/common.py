from typing import List, Tuple
from dt_protocols import PlanStep
from aido_schemas import FriendlyPose
import numpy as np


def plan_to_destination_dt(plan_step: PlanStep, start: Tuple[int, int, int], dt: float=0.5, t_start: float=0.0) -> List[Tuple[int, int, int]]:
    destinations = []
    if t_start >= plan_step.duration:
        t_start = 0.0
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
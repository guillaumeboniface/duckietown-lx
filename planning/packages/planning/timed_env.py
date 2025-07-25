from typing import Dict, List, Tuple
from dt_protocols import PlacedPrimitive, PlanStep
from aido_schemas import Context, FriendlyPose
from planning.collision_check import clean_environment
from planning.common import plan_to_destination_dt, pose_to_node, node_to_pose
import numpy as np
import itertools
import math

def plan_duration(plan: List[PlanStep]) -> float:
    return sum(map(lambda x: x.duration, plan))

def generate_trajectory(p: PlacedPrimitive, dt: float) -> List[Tuple[float, PlacedPrimitive]]:
    trajectory: List[PlacedPrimitive] = [PlacedPrimitive(p.pose,p.primitive)]
    start: FriendlyPose = p.pose
    t = 0
    for step in p.motion.steps:
        destinations = plan_to_destination_dt(step, pose_to_node(start), dt, t_start=t)
        start = node_to_pose(destinations[-1])
        t = step.duration % dt
        if step.duration < dt: continue
        if step.duration % dt != 0:
            sampled_destinations = destinations[:-1]
        else:
            sampled_destinations = destinations
        for destination in sampled_destinations:
            trajectory.append(PlacedPrimitive(node_to_pose(destination),p.primitive))
    if step.duration % dt != 0: # Make sure the last pose is added to the trajectory
        trajectory.append(PlacedPrimitive(node_to_pose(destinations[-1]),p.primitive))
    for i in range(len(trajectory)):
        trajectory[i] = (i * dt, trajectory[i])
    return trajectory


def expand_trajectory(trajectory: List[Tuple[float, PlacedPrimitive]], dt: float, max_t: float, periodic: bool) -> List[Tuple[float, PlacedPrimitive]]:
    n_samples = int(max_t / dt) + 1
    new_trajectory: List[Tuple[float, PlacedPrimitive]] = []
    if periodic:
        # Create an infinite iterator of the trajectory that traverses it forward then backward
        iterator = itertools.cycle(trajectory + trajectory[1:-1][::-1])
        for i in range(n_samples):
            new_trajectory.append((i * dt, next(iterator)[1]))
    else:
        # Keep the last pose for the rest of the trajectory
        for i in range(len(trajectory), n_samples):
            trajectory.append((i * dt, trajectory[-1][1]))
        new_trajectory = trajectory
    return new_trajectory


class TimedEnv:
    def __init__(self, environment: List[PlacedPrimitive], dt: float):
        self.original_environment: List[PlacedPrimitive] = environment
        self.dt: float = dt
        self.max_loop: float = math.ceil(self._find_max_t() / dt) * dt
        self.periodic: bool = self._is_periodic()
        self.env_trajectory: Dict[float, List[PlacedPrimitive]] = self._build_env_trajectory()

    def _find_max_t(self) -> float:
        max_t = 0
        for p in self.original_environment:
            if p.motion is not None:
                max_t = max(max_t, plan_duration(p.motion.steps))
        return max_t
    
    def _is_periodic(self) -> bool:
        for p in self.original_environment:
            if p.motion is not None:
                if p.motion.periodic:
                    return True
        return False
    
    def _build_env_trajectory(self) -> Dict[float, List[PlacedPrimitive]]:
        env_trajectory: Dict[float, List[PlacedPrimitive]] = {}
        for t in np.arange(0, self.max_loop + self.dt, self.dt):
            env_trajectory[t] = [p for p in self.original_environment if p.motion is None]
        for p in filter(lambda x: x.motion is not None, self.original_environment):
            trajectory = generate_trajectory(p, self.dt)
            if abs(plan_duration(p.motion.steps) - self.max_loop) > 1e-6:
                trajectory = expand_trajectory(trajectory, self.dt, self.max_loop, p.motion.periodic)
            for t, object in trajectory:
                if t in env_trajectory:
                    env_trajectory[t].append(object)
        for k, v in env_trajectory.items():
            env_trajectory[k] = clean_environment(v)
        return env_trajectory

    def get_env(self, t: float) -> List[PlacedPrimitive]:
        if self.max_loop == 0:
            return self.original_environment
        t = t % self.max_loop
        t = round(t / self.dt) * self.dt
        return self.env_trajectory[t]
from typing import List
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import math
from queue import PriorityQueue
from planning.collision_check import check_point_collision, clean_environment
from planning.timed_env import TimedEnv
from planning.common import plan_to_destination_dt, pose_to_node, node_to_pose, float_to_node_index
from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    PlacedPrimitive,
    PlanningQuery,
    PlanningResult,
    PlanningSetup,
    PlanStep,
    Circle,
    Rectangle,
    SimulationResult,
    simulate,
)


__all__ = ["Planner"]

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

def build_pose_matrix(pose: FriendlyPose) -> np.array:
    theta = np.deg2rad(pose.theta_deg)
    return [
        [np.cos(theta), -np.sin(theta), pose.x],
        [np.sin(theta), np.cos(theta), pose.y],
        [0, 0, 1]
    ]

def pose_in_rectangle(pose: FriendlyPose, r: Rectangle) -> bool:
    if not r.xmin <= pose.x <= r.xmax:
        return False
    if not r.ymin <= pose.y <= r.ymax:
        return False
    return True

def find_heading_and_distance(start: FriendlyPose, goal: FriendlyPose) -> List[float]:
    translation = np.array([goal.x - start.x, goal.y - start.y])
    distance = np.linalg.norm(translation)
    heading = np.arctan2(translation[1], translation[0])
    return np.rad2deg(heading), distance

def create_turn(ps: PlanningSetup, deg: float) -> PlanStep:
    if abs(deg) > 180 and deg > 0:
        deg = deg - 360
    elif abs(deg) > 180 and deg < 0:
        deg = deg + 360
    sign = 1 if deg >= 0 else -1
    duration = deg / ps.max_angular_velocity_deg_s * sign
    return PlanStep(
        duration=duration,
        velocity_x_m_s=0.0,
        angular_velocity_deg_s=ps.max_angular_velocity_deg_s * sign)

def create_straight(ps: PlanningSetup, distance: float) -> PlanStep:
    return PlanStep(
        duration=distance / ps.max_linear_velocity_m_s,
        velocity_x_m_s=ps.max_linear_velocity_m_s,
        angular_velocity_deg_s=0.0
    )

def connect_poses(ps: PlanningSetup, start: FriendlyPose, goal: FriendlyPose) -> List[PlanStep]:
    plan: List[PlanStep] = []
    if start.x == goal.x and start.y == goal.y:
        plan.append(create_turn(ps, goal.theta_deg - start.theta_deg))
    else:
        heading, distance = find_heading_and_distance(start, goal)
        plan.append(create_turn(ps, heading - start.theta_deg))
        plan.append(create_straight(ps, distance))
        plan.append(create_turn(ps, goal.theta_deg - (heading - start.theta_deg)))
    return plan

def connect_poses_with_curvature(ps: PlanningSetup, start: FriendlyPose, goal: FriendlyPose) -> Tuple[PlanStep, float]:
    heading, chord = find_heading_and_distance(start, goal)

    if 90 < abs(heading - start.theta_deg) < 270: # Going backward
        start_theta_deg = start.theta_deg - 180
        max_velocity = ps.min_linear_velocity_m_s
        backward = True
    else: # Going forward
        max_velocity = ps.max_linear_velocity_m_s
        backward = False
        start_theta_deg = start.theta_deg
    
    theta = (heading - start_theta_deg) * 2
    theta = theta if abs(theta) < 360 else theta - 720 if theta > 0 else theta + 720
    if abs(theta) < 1e-12: # Straight line
        return PlanStep(
            duration=abs(chord / max_velocity),
            velocity_x_m_s=max_velocity,
            angular_velocity_deg_s=0.0
        ), start.theta_deg
    radius = abs(chord * np.sin((np.pi - np.deg2rad(theta)) / 2) / np.sin(np.deg2rad(theta))) if abs(np.sin(np.deg2rad(theta))) > 1e-12 else chord / 2
    arc = abs(radius * np.deg2rad(theta))
    duration = abs(arc / max_velocity)
    if abs(theta / duration) > ps.max_angular_velocity_deg_s:
        duration = abs(theta / ps.max_angular_velocity_deg_s)
        velocity_x_m_s = arc / duration if not backward else -arc / duration
    else:
        duration = abs(arc / max_velocity)
        velocity_x_m_s = max_velocity

    if duration < 1e-12:
        print(f"start: {start}, goal: {goal}")
        print(f"duration: {duration}, theta: {theta}, radius: {radius}, arc: {arc}, velocity_x_m_s: {velocity_x_m_s}")
        assert False, "Duration is too small"

    plan = PlanStep(
        duration=duration,
        velocity_x_m_s=velocity_x_m_s,
        angular_velocity_deg_s=theta / duration
    )

    return plan, (start.theta_deg + theta) % 360

def circle_bounds(circle: Circle, pose: FriendlyPose) -> Rectangle:
    return Rectangle(xmin=pose.x - circle.radius, ymin=pose.y - circle.radius, xmax=pose.x + circle.radius, ymax=pose.y + circle.radius)

def rectangle_bounds(rectangle: Rectangle, pose: FriendlyPose) -> Rectangle:
    pose_matrix = np.linalg.inv(build_pose_matrix(pose))
    points = np.array([
        [rectangle.xmin, rectangle.ymin, 1],
        [rectangle.xmin, rectangle.ymax, 1],
        [rectangle.xmax, rectangle.ymax, 1],
        [rectangle.xmax, rectangle.ymin, 1]
    ])
    transformed_points = np.dot(pose_matrix, points.T).T
    return Rectangle(xmin=transformed_points[:, 0].min(), ymin=transformed_points[:, 1].min(), xmax=transformed_points[:, 0].max(), ymax=transformed_points[:, 1].max())

def closest_points(primitive_bounds: Rectangle, bounds: Rectangle, tolerance: float) -> Tuple[float]:
    xmin =  max(math.floor(primitive_bounds.xmin / tolerance) * tolerance, bounds.xmin)
    ymin =  max(math.floor(primitive_bounds.ymin / tolerance) * tolerance, bounds.ymin)
    xmax =  min(math.ceil(primitive_bounds.xmax / tolerance) * tolerance, bounds.xmax)
    ymax =  min(math.ceil(primitive_bounds.ymax / tolerance) * tolerance, bounds.ymax)
    return xmin, ymin, xmax, ymax


def compute_edge_kernel(ps: PlanningSetup) -> Dict[int, List[Tuple[float, float, PlanStep, float]]]:
    kernel = {}
    mid = (ps.bounds.xmax + ps.bounds.xmin) / 2
    window = (ps.bounds.xmax - ps.bounds.xmin) / 6
    for k in np.arange(0, 360, ps.tolerance_theta_deg):
        start = FriendlyPose(x=mid, y=mid, theta_deg=k)
        kernel[k] = []
        for i in np.arange(-window, window, ps.tolerance_xy_m):
            for j in np.arange(-window, window, ps.tolerance_xy_m):
                if abs(i) < 1e-12 and abs(j) < 1e-12:
                    continue
                goal = FriendlyPose(x=mid + i, y=mid + j, theta_deg=0.0)
                plan, new_heading = connect_poses_with_curvature(ps, start, goal)
                if plan is not None and round(new_heading) % 20 == 0:
                    kernel[k].append((i, j, plan, round(new_heading) % 360))
    return kernel

def driving_options(ps: PlanningSetup) -> Tuple[PlanStep, ...]:
    OPTIONS = []

    for m in (10, 5, 2):
        d = ps.tolerance_xy_m * m
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(d, 0, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(0, d, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(-d, 0, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(0, -d, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(d, d, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(-d, d, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(d, -d, 0))[0])
        OPTIONS.append(connect_poses_with_curvature(ps, FriendlyPose(0, 0, 0), FriendlyPose(-d, -d, 0))[0])
    
    return OPTIONS


def plan_to_destination(plan_step: PlanStep, start: Tuple[int, int, int], n_samples: int=1) -> List[Tuple[int, int, int]]:
    destinations = []
    for i in range(1, n_samples + 1):
        segment_duration = plan_step.duration / n_samples * i
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


def options_to_destinations(
        ps: PlanningSetup,
        options: Tuple[PlanStep, ...],
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        t: float) -> List[Tuple[PlanStep, Tuple[int, int, int]]]:
    destinations = []
    dt = 0.3
    for plan_step in options:
        path = plan_to_destination_dt(plan_step, start, dt, t)
        if nodes_in_bounds(path, ps.bounds) and not check_point_collision(ps, path):
            destinations.append((plan_step, path[-1]))

    plan_step, new_heading = connect_poses_with_curvature(
        ps,
        FriendlyPose(start[0] / 100, start[1] / 100, start[2]),
        FriendlyPose(goal[0] / 100, goal[1] / 100, goal[2])
    )
    path = plan_to_destination_dt(plan_step, start, dt, t)
    if nodes_in_bounds(path, ps.bounds) and not check_point_collision(ps, path):
        destinations.append((plan_step, (goal[0], goal[1], new_heading)))

    if ps.max_curvature == math.inf:
        destinations.append((create_turn(ps, goal[2] - start[2]), (start[0], start[1], goal[2])))
        heading, _ = find_heading_and_distance(node_to_pose(start), node_to_pose(goal))
        destinations.append((create_turn(ps, heading - start[2]), (start[0], start[1], heading)))

    return destinations


def create_graph(ps: PlanningSetup, bounds: Rectangle, environment: List[PlacedPrimitive]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for i in np.arange(bounds.xmin, bounds.xmax, ps.tolerance_xy_m):
        for j in np.arange(bounds.ymin, bounds.ymax, ps.tolerance_xy_m):
            for k in np.arange(0, 360, ps.tolerance_theta_deg):
                graph.add_node(float_to_node_index((i, j, k)), x=i, y=j, theta_deg=k)

    for placed_primitive in environment:
            if isinstance(placed_primitive.primitive, Circle):
                primitive_bounds: Rectangle = circle_bounds(placed_primitive.primitive, placed_primitive.pose)
            else:
                # This is a conservative approximation when the rectangle is not aligned with the axes
                primitive_bounds: Rectangle = rectangle_bounds(placed_primitive.primitive, placed_primitive.pose)

            xmin, ymin, xmax, ymax = closest_points(primitive_bounds, bounds, ps.tolerance_xy_m)

            for i in np.arange(xmin, xmax, ps.tolerance_xy_m):
                for j in np.arange(ymin, ymax, ps.tolerance_xy_m):
                    for k in np.arange(0, 360, ps.tolerance_theta_deg):
                        if float_to_node_index((i, j, k)) in graph.nodes():
                            graph.remove_node(float_to_node_index((i, j, k)))

    return graph

def add_edges(graph: nx.MultiDiGraph, ps: PlanningSetup):
    kernel = compute_edge_kernel(ps)

    for node in graph.nodes():
        x, y, theta_deg = node
        x = x / 100
        y = y / 100

        for i, j, plan, new_theta_deg in kernel[theta_deg]:
            if float_to_node_index((x + i, y + j, new_theta_deg)) in graph.nodes():
                graph.add_edge(node, float_to_node_index((x + i, y + j, new_theta_deg)), plan=plan)

        for k in np.arange(0, 360, ps.tolerance_theta_deg):
            if abs(k - theta_deg) < 1e-12:
                continue
            if float_to_node_index((x, y, k)) in graph.nodes():
                plan = create_turn(ps, k - theta_deg)
                graph.add_edge(node, float_to_node_index((x, y, k)), plan=plan)

def plan_from_path(graph: nx.MultiDiGraph, path: List[Tuple[int, int, int]]) -> List[PlanStep]:
    edges = list(zip(path, path[1:]))
    plan: List[PlanStep] = []
    for edge in edges:
        plan.append(graph.get_edge_data(edge[0], edge[1])[0]['plan'])
    return plan

def node_distance(start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
    return np.linalg.norm(np.array(start[:2]) - np.array(goal[:2]))

def node_in_bounds(node: Tuple[int, int, int], bounds: Rectangle) -> bool:
    return round(bounds.xmin * 100) <= node[0] <= round(bounds.xmax * 100) and round(bounds.ymin * 100) <= node[1] <= round(bounds.ymax * 100)

def nodes_in_bounds(nodes: List[Tuple[int, int, int]], bounds: Rectangle) -> bool:
    for node in nodes:
        if not node_in_bounds(node, bounds):
            return False
    return True

def heuristic(ps: PlanningSetup, start: Tuple[int, int, int], goal: Tuple[int, int, int], duration: float) -> float:
    step, new_heading = connect_poses_with_curvature(
        ps,
        node_to_pose(start),
        node_to_pose(goal))
    step2_duration = sum(map(lambda x: x.duration, connect_poses(ps, node_to_pose(start), node_to_pose(goal))))
    return min(step.duration, step2_duration)

def a_star(ps: PlanningSetup, start: FriendlyPose, goal: FriendlyPose, timed_env: TimedEnv) -> Tuple[Optional[List[PlanStep]], Optional[List[FriendlyPose]]]:
    queue = PriorityQueue()
    queue.put(PrioritizedItem(0, [pose_to_node(start)]))
    visited = set([pose_to_node(start)])
    options = driving_options(ps)
    goal_node = pose_to_node(goal)

    while not queue.empty():
            p_item = queue.get()
            current_cost, path = p_item.priority, p_item.item
            if len(visited) > 64000:
                break
            current = path[-1]

            x, y, theta_deg = current
            if node_distance(current, goal_node) < ps.tolerance_xy_m and abs(theta_deg - goal_node[2]) < ps.tolerance_theta_deg:
                print(f"A star finished, {len(visited)} nodes visited")
                return path[1::2], path[0::2]
            
            t = sum(map(lambda x: x.duration, path[1::2]))
            env = timed_env.get_env(t)
            ps.environment = env
            destinations = options_to_destinations(ps, options, current, goal_node, t % timed_env.dt)
            
            for plan, destination in destinations:
                if destination in visited or not node_in_bounds(destination, ps.bounds):
                    continue
                visited.add(destination)
                cost = current_cost + heuristic(ps, destination, goal_node, plan.duration)
                queue.put(PrioritizedItem(cost, path + [plan, destination]))

    return None, None

class Planner:
    params: PlanningSetup

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: PlanningSetup):
        context.info("initialized")
        self.params = data

        print(f"CHALLENGE PARAMETERS:\n{self.params}")

        # This is the interval of allowed linear velocity
        # Note that min_velocity_x_m_s and max_velocity_x_m_s might be different.
        # Note that min_velocity_x_m_s may be 0 in advanced exercises (cannot go backward)
        max_velocity_x_m_s: float = self.params.max_linear_velocity_m_s
        min_velocity_x_m_s: float = self.params.min_linear_velocity_m_s

        # This is the max curvature. In earlier exercises, this is +inf: you can turn in place.
        # In advanced exercises, this is less than infinity: you cannot turn in place.
        max_curvature: float = self.params.max_curvature

        # these have the same meaning as the collision exercises
        body: List[PlacedPrimitive] = self.params.body
        environment: List[PlacedPrimitive] = self.params.environment

        # these are the final tolerances - the precision at which you need to arrive at the goal
        tolerance_theta_deg: float = self.params.tolerance_theta_deg
        tolerance_xy_m: float = self.params.tolerance_xy_m

        # For convenience, this is the rectangle that contains all the available environment,
        # so you don't need to compute it
        bounds: Rectangle = self.params.bounds

        self.params.environment = clean_environment(environment)

        self.timed_env = TimedEnv(environment, 0.2)

        # self.graph = create_graph(self.params, bounds, environment)

        # add_edges(self.graph, self.params)


    def on_received_query(self, context: Context, data: PlanningQuery):
        # A planning query is a pair of initial and goal poses
        print(f"PLANNING QUERY: {data}")

        start: FriendlyPose = data.start
        goal: FriendlyPose = data.target

        feasible = True

        if not pose_in_rectangle(start, self.params.bounds):
            feasible = False
        if not pose_in_rectangle(goal, self.params.bounds):
            feasible = False

        plan, _ = a_star(self.params, start, goal, self.timed_env)

        if plan is None:
            feasible = False

        if not feasible:
            result: PlanningResult = PlanningResult(False, [])
            context.write("response", result)
            return

        result: PlanningResult = PlanningResult(feasible, plan)
        context.write("response", result)

from typing import List
import numpy as np
import networkx as nx
from typing import Tuple, Dict
import math
from functools import lru_cache

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
    if abs(heading - start.theta_deg) < 1e-12:
        return PlanStep(
            duration=chord / ps.max_linear_velocity_m_s,
            velocity_x_m_s=ps.max_linear_velocity_m_s,
            angular_velocity_deg_s=0.0
        ), start.theta_deg
    theta = (heading - start.theta_deg) * 2
    theta = theta if abs(theta) < 360 else theta - 720 if theta > 0 else theta + 720
    if abs(theta) < 1e-12: # Trying to move backward, not supported yet
        return None, None
    radius = max(abs(chord * np.sin((np.pi - np.deg2rad(theta)) / 2) / np.sin(np.deg2rad(theta))), chord / 2)
    arc = abs(radius * np.deg2rad(theta))
    duration = arc / ps.max_linear_velocity_m_s
    if theta / duration > ps.max_angular_velocity_deg_s:
        duration = theta / ps.max_angular_velocity_deg_s
        velocity_x_m_s = arc / duration
    else:
        duration = arc / ps.max_linear_velocity_m_s
        velocity_x_m_s = ps.max_linear_velocity_m_s

    if duration < 1e-12:
        print(f"start: {start}, goal: {goal}")
        print(f"duration: {duration}, theta: {theta}, radius: {radius}, arc: {arc}, velocity_x_m_s: {velocity_x_m_s}")
        assert False

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

def pose_to_node(ps: PlanningSetup, pose: FriendlyPose) -> Tuple[float]:
    return float_to_node_index((
        math.ceil(pose.x / ps.tolerance_xy_m) * ps.tolerance_xy_m,
        math.ceil(pose.y / ps.tolerance_xy_m) * ps.tolerance_xy_m,
        math.ceil(pose.theta_deg / ps.tolerance_theta_deg) * ps.tolerance_theta_deg
    ))

def float_to_node_index(t: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (round(t[0] * 100), round(t[1] * 100), round(t[2]))

@lru_cache(maxsize=1)
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
                    kernel[k].append((i, j, plan, new_heading))
    return kernel

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

        self.graph = create_graph(self.params, bounds, environment)

        add_edges(self.graph, self.params)


    def on_received_query(self, context: Context, data: PlanningQuery):
        # A planning query is a pair of initial and goal poses
        start: FriendlyPose = data.start
        goal: FriendlyPose = data.target

        feasible = True

        if not pose_in_rectangle(start, self.params.bounds):
            feasible = False
        if not pose_in_rectangle(goal, self.params.bounds):
            feasible = False

        try:
            path = nx.shortest_path(
                self.graph,
                pose_to_node(self.params, start),
                pose_to_node(self.params, goal),
                weight=lambda _, __, x: x[0]['plan'].duration
            )
        except nx.NetworkXNoPath:
            feasible = False

        if not feasible:
            result: PlanningResult = PlanningResult(False, None)
            context.write("response", result)
            return
        
        plan = plan_from_path(self.graph, path)

        result: PlanningResult = PlanningResult(feasible, plan)
        context.write("response", result)

import itertools
import random
from typing import List, Tuple
import numpy as np
from planning.timed_env import TimedEnv
from planning.common import distance_pose, rect_diagonal
from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
    PlanningSetup
)

__all__ = ["CollisionChecker"]

def build_pose_matrix(pose: FriendlyPose) -> np.array:
    theta = np.deg2rad(pose.theta_deg)
    return [
        [np.cos(theta), -np.sin(theta), pose.x],
        [np.sin(theta), np.cos(theta), pose.y],
        [0, 0, 1]
    ]

def transform(p: PlacedPrimitive, robot_pose: FriendlyPose):
    if p.pose.x == 0 and p.pose.y == 0 and p.pose.theta_deg == 0:
        p.pose = robot_pose
        return p
    part_pose_matrix = build_pose_matrix(p.pose)
    robot_ref_pose = build_pose_matrix(robot_pose)
    part_in_robot_ref = np.dot(robot_ref_pose, part_pose_matrix)
    new_angle = np.rad2deg(np.arctan2(part_in_robot_ref[1, 0], part_in_robot_ref[0, 0]))
    new_pose = FriendlyPose(part_in_robot_ref[0, 2], part_in_robot_ref[1, 2], new_angle)
    p.pose = new_pose
    return p




def localise_rectangle(r: Rectangle, b_pose: FriendlyPose, a_pose: FriendlyPose) -> List[np.array]:
    b_pose_matrix = build_pose_matrix(b_pose)
    a_pose_matrix = np.linalg.inv(build_pose_matrix(a_pose))
    localisation_matrix = np.dot(a_pose_matrix, b_pose_matrix)
    point1 = np.dot(localisation_matrix, np.array([r.xmin, r.ymin, 1]))
    point2 = np.dot(localisation_matrix, np.array([r.xmin, r.ymax, 1]))
    point3 = np.dot(localisation_matrix, np.array([r.xmax, r.ymax, 1]))
    point4 = np.dot(localisation_matrix, np.array([r.xmax, r.ymin, 1]))
    return [point1[:2], point2[:2], point3[:2], point4[:2]]

def localise_circle(b_pose: FriendlyPose, a_pose: FriendlyPose) -> FriendlyPose:
    a_pose_matrix = np.linalg.inv(build_pose_matrix(a_pose))
    b_in_a = np.dot(a_pose_matrix, np.array([b_pose.x, b_pose.y, 1]))
    return FriendlyPose(b_in_a[0], b_in_a[1], 0)

def check_point_in_rectange(x: float, y: float, r: Rectangle) -> bool:
    if r.xmin <= x <= r.xmax and r.ymin <= y <= r.ymax:
        return True
    return False

def check_circle_rectangle_overlap(x: float, y: float, radius: float, r: Rectangle) -> bool:
    def axis_distance(low: float, high: float, coordinate: float) -> float:
        if low <= coordinate <= high:
            return 0
        elif coordinate < low:
            return low - coordinate
        else:
            return coordinate - high

    dx: float = axis_distance(r.xmin, r.xmax, x)
    dy: float = axis_distance(r.ymin, r.ymax, y)
    return dx ** 2 + dy ** 2 <= radius ** 2

def rect_small_radius(r: Rectangle) -> float:
    return min(r.xmax - r.xmin, r.ymax - r.ymin) / 2

class CollisionChecker:
    params: MapDefinition

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data

    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        collided = check_collision(
            environment=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)


def check_collision(
    environment: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:

    rototranslated_robot: List[PlacedPrimitive] = [transform(PlacedPrimitive(FriendlyPose(p.pose.x, p.pose.y, p.pose.theta_deg), p.primitive), robot_pose) for p in robot_body]

    # Then, call check_collision_list to see if the robot collides with the environment
    collided = check_collision_list(rototranslated_robot, environment)

    return collided


def check_point_collision(ps: PlanningSetup, timed_env: TimedEnv, t: float, points: List[Tuple[int, int, int]]) -> bool:
    for i, point in enumerate(points):
        if check_collision(timed_env.get_env(t + i * timed_env.dt), ps.body, FriendlyPose(point[0] / 100, point[1] / 100, point[2])):
            return True
    return False


def check_collision_list(
    rototranslated_robot: List[PlacedPrimitive], environment: List[PlacedPrimitive]
) -> bool:
    for robot, envObject in itertools.product(rototranslated_robot, environment):
        if check_collision_shape(robot, envObject):
            return True

    return False

def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly

    if isinstance(a.primitive, Circle) and isinstance(b.primitive, Circle):
        if np.linalg.norm(np.array[a.pose.x - b.pose.x, a.pose.y - b.pose.y]) <= a.primitive.radius + b.primitive.radius:
            return True
    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Circle):
        # Convert the coordinate of the circle in the base of the rectangle
        if distance_pose(a.pose, b.pose) > b.primitive.radius + rect_diagonal(a.primitive):
            return False
        if distance_pose(a.pose, b.pose) < rect_small_radius(a.primitive) + b.primitive.radius:
            return True
        b_coord_in_a = localise_circle(b.pose, a.pose)
        if check_circle_rectangle_overlap(b_coord_in_a.x, b_coord_in_a.y, b.primitive.radius, a.primitive):
            return True
    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Rectangle):
        if distance_pose(a.pose, b.pose) > rect_diagonal(a.primitive) + rect_diagonal(b.primitive):
            return False
        if distance_pose(a.pose, b.pose) < rect_small_radius(a.primitive) + rect_small_radius(b.primitive):
            return True
        # Convert the coordinate of one rectangle in the base of the other
        b_coord_in_a = localise_rectangle(b.primitive, b.pose, a.pose)
        # Check whether one of the corner falls into the rectangle
        for x, y in b_coord_in_a:
            if check_point_in_rectange(x, y, a.primitive):
                return True 
        # Do the same in the other direction
        a_coord_in_b = localise_rectangle(a.primitive, a.pose, b.pose)
        for x, y in a_coord_in_b:
            if check_point_in_rectange(x, y, b.primitive):
                return True

    return False

from planning.planner import (
    find_heading_and_distance,
    create_turn,
    create_graph,
    rectangle_bounds,
    add_edges,
    pose_to_node,
    closest_points
)
from aido_schemas import Context, FriendlyPose
from dt_protocols import PlanningSetup, Rectangle, Circle, PlacedPrimitive
import unittest
import math

class TestOne(unittest.TestCase):
    def test_heading_and_distance(self):
        start = FriendlyPose(1.5, 1.5, 0.0)
        goal = FriendlyPose(1.0, 1.0, 45)
        heading, distance = find_heading_and_distance(start, goal)
        assert(abs(heading + 135) < 1e-12)
        assert(abs(distance - 0.7071067811865476) < 1e-12)

    def test_heading_and_distance2(self):
        start = FriendlyPose(1.5, 1.5, 0.0)
        goal = FriendlyPose(1.5, 1.5, 0.0)
        heading, distance = find_heading_and_distance(start, goal)
        assert(abs(heading - 0) < 1e-12)
        assert(abs(distance - 0) < 1e-12)

    def test_heading_and_distance3(self):
        start = FriendlyPose(0.0, 0.0, 0.0)
        goal = FriendlyPose(-1.5, 1.5, 0)
        heading, distance = find_heading_and_distance(start, goal)
        assert(abs(heading - 135) < 1e-12)
        assert(abs(distance - 2.1213203435596424) < 1e-12)

    def test_create_graph(self):
        ps = PlanningSetup(
            bounds=Rectangle(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0),
            max_linear_velocity_m_s=1.0,
            min_linear_velocity_m_s=0.0,
            max_angular_velocity_deg_s=10.0,
            max_curvature=1.0,
            tolerance_xy_m=1.0,
            tolerance_theta_deg=10.0,
            environment=[],
            body=[]
        )

        graph = create_graph(ps, ps.bounds, [])

        assert(len(graph.nodes()) == 3600)

    def test_create_graph2(self):
        ps = PlanningSetup(
            bounds=Rectangle(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0),
            max_linear_velocity_m_s=1.0,
            min_linear_velocity_m_s=0.0,
            max_angular_velocity_deg_s=10.0,
            max_curvature=1.0,
            tolerance_xy_m=1.0,
            tolerance_theta_deg=10.0,
            environment=[],
            body=[]
        )

        graph = create_graph(ps, ps.bounds, [PlacedPrimitive(pose=FriendlyPose(1.0, 1.2, 0.0), primitive=Circle(0.2))])
        assert(len(graph.nodes()) == 3600 - 36 * 2)

    def test_graph_edges(self):
        ps = PlanningSetup(
            bounds=Rectangle(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0),
            max_linear_velocity_m_s=1.0,
            min_linear_velocity_m_s=0.0,
            max_angular_velocity_deg_s=10.0,
            max_curvature=1.0,
            tolerance_xy_m=1.0,
            tolerance_theta_deg=10.0,
            environment=[],
            body=[]
        )

        graph = create_graph(ps, ps.bounds, [PlacedPrimitive(pose=FriendlyPose(1.0, 1.2, 0.0), primitive=Circle(0.2))])
        add_edges(graph, ps)

    def test_rectangle_bounds(self):
        rectangle = Rectangle(xmin=-1.0, ymin=-1.0, xmax=1.0, ymax=1.0)
        pose = FriendlyPose(0.0, 0.0, 45.0)
        bounds = rectangle_bounds(rectangle, pose)
        assert(abs(bounds.xmin + 1.414213562373095) < 1e-12)
        assert(abs(bounds.ymin + 1.414213562373095) < 1e-12)
        assert(abs(bounds.xmax - 1.414213562373095) < 1e-12)
        assert(abs(bounds.ymax - 1.414213562373095) < 1e-12)

    def test_pose_to_node(self):
        ps = PlanningSetup(
            bounds=Rectangle(xmin=0.0, ymin=0.0, xmax=3.0, ymax=3.0),
            max_linear_velocity_m_s=0.4,
            min_linear_velocity_m_s=-0.3,
            max_angular_velocity_deg_s=30.0,
            max_curvature=math.inf,
            tolerance_xy_m=0.05,
            tolerance_theta_deg=20.0,
            environment=[],
            body=[]
        )
        node = pose_to_node(ps, FriendlyPose(1.0, 0.1, 0.0))
        assert(node == (100, 10, 0.0))

    def test_closest_points(self):
        primitive_bounds = Rectangle(xmin=0.0, ymin=0.0, xmax=0.1, ymax=3.0)
        bounds = Rectangle(xmin=0.0, ymin=0.0, xmax=3.0, ymax=3.0)
        tolerance = 0.05
        xmin, ymin, xmax, ymax = closest_points(primitive_bounds, bounds, tolerance)
        assert(xmin == 0.0)
        assert(ymin == 0.0)
        assert(xmax == 0.1)
        assert(ymax == 3.0)

    def test_closest_points2(self):
        primitive_bounds = Rectangle(xmin=0.8, ymin=1.0, xmax=1.2, ymax=1.4)
        bounds = Rectangle(xmin=0.0, ymin=0.0, xmax=3.0, ymax=3.0)
        tolerance = 1.0
        xmin, ymin, xmax, ymax = closest_points(primitive_bounds, bounds, tolerance)
        assert(xmin == 0.0)
        assert(ymin == 1.0)
        assert(xmax == 2.0)
        assert(ymax == 2.0)

if __name__ == '__main__':
    unittest.main()


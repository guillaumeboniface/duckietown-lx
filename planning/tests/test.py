from planning.planner import (
    find_heading_and_distance,
    create_turn,
    create_graph,
    rectangle_bounds,
    add_edges,
    pose_to_node,
    closest_points,
    plan_to_destination,
)
from planning.collision_check import check_point_collision
from planning.timed_env import TimedEnv
from aido_schemas import Context, FriendlyPose
from dt_protocols import PlanningSetup, Rectangle, Circle, PlacedPrimitive, PlanStep, Appearance, Motion
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
        node = pose_to_node(FriendlyPose(1.0, 0.1, 0.0))
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

    def test_plan_to_destination(self):
        start = (0, 0, 0)
        plan_step = PlanStep(1, math.pi / 2, 90)
        dest = plan_to_destination(plan_step, start)[-1]
        assert(dest == (100, 100, 90))

    def test_plan_to_destination2(self):
        start = (150, 150, 0)
        plan_step = PlanStep(duration=6.0, velocity_x_m_s=0.1308996938995747, angular_velocity_deg_s=30.0)
        dest = plan_to_destination(plan_step, start)[-1]
        assert(dest == (150, 200, 180))

    def test_plan_to_destination3(self):
        start = (150, 150, 0)
        plan_step = PlanStep(duration=1.0, velocity_x_m_s=0.4, angular_velocity_deg_s=0.0)
        dest = plan_to_destination(plan_step, start)[-1]
        assert(dest == (190, 150, 0))

    def test_plan_to_destination4(self):
        start = (150, 150, 90)
        plan_step = PlanStep(1, math.pi / 2, 90)
        dest = plan_to_destination(plan_step, start)[-1]
        assert(dest == (50, 250, 180))

    def test_plan_to_destination5(self):
        start = (0, 0, 0)
        plan_step = PlanStep(1, 1, 0)
        dest = plan_to_destination(plan_step, start, 2)
        assert(dest[0] == (50, 0, 0))
        assert(dest[1] == (100, 0, 0))

    def test_plan_to_destination6(self):
        start = (150, 150, 0)
        plan_step = PlanStep(duration=6.0, velocity_x_m_s=0.1308996938995747, angular_velocity_deg_s=30.0)
        dest = plan_to_destination(plan_step, start, 2)
        assert(dest[0] == (175, 175, 90))
        assert(dest[1] == (150, 200, 180))

    def test_collision(self):
        start = (207, 451, 337)
        plan_step = PlanStep(duration=11.333024885304347, velocity_x_m_s=0.4, angular_velocity_deg_s=-8.372301118745584)
        dest = plan_to_destination(plan_step, start, 2)
        ps = PlanningSetup(
            environment=[
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=0.0, ymin=0.0, xmax=0.1, ymax=5.0),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=4.9, ymin=0.0, xmax=5.0, ymax=5.0),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=0.0, ymin=4.9, xmax=5.0, ymax=5.0),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=0.0, ymin=0.0, xmax=5.0, ymax=0.1),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.539619150260162, y=2.764151352098707, theta_deg=200.25092590378074),
                    primitive=Circle(radius=0.4955915626161882),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=4.028568920007228, y=2.285092439438581, theta_deg=178.09485615456518),
                    primitive=Circle(radius=0.7575022270458687),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.6254208399349936, y=0.21987096862042532, theta_deg=320.4926048628041),
                    primitive=Circle(radius=0.6675524439185758),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.125671199410717, y=4.170206386803661, theta_deg=175.58822051345493),
                    primitive=Circle(radius=0.5670801581108957),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.51392445095758, y=3.9871575410872353, theta_deg=206.0268476129309),
                    primitive=Circle(radius=0.45107485977504896),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.6731420817120276, y=3.9482933242598985, theta_deg=42.59271886758431),
                    primitive=Circle(radius=0.6283525898487949),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.3740911457165907, y=3.1708367703589997, theta_deg=349.73363772371755),
                    primitive=Circle(radius=0.34614476347426265),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=2.9854067191328992, y=2.204884814489365, theta_deg=224.0830942194504),
                    primitive=Rectangle(
                        xmin=-0.3451722951775122, ymin=-0.22614183237669566,
                        xmax=0.3451722951775122, ymax=0.22614183237669566
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.2676078081257216, y=2.5452693717253356, theta_deg=326.80660552685725),
                    primitive=Rectangle(
                        xmin=-0.3024006603443898, ymin=-0.32153226185970135,
                        xmax=0.3024006603443898, ymax=0.32153226185970135
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=1.2234594147202889, y=2.8629082010227744, theta_deg=331.0065395116919),
                    primitive=Rectangle(
                        xmin=-0.2503264595125117, ymin=-0.3295076552525854,
                        xmax=0.2503264595125117, ymax=0.3295076552525854
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.35700000513042396, y=0.4306467698119637, theta_deg=49.52091657795261),
                    primitive=Rectangle(
                        xmin=-0.25126489066548285, ymin=-0.23991484191677537,
                        xmax=0.25126489066548285, ymax=0.23991484191677537
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.8050801066863106, y=2.090759274161887, theta_deg=155.73643133572406),
                    primitive=Rectangle(
                        xmin=-0.31161477269728577, ymin=-0.3700610259863343,
                        xmax=0.31161477269728577, ymax=0.3700610259863343
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=1.786671021694601, y=0.12334100795562519, theta_deg=127.87379050822763),
                    primitive=Rectangle(
                        xmin=-0.213766825486308, ymin=-0.23112790550116707,
                        xmax=0.213766825486308, ymax=0.23112790550116707
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=2.1969914039975658, y=0.57650745756812, theta_deg=143.55133291933856),
                    primitive=Rectangle(
                        xmin=-0.35460471946640426, ymin=-0.22756394716538508,
                        xmax=0.35460471946640426, ymax=0.22756394716538508
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
            ],
            body=[
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=-0.13, ymin=-0.045, xmax=0.07, ymax=0.045),
                    appearance=Appearance(fillcolor="blue", rel_zorder=1),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=-0.03, ymin=-0.065, xmax=0.03, ymax=0.065),
                    appearance=Appearance(fillcolor="black", rel_zorder=-1),
                ),
            ],
            bounds=Rectangle(xmin=0.0, ymin=0.0, xmax=5.0, ymax=5.0),
            max_linear_velocity_m_s=0.4,
            min_linear_velocity_m_s=-0.3,
            max_angular_velocity_deg_s=30.0,
            max_curvature=float("inf"),
            tolerance_xy_m=0.05,
            tolerance_theta_deg=20.0,
        )
        assert(check_point_collision(ps, dest))

    def test_create_turn(self):
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
        turn = create_turn(ps, 193 - 130)

    def test_timed_env(self):
        ps = PlanningSetup(
            environment=[
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.539619150260161, y=2.764151352098707, theta_deg=200.25092590378074),
                    primitive=Circle(radius=0.4955915626161882),
                    motion=Motion(
                        steps=[PlanStep(duration=11.333024885304347, velocity_x_m_s=0.4, angular_velocity_deg_s=-8.372301118745584)],
                        periodic=False
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.539619150260162, y=2.764151352098707, theta_deg=200.25092590378074),
                    primitive=Circle(radius=0.4955915626161882),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=3.539619150260163, y=2.764151352098707, theta_deg=200.25092590378074),
                    primitive=Circle(radius=0.4955915626161882),
                    motion=Motion(
                        steps=[PlanStep(duration=5.2, velocity_x_m_s=0.4, angular_velocity_deg_s=-8.372301118745584)],
                        periodic=False
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=4.0, y=2.0, theta_deg=200.25092590378074),
                    primitive=Circle(radius=0.4955915626161882),
                    motion=Motion(
                        steps=[PlanStep(duration=5.0, velocity_x_m_s=0.4, angular_velocity_deg_s=-8.372301118745584)],
                        periodic=True
                    ),
                    appearance=Appearance(fillcolor="brown"),
                ),
            ],
            body=[
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=-0.13, ymin=-0.045, xmax=0.07, ymax=0.045),
                    appearance=Appearance(fillcolor="blue", rel_zorder=1),
                ),
                PlacedPrimitive(
                    pose=FriendlyPose(x=0.0, y=0.0, theta_deg=0.0),
                    primitive=Rectangle(xmin=-0.03, ymin=-0.065, xmax=0.03, ymax=0.065),
                    appearance=Appearance(fillcolor="black", rel_zorder=-1),
                ),
            ],
            bounds=Rectangle(xmin=0.0, ymin=0.0, xmax=5.0, ymax=5.0),
            max_linear_velocity_m_s=0.4,
            min_linear_velocity_m_s=-0.3,
            max_angular_velocity_deg_s=30.0,
            max_curvature=float("inf"),
            tolerance_xy_m=0.05,
            tolerance_theta_deg=20.0,
        )
        t_env = TimedEnv(ps.environment, 5)
        assert(len(t_env.get_env(15.0)) == 4)
        periodic_prim = t_env.get_env(10.0)[3]
        assert(round(periodic_prim.pose.x) == 4)
        assert(round(periodic_prim.pose.y) == 2)


if __name__ == '__main__':
    unittest.main()


from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
        ticks: current number of ticks.
    """


    d_ticks = ticks - prev_ticks
    dphi = d_ticks / resolution * 2 * np.pi

    return dphi, ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x_curr:                  estimated x coordinate
        y_curr:                  estimated y coordinate
        theta_curr:              estimated heading
    """

    wheel_c = 2 * np.pi * R
    d_left = wheel_c * delta_phi_left / 2 / np.pi
    d_right = wheel_c * delta_phi_right / 2 / np.pi
    d_a = (d_left + d_right) / 2

    delta_theta = (d_right - d_left) / baseline
    theta_curr = theta_prev + delta_theta
    d_x = d_a * np.cos(theta_curr)
    d_y = d_a * np.sin(theta_curr)

    x_curr = x_prev + d_x
    y_curr = y_prev + d_y
    
    return x_curr, y_curr, theta_curr

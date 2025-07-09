from typing import Tuple

import numpy as np
import math


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    gradient = np.linspace(-1, -0.2, math.ceil(shape[1]/2) - math.ceil(shape[1]/4))
    res[:, math.floor(shape[1]/2):math.floor(shape[1]/4*3)] = np.tile(gradient, (shape[0], 1))
    res[:shape[0]//12*6,:] = 0 # get the top half to be neutral (too far to matter)
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    gradient = np.linspace(-0.2, -1, math.ceil(shape[1]/2) - math.ceil(shape[1]/4))
    res[:, math.ceil(shape[1]/4):math.ceil(shape[1]/2)] = np.tile(gradient, (shape[0], 1))
    res[:shape[0]//12*6,:] = 0 # get the top half to be neutral (too far to matter)
    return res

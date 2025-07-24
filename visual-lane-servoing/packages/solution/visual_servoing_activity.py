from typing import Tuple

import numpy as np
import cv2
import math

B = 4 / 15
A = 15 / 15

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    left_gradient = np.linspace(1, 0, shape[1]//2)
    steer_matrix_left = np.zeros(shape)
    b = math.floor(B * shape[1])
    a = (15 / 15 * shape[1] - b) / shape[0]
    for i in range(shape[0]):
        length = len(steer_matrix_left[i, max(0, math.floor(a*i+b) - shape[1]//2):math.floor(a*i+b)])
        steer_matrix_left[i, max(0, math.floor(a*i+b) - shape[1]//2):math.floor(a*i+b)] = left_gradient[-length:]
    steer_matrix_left[:, :shape[1]//2] = 0
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    right_gradient = np.linspace(0, -1, shape[1]//2)
    steer_matrix_right = np.zeros(shape)
    b = math.floor((1 - B) * shape[1])
    a = ((1 - A) * shape[1] - b) / shape[0]
    for i in range(shape[0]):
        length = len(steer_matrix_right[i, math.floor(a*i+b): min(math.floor(a*i+b) + shape[1]//2, shape[1])])
        steer_matrix_right[i, math.floor(a*i+b): min(math.floor(a*i+b) + shape[1]//2, shape[1])] = right_gradient[:length]
    steer_matrix_right[:, shape[1]//2:] = 0
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur the image
    imggray = cv2.GaussianBlur(imggray,(0,0), 7)

    # detect edges
    sobelx = cv2.Sobel(imggray,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(imggray,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients and threshold it
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    mask_mag = Gmag > 15

    # Get the inner line edges
    sobel_x_pos = sobelx > 0
    sobel_x_neg = sobelx < 0
    sobel_y_neg = sobely < 0

    # create colour masks
    white_lower_hsv = np.array([0, 0, 150])         
    white_upper_hsv = np.array([255, 100, 255])  
    yellow_lower_hsv = np.array([0, 100, 50])      
    yellow_upper_hsv = np.array([30, 255, 255])

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # mask half the screen
    left_mask = np.ones(imggray.shape)
    left_mask[:,imggray.shape[1]//2:] = 0
    right_mask = np.ones(imggray.shape)
    right_mask[:,:imggray.shape[1]//2] = 0

    mask_left_edge = left_mask * mask_mag * sobel_x_neg * sobel_y_neg * mask_yellow
    mask_right_edge = right_mask * mask_mag * sobel_x_pos * sobel_y_neg * mask_white

    return mask_right_edge, mask_left_edge

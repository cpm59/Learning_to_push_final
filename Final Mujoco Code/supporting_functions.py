#!/usr/bin/env python
"""Supporting functions used in the maths"""

import numpy as np
import matplotlib.pyplot as plt

def normalise_vector(vector):
    """Normalises a vector (assumes np already)"""
    return vector/np.linalg.norm(vector)

def plot_vector(axes, start_point, vector, label):
    """Plots a vector onto provided axes"""
    axes.plot([start_point[0], start_point[0] + vector[0]],[start_point[1], start_point[1] + vector[1]], label = label )

def rotate_vector(angle, vector):
    """Rotates a vector by a desired angle (radians)"""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    return np.matmul(rotation_matrix, vector)

def angle_between_vectors(vector1, vector2):
    """ Returns the angle between two vectors, between pi and -pi"""

    # Finds the angle
    vector1 = normalise_vector(vector1)
    vector2 = normalise_vector(vector2)
    theta = np.arccos(np.dot(vector1, vector2))

    # Finds the sign of the angle. 
    if np.cross(vector1, vector2) < 0:
        theta *=-1

    return theta

def random_start_position(radius):
    """Returns x, y coordinates of a random location on a circle of defined radius r, """

    theta = 2*np.pi*np.random.random()
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return np.array([x, y])

def plot_vector(axes, start_point, vector, label):
        axes.plot([start_point[0], start_point[0] + vector[0]],[start_point[1], start_point[1] + vector[1]], label = label )



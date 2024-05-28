#!/usr/bin/env python
"""Supporting functions used in the maths"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

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

def print_padded_text(text, total_length=80):
    text_length = len(text)
    padding_length = max(0, total_length - text_length)
    padding_left = padding_length // 2
    padding_right = padding_length - padding_left
    padded_text = '<'+'-' * padding_left + text + '-' * padding_right+ '>'
    print(padded_text)

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                                 [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                                 [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]])
    return rotation_matrix

def get_datetime():
    """Gets the date time and strips it for bad characters"""
    date_time = str(datetime.datetime.now())
    date_time = date_time.replace('.', '_')
    date_time = date_time.replace(':', '_')
    date_time = date_time.replace('-', '_')
    return date_time

def get_most_recent_file(folder):
    """ Gets the most recent file from a folder; assuming that they're named in the date time format"""
    files = os.listdir(folder)
    files.sort()
    most_recent_calibration = files[-1]
    return most_recent_calibration

def contains_subfolder(folder):
    """ Checks if something contains a subfolder. """

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            return True
    return False


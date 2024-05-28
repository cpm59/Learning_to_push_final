#!/usr/bin/env python
"""This file runs the main code, irrespective of robot/mujoco control"""
# Module import
import numpy as np
import matplotlib.pyplot as plt

# Other file import
import physical_classes
import supporting_functions
import mujoco_code
import controllers

def main(implementation="mujoco", method="window", step_size = 0.01, mode = "normal"):
    """Runs the code!"""
    """methods = "window", "dipole", "krivic" """

    # Sets a target point, gets robot/target positions, and starts the tracking objects
    system = intialise(implementation)
    system.method = method
    system.step_size = step_size
    system.mode = mode

    if method =="krivic":
        system.kro = controllers.krivics_object()

    if implementation == "mujoco":
        
        mujoco_code.run_mujoco_model(system, method)

    system.plot_trajectories()
    if method =="krivic":
        system.kro.plot_current_v_original_distribution()

    # plt.scatter(np.linspace(0, 100, len(system.target.orientation)), system.target.orientation)
    # plt.show()


def intialise(implementation):
    """Sets a target point, gets robot/target positions, and starts the tracking objects"""

    # Establishes a goal point
    goal_point = supporting_functions.random_start_position(1)

    # The initial positions should be known for the start of the code.
    if implementation == "mujoco":
        robot_position = supporting_functions.random_start_position(0.25)
        # robot_position = goal_point
        target_position = np.array([0,0])
    
    # Starts the tracking objects

    target = physical_classes.physical_object(target_position)
    robot = physical_classes.robot(robot_position)
    system = physical_classes.system(robot, target, goal_point)

    return system



main(method="krivic", step_size=0.06, mode="normal")
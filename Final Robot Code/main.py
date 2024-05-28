#!/usr/bin/env python
"""This file runs the main code, irrespective of robot/mujoco control"""
# Module import
import numpy as np
import matplotlib.pyplot as plt
import transformations as tf
import os
from os.path import isfile, join
import csv

# Other file import
import physical_classes
import supporting_functions
import robot_code
import controllers
import threading
import time
import datetime
from mpl_toolkits.mplot3d import Axes3D
import progressbar
import saving


def main(implementation="mujoco", method="window",step_size=0.1 ,calibrate =True, use_previous_learning =False, vary_step_size = False):
    """Runs the code!"""
    """methods = "window", "dipole", "krivic" """

    # Sets a target point, gets robot/target positions, and starts the tracking objects
    supporting_functions.print_padded_text("Initialisation")
    system = intialise()
    system.method = method
    system.mode = "normal"
    system.kro = controllers.krivics_object()
    system.saver = saving.data_tracker()
    system.calibrate = calibrate
    system.use_previous_learning = use_previous_learning
    system.vary_step_size = vary_step_size
    system.step_size = step_size
    system.z = 0.115 +0.0
    

    # Starts listening to the robot and decision threads
    rr = robot_code.rospy_runner()
    thread1 = threading.Thread(target=rr.run)
    thread2 = threading.Thread(target=robot_working, args=(rr, system))

    # system.plot_trajectories()
    thread1.start()
    thread2.start()

def robot_working(rr, system):
    """runs the decision making thread"""
    time.sleep(1)

    system.date_time = supporting_functions.get_datetime()

    if system.calibrate:
        system.t, system.R = calibrate_2(rr, system)
    else:
        system.t, system.R = load_calibration_file()

    system.R_inv = np.linalg.inv(system.R)

    # input("Press Key to start Pushing")

    system.robot.trajectory = []
    system.target.trajectory = []

    robot_position_update(rr, system)

    if system.method == "krivic" and system.use_previous_learning ==True and supporting_functions.contains_subfolder("Run_files") == True:
        system.kro.load_data()

    
    
    if system.step_size >0.2:
        raise ValueError("Step size seems inappropriately large")
    step_size = system.step_size
    

    # threshold defines the accuracy to the goal point
    threshold = min(step_size, 0.02)

    system.robot.distance_to_path_point = 0

    # angle = -2*np.pi
    # num = 36
    # angles = np.linspace(angle*1/num, angle, num)
    # radius = 0.15
    # goals = []
    # for angle in angles:
    #     point = np.array([0, radius])
    #     goal = radius*supporting_functions.rotate_vector(angle, np.array([0,-1]))+point
    #     goals.append(goal)

    # goals = [np.array([0,0.4]),np.array([0.1,0.4]), np.array([0.1,0]), np.array([0,0])]
    goals = [np.array([0,0.6])]
    # goals = [np.array([0, 0.2])]

    # Everything here is in the robot frame of reference
    system.goal_point[0] = system.target.trajectory[0]+ goals[0]
    move_to_start(rr, system)
    system.find_distance_to_goal_point()

    # input("Press Key to start Pushing")
    supporting_functions.print_padded_text("Pushing")
    t1 = time.time()

    system.start_time = t1
    for i in range(len(goals)):
        # move_to_start(rr, system)
        
        
        system.robot.distance_to_path_point = 0
        system.goal_point[0] = system.target.trajectory[0]+ goals[i]
        system.find_distance_to_goal_point()

        if i != 0:

            move_in_plane_to_new_start(system, rr)
            # move_to_start(rr, system)
        inner_loop(rr, system, step_size, threshold)
        # move_to_end(rr, system)
        
        
    
    t2 = time.time()
    
    move_to_end(rr, system)

    # Saves the run level data
    system.saver.update_run_file(t2-t1, step_size, threshold, system.method)
    system.saver.save_learned_data(system, system.kro)
    system.saver.save_files(system.date_time)

    supporting_functions.print_padded_text("Program End")
    
def inner_loop(rr, system, step_size, threshold):
    
    initial_distance_to_goal = system.distance_to_goal_point - threshold 
    
    bar = progressbar.ProgressBar(maxval=100)
    bar.start()

    upper = step_size
    if system.vary_step_size == True:
        lower = 0.04
    else:
        lower = upper
    step_size = upper
    
    while system.distance_to_goal_point > threshold and (time.time() - system.start_time) < 30:

        robot_position_update(rr, system)
        
        
        system.find_distance_to_goal_point()
        percentage_to_goal = (initial_distance_to_goal - (system.distance_to_goal_point-threshold ))/initial_distance_to_goal
        

        if system.robot.distance_to_path_point < step_size-0.005:
        # if system.robot.distance_to_path_point < 5*step_size/10:
            step_size = get_variable_step_size(percentage_to_goal, upper, lower)
            controllers.set_path_point(system, system.method, rr, step_size)
            system.robot.find_distance_to_path_point()
            z_position = max(system.z, rr.robot_robot_pose[2]-0.05)
            # debugging_info(rr, system)
            system.saver.update_time_step_file(system)
            rr.publish_pose(system.robot.path_point[-1][0], system.robot.path_point[-1][1], z_position, 0.95, -0.3, 0.013, 0.0)
        else:
            system.robot.find_distance_to_path_point()
            # debugging_info(rr, system)
            system.saver.update_time_step_file(system)

        # percentage_to_goal = int(100*(initial_distance_to_goal - (system.distance_to_goal_point-threshold ))/initial_distance_to_goal)
        
        bar.update(min(max(int(100*percentage_to_goal),0),100))
        system.time += 1
        time.sleep(0.001)

    bar.finish()
    if (time.time() - system.start_time) > 30:
        supporting_functions.print_padded_text("Goal Failed")
    else:
        supporting_functions.print_padded_text("Goal Reached")
    
def get_variable_step_size(percentage_to_goal, upper, lower):
    step_size = upper - percentage_to_goal**2 * (upper-lower)
    return step_size

def move_in_plane_to_new_start(system, rr):
    """this should move the robot to push toward the next goal point. """

    z = system.z
    robot_position_update(rr, system)
    if abs(system.gammas[-1]) < 60*np.pi/180:
        return
    rr.publish_pose(rr.robot_robot_pose[0] + system.push_vector[0]*-(0.05), rr.robot_robot_pose[1] + system.push_vector[1]*-(0.05), z, 0.95, -0.3, 0.013, 0.0)
    time.sleep(0.2)
    robot_position_update(rr, system)
    controllers.set_direction_vector(system)
    angle_to_move = supporting_functions.angle_between_vectors(system.push_vector, system.goal_vector_absolute_frame)/2
    distance_to_object =  np.linalg.norm(system.robot.trajectory[-1] - system.target.trajectory[-1])

    angles= np.linspace(0, angle_to_move, 10)
    for angle in angles:
        if angle == 0.0:
            continue
        # Finds the point along which to move with respect to the target
        point = system.target.trajectory[-1]-distance_to_object*supporting_functions.rotate_vector(angle, system.push_vector)

        # Speeds up the movement a bit
        movement_vector = point - system.robot.trajectory[-1]
        point = supporting_functions.normalise_vector(movement_vector)*0.1 + system.robot.trajectory[-1]
        rr.publish_pose(point[0], point[1], z, 0.95, -0.3, 0.013, 0.0)
        time.sleep(0.05)
        robot_position_update(rr, system)

    time.sleep(0.2)



def move_to_start(rr, system):

    z_lim = system.z
    target_x_y = system.target.trajectory[-1] + 0.13*supporting_functions.normalise_vector(-system.goal_point[0] + system.target.trajectory[-1])
    # target_x_y[0]-=0.01

    movement_vector = target_x_y-rr.robot_robot_pose[:2]
    movement_vector = supporting_functions.normalise_vector(movement_vector)
    z = rr.robot_robot_pose[2]

    while np.linalg.norm(target_x_y-rr.robot_robot_pose[:2]) > 0.1:
        rr.publish_pose(rr.robot_robot_pose[0] + movement_vector[0]*0.2, rr.robot_robot_pose[1] + movement_vector[1]*0.2, z, 0.95, -0.3, 0.013, 0.0)
        time.sleep(0.05)
        robot_position_update(rr, system)
        movement_vector = supporting_functions.normalise_vector(movement_vector)
    
    z = max(z_lim,rr.robot_robot_pose[2]-0.05)
    while z != z_lim:
        rr.publish_pose(target_x_y[0], target_x_y[1], z, 0.95, -0.3, 0.013, 0.0)
        z = max(z_lim,rr.robot_robot_pose[2]-0.1)
        time.sleep(0.01)
        robot_position_update(rr, system)
    
    time.sleep(1.5)


def move_to_end(rr, system):
    
    z = min(0.4,rr.robot_robot_pose[2]+0.05)
    while z != 0.4:
        rr.publish_pose(rr.robot_robot_pose[0], rr.robot_robot_pose[1], z, 0.95, -0.3, 0.013, 0.0)
        z = min(0.4,rr.robot_robot_pose[2]+0.1)
        time.sleep(0.01)
        robot_position_update(rr, system)


def debugging_info(rr, system):

    names = [
    "\nGoal point: ",
    "Robot Position (Opti):",
    "Robot Position (Robot):",
    "Target Position:",
    "Path Point:",
    "Distance to Point:"
    ]

    x = [
        system.goal_point[0][0],
        system.robot.trajectory[-1][0],
        rr.robot_robot_pose[0],
        system.target.trajectory[-1][0],
        system.robot.path_point[-1][0],
        system.robot.distance_to_path_point
    ]

    y = [
        system.goal_point[0][1],
        system.robot.trajectory[-1][1],
        rr.robot_robot_pose[1],
        system.target.trajectory[-1][1],
        system.robot.path_point[-1][1],
        0
    ]

    max_len = max(len(name) for name in names)
    for name, xi, yi in zip(names, x, y):
        print(f"{name.ljust(max_len + 2)}{round(xi, 8)}, {round(yi, 8)}")
    # print(f"Current Direction: \t\t: {system.current_direction[0]}, {system.current_direction[1]}\n")

def robot_position_update(rr, system):

    x1 = system.t+ system.R @ rr.opti_robot_pose[:3]
    x2 = system.t + system.R @ rr.opti_target_pose[:3]
    x3 = rr.robot_robot_pose[:3]
    target_quaternion = rr.opti_target_pose[3:]
    rotation_matrix = supporting_functions.quaternion_to_rotation_matrix(target_quaternion)
    theta_1 = rotation_matrix[2, 0]
    theta_2 = rotation_matrix[2, 2]
    target_orientation = np.arctan2(theta_1, theta_2)
    system.target.orientation.append(target_orientation)
    # print(x3, "\n",x1, "\n", x2, "\n")
    system.robot.trajectory.append(x1[:2])
    system.target.trajectory.append(x2[:2])

    system.find_push_vector()
    orientation_vector = supporting_functions.rotate_vector(system.target.orientation[-1] , np.array([1,0]))
    orientation_robot_target = supporting_functions.angle_between_vectors(system.push_vector, orientation_vector)
    system.orientations_robot_target.append(orientation_robot_target)

    system.gammas.append(system.find_gamma())

def calibrate_2(rr, system):
    supporting_functions.print_padded_text("Calibrating")
    calibration_data_robot = [rr.robot_robot_pose[:3]]
    calibration_data_optitrack = [rr.opti_robot_pose[:3]]
    # calibration_poses = [[0.1, 0, 0], [0, 0.1, 0],[0, 0, 0.1],[-0, 0, -0.1],[-0, -0.1, 0],[-0.1, 0, -0]]

    # Moves in a Cube
    calibration_poses = [[0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0], [0, -0.1, 0], [0,0,0.1],[0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0], [0, -0.1, 0], [0,0,-0.1]]

    # Random poses
    # calibration_poses = random_calibration_poses(20)
    # Move the robot to collect calibration data
    bar = progressbar.ProgressBar(maxval=len(calibration_poses))
    bar.start()
    
    for j, pose in enumerate(calibration_poses):
        pose += rr.robot_robot_pose[0:3]
        rr.publish_pose(pose[0], pose[1], pose[2], 0.95, -0.3, 0.013, 0.0)
        bar.update(j)

        # Wait for some time to stabilize
        time.sleep(0.75)  # Adjust the sleep duration as needed

        for i in range(1):
            robot_pose = rr.robot_robot_pose[:3]
            optitrack_pose = rr.opti_robot_pose[:3]

            # Append the collected poses
            calibration_data_robot.append(robot_pose)
            calibration_data_optitrack.append(optitrack_pose)

            # time.sleep(.2)
    bar.finish()

    calibration_data_robot = np.array(calibration_data_robot)
    calibration_data_optitrack = np.array(calibration_data_optitrack)

    date_time = system.date_time
    # np.savetxt(f"Calibration_files/opti_test_data_{date_time}.csv", calibration_data_optitrack, delimiter=",")
    # np.savetxt(f"Calibration_files/robot_test_data_{date_time}.csv", calibration_data_robot, delimiter=",")

    t, R  = find_transformation(calibration_data_optitrack, calibration_data_robot)


    calibration_data_opti_in_robot = [(t + R @ calibration_data_optitrack[i]) for i in range(calibration_data_optitrack.shape[0])]
    # np.savetxt(f"Calibration_files/calibrated_test_data_{date_time}.csv", calibration_data_opti_in_robot, delimiter=",")
    np.savetxt(f"Calibration_files/{date_time}.csv", np.array([t, R[0], R[1], R[2]]), delimiter=',')
    # plot_point_sets(calibration_data_robot, calibration_data_opti_in_robot, calibration_data_optitrack)
    return t, R

def load_calibration_file():
    most_recent_calibration = supporting_functions.get_most_recent_file("Calibration_files")
    array = np.genfromtxt(f"Calibration_files/{most_recent_calibration}", delimiter=',')
    t = array[0]
    R = np.ones((3,3))
    R[0] = array[1]
    R[1] = array[2]
    R[2] = array[3]

    return t, R


def find_transformation(pointsA, pointsB):
    """This finds a mapping from A to B. """
    assert len(pointsA) == len(pointsB)
    N = len(pointsA)
    comA = np.sum(pointsA, axis=0) / len(pointsA)
    comB = np.sum(pointsB, axis=0) / len(pointsB)

    p_A = [p - comA for p in pointsA]
    p_B = [p - comB for p in pointsB]

    H = np.zeros((3, 3))
    for i in range(N):
        H += np.outer(p_A[i], p_B[i])

    U, _, V = np.linalg.svd(H)
    # Flip direction of least significant vector to turn into a rotation if a reflection is found.
    # Required when the points are all coplanar.
    V = V.T @ np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, np.linalg.det(V) * np.linalg.det(U)]])
    
    R = V @ U.T
    # Double check
    # assert np.allclose(R @ comA, comB - R @ comA)
    assert np.isclose(np.linalg.det(R), 1)

    translation = comB - R @ comA

    # Compute errors
    errs = [pointsB[i] - (translation + R @ pointsA[i]) for i in range(N)]
    dists = [np.linalg.norm(err) for err in errs]
    avg_error = sum(dists) / len(dists)
    max_error = max(dists)
    
    max_vector_error, average_vector_error = check_error_between_vectors(pointsA, pointsB)
    print(f"Calibration Error: Average: {1e3 * avg_error}mm Max: {1e3 * max_error}mm\nVector Error:      Average: {1e3 * average_vector_error}mm Max: {1e3* max_vector_error}mm")

    return translation, R

def check_error_between_vectors(set_A, set_B):
    set_a_vector = np.diff(set_A, axis=0)
    set_b_vector = np.diff(set_B, axis=0)

    set_a_norm = np.linalg.norm(set_a_vector, axis=1)
    set_b_norm = np.linalg.norm(set_b_vector, axis=1)

    norms = np.abs(set_a_norm - set_b_norm)

    return np.max(norms), np.average(norms)

def intialise():
    """Sets a target point, gets robot/target positions, and starts the tracking objects"""

    # Establishes a goal point
    # goal_point = supporting_functions.random_start_position(0.2)
    goal_point = np.array([0, 0.6])

    
    target_position = np.array([0,0])
    robot_position = np.array([0,0])
    
    # Starts the tracking objects
    target = physical_classes.physical_object(target_position)
    robot = physical_classes.robot(robot_position)
    system = physical_classes.system(robot, target, goal_point)

    return system



def set_equal_aspect(ax):
    ax.set_box_aspect([u - l for l, u in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

def setup3Daxis(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_equal_aspect(ax)

def plot_point_sets(robot, opti_in_robot, opti):
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    plot_pointset(ax1, robot, 'tab:blue', "Robot")
    plot_pointset(ax2, opti_in_robot, 'tab:purple', "Opti in Robot")
    plot_pointset(ax3, opti, 'tab:red', "Opti ")

    plot_pointset(ax4, robot, 'tab:blue', "Robot")
    plot_pointset(ax4, opti_in_robot, 'tab:purple', "Opti in Robot")
    
    plot_lines(ax4, robot, opti_in_robot)

    for ax in (ax1, ax2, ax3, ax4):
        setup3Daxis(ax)

    # ax.set_title('Robot and Opti->Robot Points')
    

    plt.show()

def plot_pointset(ax, vertices, color, label):
    # Plot the vertices
    for i, vertex in enumerate(vertices):
        ax.scatter(vertex[0], vertex[1], vertex[2], color=color, label = label)
        ax.text(vertex[0], vertex[1], vertex[2], f"{i}", color=color)

def plot_lines(ax, pointset1, pointset2):
    for i in range(pointset1.shape[0]):
        ax.plot3D([pointset1[i][ 0], pointset2[i][0]],
                  [pointset1[i][1], pointset2[i][1]],
                  [pointset1[i][2], pointset2[i][2]], color="black")

def test_integral_control_thread():
    rr = robot_code.rospy_runner()
    thread1 = threading.Thread(target=rr.run)
    thread2 = threading.Thread(target=test_integral_control, args=([rr]))
    thread1.start()
    thread2.start()

def test_integral_control(rr):
    time.sleep(1)
    goal_point = rr.robot_robot_pose[:3] + np.array([0.1,0,0])
    print(goal_point)
    rr.publish_pose(goal_point[0], goal_point[1], goal_point[2], 0.95, -0.3, 0.013, 0.0)
    time.sleep(10)
    print(np.linalg.norm(goal_point-rr.robot_robot_pose[:3]))
    np.savetxt("integral testing.csv", np.array(rr.robot_poses), delimiter=",")

def get_mostrecent_csv_that_starts_with(csvs, startswith, path):
    filtered = [f for f in csvs if f.startswith(startswith)]
    file = sorted(filtered)[-1]
    print(file)
    return np.genfromtxt(f"{path}/{file}", delimiter=",")

def plot_saved_files():
    path = "Calibration_files"
    files = os.listdir(path)
    csvs = [f for f in files if (isfile(join(path,f)) and f.endswith(".csv"))]

    robot = get_mostrecent_csv_that_starts_with(csvs, "robot", path)
    # robot = np.array([[point[1], point[0], point[2]] for point in robot])
    opti = get_mostrecent_csv_that_starts_with(csvs, "opti", path)
    # calibrated = get_mostrecent_csv_that_starts_with(csvs, "cali", path)
    t, R = find_transformation(opti, robot)
    # print(t)
    calibrated = [(t + R @ point)  for point in opti]
    calibrated = np.array(calibrated)
    plot_point_sets(robot, calibrated, opti)



if __name__ == "__main__":
    supporting_functions.print_padded_text("Program Start")
    main(method="krivic", calibrate=False, use_previous_learning=True, vary_step_size=False, step_size=0.08)
    # test_integral_control_thread()
    # load_calibration_file()





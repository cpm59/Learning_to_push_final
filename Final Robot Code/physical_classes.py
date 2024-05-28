#!/usr/bin/env python
"""Robot/Object classes"""
import numpy as np
import matplotlib.pyplot as plt
import supporting_functions

class physical_object:
        
    def __init__(self, position):
        """Handles common features between robot and object"""

        # Starts lists to track position and object orientation
        self.trajectory = [position]
        self.orientation = []
        
class robot(physical_object):
        
    def __init__(self, position):

        # position and orientation handled by super
        super().__init__(position)

        # velocity, path planning lists
        self.velocity = [np.array([0,0])]
        self.path_point = []

    def find_distance_to_path_point(self):
        """Finds the euclidean distance to the robots path point"""
        vector = self.path_point[-1] - self.trajectory[-1]

        distance = np.linalg.norm(vector, ord=2)

        self.distance_to_path_point = distance

class system:
    # tracks system wide concepts; object-robot interactions, controller terms, goal points, etc.
    def __init__(self, robot, target, goal_point):
        
        self.robot_object_angle = []
        self.goal_point = [goal_point]
        self.robot = robot
        self.target = target
        self.time = 0
        self.orientations_robot_target = []
        self.gammas=[]

    def find_push_vector(self):
        """Finds the vector between the target and robot, from the robot frame."""

        # Finds the direction
        push_vector = self.target.trajectory[-1] - self.robot.trajectory[-1]

        # Normalises the vector
        push_vector = supporting_functions.normalise_vector(push_vector)
        
        # Stores it in the system object
        self.push_vector = push_vector
        
    def goal_vector(self):
        """Finds the vector between the target and goal point, in the cartesian plane"""

        # Finds the direction
        goal_vector = self.goal_point[-1] - self.target.trajectory[-1]

        # Normalises the vector
        goal_vector = supporting_functions.normalise_vector(goal_vector)

        # Stores in the system object.
        self.goal_vector_absolute_frame = goal_vector 

        # Finds the same vector in the target fram via rotation from the most recent orientation call.
        self.goal_vector_target_frame = supporting_functions.rotate_vector(self.target.orientation[-1] - self.target.orientation[0], goal_vector)
    
    def find_distance_to_goal_point(self):
        """Finds the euclidean distance to the objects goal point"""
        vector = self.goal_point[-1] - self.target.trajectory[-1]

        distance = np.linalg.norm(vector, ord=2)

        self.distance_to_goal_point = distance


    def find_gamma(self):
        self.goal_vector()
        self.find_push_vector()
        theta_error = supporting_functions.angle_between_vectors(self.goal_vector_absolute_frame, self.push_vector)
        return theta_error
    
    def plot_trajectories(self):
        """Plots the robot and target trajectories on a 2d plane"""


        if len(self.robot.trajectory) <2:
            raise ValueError("Trajectory is less than length two, so graph cannot be plotted.")
        robot_trajectory =np.array(self.robot.trajectory)
        target_trajectory =np.array(self.target.trajectory)
        goal_point=np.array(self.goal_point)
        plt.scatter(robot_trajectory[:,0], robot_trajectory[:,1], color = 'tab:blue', label = "Robot", s=2)
        plt.scatter(target_trajectory[:,0], target_trajectory[:,1], color = 'tab:orange', label = "Target", s=2)
        plt.scatter(goal_point[-1,0], goal_point[-1,1], color= 'tab:red', label="Goal", s=3)
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Robot-Object Trajectories")
        limit = 1
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.show()





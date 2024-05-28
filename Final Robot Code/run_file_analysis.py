import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from os.path import isfile, join

def gather_file():
    path = "Fourth Year Project\\004 Programming\learning-to-push\Linux code\Run_files"
    files = os.listdir(path)
    csvs = [f for f in files if (isfile(join(path,f)) and f.endswith(".csv"))]
    for csv in sorted(csvs, reverse=True):
        df = pd.read_csv(join(path,csv))
        goal_point = np.array(df.iloc[:, 0])
        robot_trajectory = np.array(df.iloc[:, 1])
        robot_robot_pose = np.array(df.iloc[:, 2])
        target_trajectory = np.array(df.iloc[:,3])
        robot_path_point = np.array(df.iloc[:, 4])
        distance_to_path_point = np.array(df.iloc[:, 5])

        for i, goal_point_row in enumerate(goal_point):
            float_list = [float(x) for x in goal_point_row.split(',')]
            goal_point[i] = float_list

        for i, goal_point_row in enumerate(robot_trajectory):
            float_list = [float(x) for x in goal_point_row.split(',')]
            robot_trajectory[i] = float_list
        
        robot_trajectory = robot_trajectory.astype(float)

        for i, goal_point_row in enumerate(target_trajectory):
            float_list = [float(x) for x in goal_point_row.split(',')]
            target_trajectory[i] = float_list
        target_trajectory = np.array(target_trajectory).astype(float)


        
        plt.scatter(goal_point[0][0], goal_point[0][1], s = 5, color='tab:red')
        plt.scatter(robot_trajectory[:, 0], robot_trajectory[:, 1], s= 3, color='tab:blue')
        plt.scatter(target_trajectory[:, 0], target_trajectory[:, 0], s= 3, color='tab:orange')
        plt.title(csv)
        plt.show()
        
           
# filename = 'your_file.csv'
# data = read_csv_file(filename)

# for row in data:
#     goal_point, robot_trajectory, robot_robot_pose, target_trajectory, robot_path_point, distance_to_path_point = row
#     print("Goal Point:", goal_point)
#     print("Robot Trajectory:", robot_trajectory)
#     print("Robot Robot Pose:", robot_robot_pose)
#     print("Target Trajectory:", target_trajectory)
#     print("Robot Path Point:", robot_path_point)
#     print("Distance to Path Point:", distance_to_path_point)
        
gather_file()
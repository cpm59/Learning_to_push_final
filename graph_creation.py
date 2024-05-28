import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter
import pickle
import time
from tkinter import filedialog
from shapely.geometry import Point, LineString
from scipy import integrate
from scipy.special import i0
from tol_colors import tol_cmap, tol_cset
tex_fonts = {

    "text.usetex": True,
    "font.family": "serif",
    'font.serif': 'Computer Modern',

    "axes.labelsize": 12,
    "font.size": 12,

    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

# plt.rcParams.update(tex_fonts)
import shutil
robo_colour = "tab:gray"
# Control Plot font size

font_size = 24
plt.rcParams.update({'font.size': font_size})
plt.rcParams['lines.linewidth'] = 5
plt.rcParams["figure.autolayout"] = True

plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.axisbelow'] = True

class run:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.time_step_df = pd.read_csv(f"{self.folder_path}/timestep.csv")
        self.run_df = pd.read_csv(f"{self.folder_path}/run.csv")
        self.pathstep_df = pd.read_csv(f"{self.folder_path}/pathstep.csv")   
        self.method = self.run_df['implementation'][0]
        self.t = np.asarray(self.time_step_df['time'])
        self.threshold = self.run_df['threshold'][0]

        self.get_learning_data()
        self.add_xy_data()
        self.get_goal_point_array()
        self.segment_data_into_goal_point_chunks()
        self.find_ideal_path()
        self.recover_gammas()
        self.get_error_from_path(0.01)

    def pickle_run(self, title):
        folder_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Pickle Files"
        with open(f"{folder_path}\{title}", 'wb') as file:
            pickle.dump(self, file)

    def recover_gammas(self):

        gamma = []
        push_vector = np.array([self.x-self.robo_x, self.y-self.robo_y])
        goal_vector = np.array([self.full_goal_x - self.x, self.full_goal_y - self.y ])
        
        for i in range(len(self.x)):
            gamma.append(angle_between_vectors(goal_vector[:,i], push_vector[:,i]))
        
        self.gammas = np.array(gamma)

    def plot_gammas(self):
        
        distance_travelled = self.distance_travelled[self.learning_time_step]
        distance_travelled = self.distance_travelled
        plt.plot(distance_travelled, self.gammas[:-1])
        plt.show()

    def get_learning_data(self):
        """If the method is adaptive it'll pull the learning data"""
        if self.method == "krivic":
            self.learning_df = pd.read_csv(f"{self.folder_path}/learning.csv")
            self.ff = np.array(self.learning_df['psi_feedforward'])
            self.fb = np.array(self.learning_df['psi_feedback'])
            self.learning_time_step = np.array(self.learning_df['time_step'])
            self.gammas = np.array(self.learning_df['error'])

            self.mus = np.array(self.learning_df['mu'])
            self.kappas = np.array(self.learning_df['kappa'])

            self.learned_data_df = pd.read_csv(f"{self.folder_path}/learned_data.csv")
            self.alphas = np.array(self.learned_data_df['alphas'])

        
    def add_xy_data(self):
        time_step_df = self.time_step_df
        x = np.array(time_step_df['Target_x'])
        y = np.array(time_step_df['Target_y'])

        x_0 = time_step_df['Target_x'][0]
        y_0 = time_step_df['Target_y'][0]

        self.x = normalise_array_to_reference_value(x, x_0)
        self.y = normalise_array_to_reference_value(y, y_0)
        self.x_0 = x_0
        self.y_0 = y_0

        robo_x = np.array(time_step_df['Robot_x'])
        robo_y = np.array(time_step_df['Robot_y'])
        robo_x = normalise_array_to_reference_value(robo_x, x_0)
        robo_y = normalise_array_to_reference_value(robo_y, y_0)

        self.robo_x = robo_x
        self.robo_y = robo_y

        # add distance travelled
        x_diff =np.diff(x)
        y_diff = np.diff(y)

        self.x_diff = x_diff
        self.y_diff = y_diff

        absolute_diff = np.sqrt(x_diff**2 + y_diff**2)
        self.distance_travelled = np.cumsum(absolute_diff)
        self.path_length = self.distance_travelled[-1]

        self.distance_to_goal = np.asarray(self.time_step_df['Distance_to_goal'])

        self.target_orientation = np.asarray(self.time_step_df['Target_theta'])
        self.target_orientation = self.clean_theta_0(self.target_orientation)

        self.contact_orientation = np.asarray(self.time_step_df['robot_target_orientation'])

        self.theta_0 = self.target_orientation[0]

        self.target_orientation = circular_normalise_array_to_reference_value(self.target_orientation, self.theta_0)
        self.contact_orientation = circular_normalise_array_to_reference_value(self.contact_orientation, self.theta_0)

        full_goal_x = np.array(self.time_step_df['Goal_x'])
        full_goal_y = np.array(self.time_step_df['Goal_y'])

        self.full_goal_x = normalise_array_to_reference_value(full_goal_x, self.x_0)
        self.full_goal_y = normalise_array_to_reference_value(full_goal_y, self.y_0)

    def find_distance_projected_onto_goal_vector(self):
        """Projects movement at each timestep onto the goal vector"""
        movement_along_goal_vector = [0]
        # skipping the first index as no movement has occured; this accounts for the difference in the array lengths. 
        for i in range(1, len(self.full_goal_x)):
            ith_position = np.array([self.x[i], self.y[i]])
            ith_goalposition = np.array([self.full_goal_x[i], self.full_goal_y[i]])
            ith_goal_vector = normalise_vector(ith_goalposition - ith_position)
            movement_vector =  np.array([self.x_diff[i-1], self.y_diff[i-1]])
            
            movement_along_goal_vector.append(np.dot(movement_vector, ith_goal_vector)+movement_along_goal_vector[-1])

        self.distance_along_goal_vector = movement_along_goal_vector

    def clean_theta_0(self, target_orientation):
        allowance = 0.1
        for i in range(1, len(target_orientation)):
            if target_orientation[i] <  target_orientation[i-1] + (np.pi * (1+allowance))  and  target_orientation[i] >  target_orientation[i-1] + (np.pi * (1-allowance)):        
                target_orientation[i] -= np.pi
            elif target_orientation[i] <  target_orientation[i-1] - (np.pi * (1-allowance))  and  target_orientation[i] >  target_orientation[i-1] - (np.pi * (1+allowance)):
                target_orientation[i] += np.pi
        return target_orientation

    def get_goal_point_array(self):
        goal_x = np.array(self.time_step_df['Goal_x'])
        goal_y = np.array(self.time_step_df['Goal_y'])

        # First point is always unique
        unique_x = [goal_x[0]]
        unique_y = [goal_y[0]]
        unique_positions = [0]

        for i in range(1, goal_x.shape[0]):
                if goal_x[i] != goal_x[i-1] or goal_y[i] != goal_y[i-1]:
                    unique_x.append(goal_x[i])
                    unique_y.append(goal_y[i])
                    unique_positions.append(i)

        # Checks to see they're the same length
        if len(unique_x) != len(unique_y):
            raise(ValueError("When getting unique values from the goal point segments, different numbers of points were found in x and y")) 
        

        self.goal_x = normalise_array_to_reference_value(unique_x, self.x_0)
        self.goal_y = normalise_array_to_reference_value(unique_y, self.y_0)
        self.unique_goal_timesteps = unique_positions

    def segment_data_into_goal_point_chunks(self):
        """Divides time_segment data into chunks based on each goal point to calculate errors and path lengths."""

        n = len(self.unique_goal_timesteps)
        # If there's only one goal point then we want to pass that as the time_step_data
        segments = []
        if n > 1:
            for i in range(n):
                if i == n-1:
                    data = self.time_step_df[self.unique_goal_timesteps[i]:]
                else:
                    data = self.time_step_df[self.unique_goal_timesteps[i]:self.unique_goal_timesteps[i+1]]
                segments.append(data)
        else:
            segments = [self.time_step_df]

        self.segments = segments

    def make_trajectory_graph(self):
        fig, ax = plt.subplots()
        ax.plot(self.robo_x, self.robo_y,  color = 'tab:blue', label = 'Robot')
        ax.plot(self.x, self.y, color = 'tab:orange', label = 'Target')

        threshold = self.threshold
        theta = np.linspace(-np.pi, np.pi, 360)

        for i in range(len(self.goal_x)):
            ax.plot(self.goal_x[i]+threshold*np.cos(theta), self.goal_y[i]+threshold*np.sin(theta), color= 'black', linewidth = 0.5, linestyle=":")
        ax.scatter(self.goal_x, self.goal_y, marker = "x", color = "black", label = f"Goal Point(s)", linewidths=0.75, s =7 )
        
        ax.plot(self.ideal_path_x, self.ideal_path_y, color= 'black', linewidth = 0.5, label="Ideal Path")

        y_lim = 0.7
        x_lim = 0.35
        ax.set_xlim((-x_lim, x_lim))
        ax.set_ylim((-0.15, y_lim))
        ax.set_xlabel("x_position (m)")
        ax.set_ylabel("y_position (m)")
        ax.grid()
        # ax.legend()

        # ax.set_title(f"Method = {self.run_df['implementation'][0]}, Time = {round(self.run_df['Run_time'][0], 3)}s, Step Size = {self.run_df['Step_size'][0]}")
        ax.set_aspect('equal')
        # fig.savefig(f"{self.folder_path}/Trajectory.png")
        # plt.close('all')
        plt.show()

    def plot_distance_v_time(self):
        fig, ax = plt.subplots()
        
        ax.plot(self.t, self.distance_to_goal, color='tab:blue')
        ax.grid()
        ax.set_title(f"Method = {self.run_df['implementation'][0]}, Time = {round(self.run_df['Run_time'][0], 3)}s, Step Size = {self.run_df['Step_size'][0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance to Goal")
        
        fig.savefig(f"{self.folder_path}/Distance_v_Time.png")
        plt.close('all')
        # plt.show()

    def plot_feedforward_feedback(self):
        """Plots the feedforward v. feedback angle from Krivics, with respect to distance travelled. """
        
        distance_travelled = self.distance_travelled[self.learning_time_step]

        plt.plot(distance_travelled, self.fb, color='tab:blue', label = f"Feedback")
        plt.plot(distance_travelled, self.ff, color='tab:orange', label = f"Feedforward")

        # plt.hlines([np.pi, -np.pi], [0, 0], [distance_travelled[-1], distance_travelled[-1]], color='black', linestyles=['--', '--'])
        plt.legend()
        plt.xlabel(f"Distance Travelled [m]")
        plt.ylabel(f"Angle [Radians]")
        plt.title("Feedback and Feedforward terms. Distance Travelled")

        plt.show()
        # plt.savefig(f"{self.folder_path}/Feedforward v. feedback.png")
        # plt.close('all')

    def find_ideal_path(self):
        path_length = 0 
        x = [self.x_0]
        y = [self.y_0]
        for segment in self.segments:
            goal_x = segment.iloc[0]['Goal_x']
            goal_y = segment.iloc[0]['Goal_y']

            vector = np.array([goal_x - x[-1], goal_y - y[-1]])
            vector_size = np.linalg.norm(vector, 2)
            
            scaling = (vector_size-self.threshold)/vector_size
            x.append(x[-1] + vector[0]*scaling)
            y.append(y[-1] + vector[1]*scaling)

            path_length += vector_size-self.threshold
        
        self.ideal_path_length = path_length
        self.ideal_path_x = normalise_array_to_reference_value(x, self.x_0)
        self.ideal_path_y = normalise_array_to_reference_value(y, self.y_0)

    def get_error_from_path(self, error_window_length):
        
        errors = []
        integrated_errors = []
        start_points = []

        full_goal_x = np.array(self.time_step_df['Goal_x'])
        full_goal_y = np.array(self.time_step_df['Goal_y'])

        full_goal_x = normalise_array_to_reference_value(full_goal_x, self.x_0)
        full_goal_y = normalise_array_to_reference_value(full_goal_y, self.y_0)
        distance_travelled = error_window_length + 1
        for i in range(self.x.size):
            # If we're greater than the window length we reset the window and define a new line
            # Alternatively, we hit a new goal point.
            if distance_travelled > error_window_length or np.isin(i, np.array(self.unique_goal_timesteps)):
                
                distance_travelled = 0

                start_x = self.x[i]
                start_y = self.y[i]

                line = LineString([(full_goal_x[i], full_goal_y[i]), (start_x, start_y)])
                start_points.append([start_x, start_y, i])
                errors.append([0])
                integrated_errors.append([0])

            # We don't evaluate the error for the first case, as it'll always be 0, and no movement has occured yet. 
            else:
                # Add the errors
                point = Point(self.x[i],self.y[i])
                error = point.distance(line)
                errors[-1].append(error)

                # track the distance travelled
                distance = np.sqrt((self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2)
                distance_travelled += distance

                # integrate the error over dx
                integrated_errors[-1].append(error*distance)

        self.errors = errors
        self.integrated_errors = integrated_errors
        self.error_window_start_points = start_points
        self.error_window_length = error_window_length

    def plot_error_graph(self):
        cumulative_error = self.get_cumulative_errors()
        # We need to strip an error value since the distance array is 1 smaller
        # The first error is always 0 by definition so we can strip it without loss of generality 
        plt.plot(self.distance_travelled, cumulative_error[1:])
        plt.show()

    def get_cumulative_errors(self):
        cumulative_error = []
        for error in self.integrated_errors:
            for i in range(len(error)):
                cumulative_error.append(error[i])

        cumulative_error = np.cumsum(np.array(cumulative_error))

        return cumulative_error
    
    def plot_example_error_graph(self):
        self.get_error_from_path(0.05)
        start_points = self.error_window_start_points

        x_coords = [point[0] for point in start_points]
        y_coords = [point[1] for point in start_points]

        full_goal_x = np.array(self.time_step_df['Goal_x'])
        full_goal_y = np.array(self.time_step_df['Goal_y'])

        full_goal_x = normalise_array_to_reference_value(full_goal_x, self.x_0)
        full_goal_y = normalise_array_to_reference_value(full_goal_y, self.y_0)
        plt.rcParams.update({'font.size': 12})
        # Creating the scatter plot
        fig, ax = plt.subplots()
        ax.scatter(x_coords, y_coords,s=20, marker = "x", color='tab:blue', label="Initial Points", linewidths=1.5)
        for point in start_points:
            if point == start_points[0]:
                ax.plot([point[0], full_goal_x[point[2]]], [point[1],full_goal_y[point[2]]], color ='tab:orange', alpha=0.75, linewidth = 2, label="Error Lines")
            ax.plot([point[0], full_goal_x[point[2]]], [point[1],full_goal_y[point[2]]], color ='tab:orange', alpha=0.75, linewidth = 2)
        
        theta = np.linspace(-np.pi, np.pi, 360)
        threshold = self.threshold
        for i in range(len(self.goal_x)):
            ax.plot(self.goal_x[i]+threshold*np.cos(theta), self.goal_y[i]+threshold*np.sin(theta), color= 'black', linewidth = 1, linestyle=":")
        ax.scatter(self.goal_x, self.goal_y, marker = "x", color = "black", label = f"Goal Point(s)", linewidths=1, s =20 )
        

        y_lim = 0.45
        x_lim = 0.17
        ax.set_xlim((-0.1, 0.15))
        ax.set_ylim((-0.05, y_lim))
        ax.set_xlabel("x_position (m)")
        ax.set_ylabel("y_position (m)")
        ax.grid()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        ax.set_aspect('equal')
        # ax.set_title(f"$e_p$ Measurement Lines $\\Delta d$ = {self.error_window_length}m ")
        # plt.show()
        fig.savefig(f"Graph_Files\Trajectories\example_path_error.png", bbox_inches='tight')
        plt.close('all')

    def plot_example_ideal_path(self):
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots()
        threshold = self.threshold
        theta = np.linspace(-np.pi, np.pi, 360)

        for i in range(len(self.goal_x)):
            ax.plot(self.goal_x[i]+threshold*np.cos(theta), self.goal_y[i]+threshold*np.sin(theta), color= 'black', linewidth = 1, linestyle=":")
        ax.scatter(self.goal_x, self.goal_y, marker = "x", color = "black", label = f"Goal Point(s)", linewidths=1.2, s =20 )
        # ax.scatter(self.goal_x, self.goal_y, marker = "x", color = "black", label = f"Goal Point(s)", linewidths=0.75, s =7 )

        U = np.diff(self.ideal_path_x)
        V = np.diff(self.ideal_path_y)
        X_start = self.ideal_path_x[:-1]
        Y_start = self.ideal_path_y[:-1]

        ax.quiver(X_start, Y_start, U, V, angles='xy', scale_units='xy', scale=1, color='black', label="Ideal Path", width=0.02)

        # ax.plot(self.ideal_path_x, self.ideal_path_y, color= 'black', linewidth = 1, label="Ideal Path")

        y_lim = 0.5
        x_lim = 0.2
        ax.set_xlim((-0.1, x_lim))
        ax.set_ylim((-0.1, y_lim))
        ax.set_xlabel("x_position (m)")
        ax.set_ylabel("y_position (m)")
        ax.grid()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        ax.set_aspect('equal')
        # ax.set_title(f"$e_p$ Measurement Lines $\\Delta d$ = {self.error_window_length}m ")
        # plt.show()
        fig.savefig(f"Graph_Files\Trajectories\Ideal_path.png", bbox_inches='tight')
        plt.close('all')
        # plt.show()

def select_folder(initialdir="G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests"):
    """Opens a file explorer window"""
    tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing

    folder_path = filedialog.askdirectory(initialdir=initialdir)
    # folder_path = "Trial 1 Basic Learning\circle\Dipole\\2024_03_14 12_35_18_978959"
    return folder_path


def select_file(initialdir="G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests"):
    tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing

    file_path = filedialog.askopenfilename(initialdir=initialdir)
    # folder_path = "Trial 1 Basic Learning\circle\Dipole\\2024_03_14 12_35_18_978959"
    return file_path
def test_run_file_function():
    """Designed to allow for testing of functions meant to run on csvs."""

    # folder_path = "Trial 1 Basic Learning\circle\Dipole\\2024_03_14 12_35_18_978959"
    # folder_path = "Trial 4 Trajectories\circle\circle\\no_ori\\2024_05_01 17_17_23_000797"
    folder_path = select_folder()
    run_test = run(folder_path)
    # run_test.make_trajectory_graph()
    run_test.plot_example_ideal_path()
    # run_test.plot_distance_v_time()
    # run_test.plot_feedforward_feedback()
    # print(run_test.gammas)
    # run_test.plot_example_error_graph()
    # run_test.plot_gammas()
    # time_step_df, run_df = pull_dataframes(folder_path)
    # learning_df, pathstep_df = pull_extra_dataframes(folder_path)
    # goal_x, goal_y, poss = get_goal_point_array(time_step_df)
    # # plot_feedforward_feedback(time_step_df, learning_df, folder_path)
    # plot_individual_trajectory(time_step_df, run_df, folder_path, goal_x, goal_y)

def normalise_vector(vector):
    """Normalises a vector (assumes np already)"""
    return vector/np.linalg.norm(vector)

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

def compare_files(n):

    title = "real_world circle Learning Comparison mouse"
    names = ["No Orientation Run 1", "No Orientation Run 3", "Orientation Run 1", "Orientation Run 3"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]

    # title = "Speed All 3 Comparison 0_04 quarter_circle"
    # names = ["Dipole Run 10", "No Orientation Run 10", "Orientation Run 10"]
    # colors = ["tab:green", "tab:red", "tab:blue"]

    # title = "real_world circle Learning Comparison book"
    # names = ["No Orientation Run 1", "Orientation Run 1"]
    # colors = ["tab:red", "tab:blue"]
    
    if len(names) != n or len(colors)!= n:
        raise ValueError("You need more names/colors for this graph")
    
    parent_file = select_folder()

    runs = []
    run_names = []
    for i in range(n):
        print(i)
        folder_path = select_folder(initialdir=parent_file)
        run_names.append(folder_path)
        runs.append(run(folder_path))


    graph_df = pd.DataFrame({"run_files": run_names,
                             'plot_names' : names,
                             "colours": colors})
    
    graph_df.to_csv(f"G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Files\{title}.csv", sep=',', index=False, encoding='utf-8')
    
    plot_comparison_graphs(runs, names, title, colors)


def plot_comparison_graphs(runs, names, title, colors):
    t0 = time.time()
    plot_error_graph(runs, names, title, colors)
    t1 = time.time()
    plot_gammas(runs, names, title, colors)
    t2 = time.time()
    if "Trajectories Circle" in title or "real_world circle" in title:
        plot_distance_along_goal_v_time(runs, names, title, colors)
    else:
        plot_distance_v_time(runs, names, title, colors)
    t3 = time.time()
    plot_contact_v_distance(runs, names, title, colors)
    t4 = time.time()
    # plot_orientation_v_distance(runs, names, title, colors)
    
    plot_trajectories(runs, names, title, colors)
    t5 = time.time()
    # plot_current_v_original_distribution(runs, names, title, colors)
    make_table(runs, names, title, colors)
    t6 = time.time()
    times = np.array([t0,t1,t2,t3,t4,t5,t6])
    print(np.diff(times))
    # create_legend_figure(names, title, colors)

def plot_trajectories(runs, names, title, colors):
    """Creates multiple trajectory graphs at the same time"""
    plt.rcParams.update({"font.size": 14})
    run = runs[0]
    j = 0
    fig, ax = plt.subplots()
    ax.plot(run.ideal_path_x, run.ideal_path_y, color= 'black', linewidth = 1.5, label="Ideal Path")

    threshold = run.threshold
    theta = np.linspace(-np.pi, np.pi, 360)

    for i in range(len(run.goal_x)):
        ax.plot(run.goal_x[i]+threshold*np.cos(theta), run.goal_y[i]+threshold*np.sin(theta), color= 'black', linewidth = 1.5, linestyle=":")
    ax.scatter(run.goal_x, run.goal_y, marker = "x", color = "black", label = f"Goal Point(s)", linewidths=1.5, s =20 )
    
    y_lim = 0.7
    x_lim = 0.2

    if "Precision" in title:
        y_lim = 0.5
        x_lim = 0.18

    elif "book" in title:
        y_lim = 0.8
        x_lim = 0.2

    elif "Trajectories Circle" in title or "real_world circle" in title:
        y_lim = 0.5
        x_lim = 0.3
    
    elif "Trajectories Rectangle" in title:
        y_lim = 0.5
        x_lim = 0.3

    elif "Trajectories t_and_b" in title:
        y_lim = 0.5
        x_lim = 0.3

    # if run.t[-1] > 29.95:
    #     ax.spines['bottom'].set_color('red')
    #     ax.spines['left'].set_color('red')

    ax.set_xlim((-x_lim, x_lim))
    ax.set_ylim((-0.15, y_lim))
    ax.set_xlabel("x_position [m]")
    ax.set_ylabel("y_position [m]")
    ax.grid()
    # ax.legend()

    # ax.set_title(f"Method = {run.run_df['implementation'][0]}, Time = {round(run.run_df['Run_time'][0], 3)}s, Step Size = {run.run_df['Step_size'][0]}")
    ax.set_aspect('equal')
    plt.tight_layout()

    for j, run in enumerate(runs):
        robo, = ax.plot(run.robo_x, run.robo_y,  color =robo_colour, label = 'Robot')
        target, = ax.plot(run.x, run.y, color = colors[j], label = 'Target')
        plt.draw()
        fig.savefig(f"Graph_Files\Trajectories\{title}_{names[j]}_trajectory.png", bbox_inches='tight')
        robo.remove()
        target.remove()

    plt.close('all')
    plt.rcParams.update({"font.size": font_size})
        # plt.show()


def plot_error_graph(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i,  run in enumerate(runs):
        if "Precision" in title:
                run.get_error_from_path(0.001)
        cumulative_error = run.get_cumulative_errors()
    # We need to strip an error value since the distance array is 1 smaller
    # The first error is always 0 by definition so we can strip it without loss of generality 
        ax.plot(run.distance_travelled, cumulative_error[1:], label=names[i], color = colors[i])
    # ax.legend()
    ax.set_xlabel("Distance Travelled [m]")
    ax.set_ylabel("Cumulative $e_p$ [$m^2$]")
    # ax.set_title(f"$\\Delta d$ = {runs[0].error_window_length}m")
    # if "Precision" in title:
    #     ax.set_xlim((run.threshold, ax.get_xlim()[1]))
    ax.grid()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1]))
    ax.vlines([run.ideal_path_length], ax.get_ylim()[0], ax.get_ylim()[1], color = "black", linestyles="--", linewidth=3)
    # ax.set_aspect('equal')
    fig.savefig(f"Graph_Files/Graphs/{title}_path_errors.png", bbox_inches='tight')
    plt.close('all')
    # plt.show()

def plot_gammas(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i,  run in enumerate(runs):
        distance_travelled = run.distance_travelled
        ax.scatter(distance_travelled, run.gammas[:-1], label=names[i], color = colors[i],s =15)
    # ax.legend()
    ax.set_xlabel("Distance travelled [m]")
    ax.grid()
    ax.set_ylabel("$\\gamma$ [rad]")

    # Get current x and y limits

    ylim = plt.gca().get_ylim()
    ymax = max(abs(ylim[0]), abs(ylim[1]))
    
    j = 10
    while ymax // (np.pi/j) and j > 1:
        j -= 1

    ticks = np.arange(-np.pi/j, 3*np.pi/(2*j), np.pi/(2*j))
    tick_labels = [
    r'$-\frac{\pi}{' + str(j) + '}$',
    r'$-\frac{\pi}{' + str(2*j) + '}$',
    r'$0$',
    r'$\frac{\pi}{' + str(2*j) + '}$',
    r'$\frac{\pi}{' + str(j) + '}$'
    ]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    plt.gca().set_ylim(-np.pi/j, np.pi/j)

    ax.vlines([run.ideal_path_length], -np.pi/j, np.pi/j, color = "black", linestyles="--", linewidth=3)
    fig.savefig(f"Graph_Files/Graphs/{title}_gammas.png", bbox_inches='tight')
    plt.close('all')
    # plt.show()

def plot_distance_v_time(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i,  run in enumerate(runs):
        ax.plot(run.t, run.distance_to_goal, label = names[i], color = colors[i])
    ax.grid()
    # ax.set_title(f"Method = {run.run_df['implementation'][0]}, Time = {round(run.run_df['Run_time'][0], 3)}s, Step Size = {run.run_df['Step_size'][0]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to Goal [m]")
    # ax.legend()
    # Restricting the y axis for the precision data to see a tighter range.
    if "Precision" in title:
        y_lim = 0.03
        ax.set_ylim((0, y_lim))
        ax.hlines(run.threshold, ax.get_xlim()[0], ax.get_xlim()[1], color = "black", linestyles=":", linewidth=2)
    
    fig.savefig(f"Graph_Files/Graphs/{title}_Distance_v_Time.png", bbox_inches='tight')
    plt.close('all')

def plot_distance_along_goal_v_time(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i,  run in enumerate(runs):   
        run.find_distance_projected_onto_goal_vector()
        ax.plot(run.t, run.distance_along_goal_vector, label = names[i], color = colors[i])
        
    ax.grid()
    # ax.set_title(f"Method = {run.run_df['implementation'][0]}, Time = {round(run.run_df['Run_time'][0], 3)}s, Step Size = {run.run_df['Step_size'][0]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance along Goal Vector [m]")
    # ax.legend()
    # Restricting the y axis for the precision data to see a tighter range.
    if "Precision" in title:
        y_lim = 0.03
        ax.set_ylim((0, y_lim))
        ax.hlines(run.threshold, ax.get_xlim()[0], ax.get_xlim()[1], color = "black", linestyles=":", linewidth=2)
    
    fig.savefig(f"Graph_Files/Graphs/{title}_Distance_v_Time.png", bbox_inches='tight')
    plt.close('all')

def plot_orientation_v_distance(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i,  run in enumerate(runs):     
        ax.plot(run.distance_travelled, run.target_orientation[1:], label = names[i], color = colors[i])
    ax.grid()
    ax.set_xlabel("Distance Travelled [m]")
    ax.set_ylabel("$\\theta_0$ [rad]")
    ax.vlines([run.ideal_path_length], ax.get_ylim()[0], ax.get_ylim()[1], color = "black", linestyles="--", linewidth=3)
    # ax.legend()
    
    fig.savefig(f"Graph_Files/Graphs/{title}_Theta_0.png", bbox_inches='tight')
    plt.close('all')

def plot_contact_v_distance(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i,  run in enumerate(runs):     
        ax.scatter(run.distance_travelled, run.contact_orientation[1:], label = names[i], color = colors[i], s= 15)
    ax.grid()
    ax.set_xlabel("Distance Travelled [m]")
    ax.set_ylabel("$\\theta_r$ [rad]")

    ticks = np.arange(-np.pi, np.pi + np.pi/2, np.pi/2)
    tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    ax.vlines([run.ideal_path_length], -np.pi-0.1, np.pi+0.1, color = "black", linestyles="--", linewidth=3)
    # ax.legend()
    ax.set_ylim((-np.pi-0.1, np.pi+0.1))
    
    fig.savefig(f"Graph_Files/Graphs/{title}_Theta_r.png", bbox_inches='tight')
    plt.close('all')

def create_legend_figure(names, title, colors):
    fig, ax = plt.subplots()
    for i in range(len(names)):
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[i], color=colors[i])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Robot", color=robo_colour)
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure(figsize=(12, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(labels))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/{title}_legend.png", bbox_inches='tight')

    # Show the plot
    plt.close('all')

def make_legends(update=True):
    plt.rcParams.update({"font.size": 14})
    create_trajectory_legend_figure_1()
    create_trajectory_legend_figure_2()
    create_trajectory_legend_figure_3()
    create_trajectory_legend_figure_4()
    create_trajectory_legend_figure_5()
    create_trajectory_legend_figure_6()
    create_trajectory_legend_figure_7()
    plt.rcParams.update({"font.size": font_size})
    if update:
        update_report_figures_folder()

def create_trajectory_legend_figure_1():

    names = ["Wide Run 1", "Wide Run 10", "Narrow Run 1", "Narrow Run 10"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Robot", color=robo_colour)
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Ideal Path", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])
    
    
    
    ax.scatter([1, 2, 3, 4], [1, 4, 2, 3], marker = "x", color = "black", label = f"Goal Point", linewidths=2, s =40 )
    
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle=":", label = "Threshold")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(9, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_1.png" )

    # Show the plot
    plt.close('all')

def create_trajectory_legend_figure_2():

    names = ["Wide Run 1", "Wide Run 10", "Narrow Run 1", "Narrow Run 10"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle="--", label = "Ideal Path Length")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Prior Distribution", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(10, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_2.png" )

    # Show the plot
    plt.close('all')

def create_trajectory_legend_figure_3():

    names = ["Wide Run 1", "Wide Run 3", "Narrow Run 1", "Narrow Run 3"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Robot", color=robo_colour)
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Ideal Path", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])
    
    
    
    ax.scatter([1, 2, 3, 4], [1, 4, 2, 3], marker = "x", color = "black", label = f"Goal Point", linewidths=2, s =40 )
    
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle=":", label = "Threshold")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(9, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_3.png" )

    # Show the plot
    plt.close('all')

def create_trajectory_legend_figure_4():

    names = ["Wide Run 1", "Wide Run 3", "Narrow Run 1", "Narrow Run 3"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle=":", label = "Threshold")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle="--", label = "Ideal Path Length")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Prior Distribution", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(10, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_4.png" )

    # Show the plot
    plt.close('all')

def create_trajectory_legend_figure_5():

    names = ["Wide Run 1", "Wide Run 3/6", "Narrow Run 1", "Narrow Run 3/6"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Robot", color=robo_colour)
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Ideal Path", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])
    ax.scatter([1, 2, 3, 4], [1, 4, 2, 3], marker = "x", color = "black", label = f"Goal Point", linewidths=2, s =40 )
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle=":", label = "Threshold")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(9, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_5.png" )

    # Show the plot
    plt.close('all')

def create_trajectory_legend_figure_6():

    names = ["Wide Run 1", "Wide Run 3/6", "Narrow Run 1", "Narrow Run 3/6"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle="--", label = "Ideal Path Length")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Prior Distribution", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(10, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_6.png" )

    # Show the plot
    plt.close('all')

def create_trajectory_legend_figure_7():

    names = ["Wide Run 1", "Wide Run 3", "Narrow Run 1", "Narrow Run 3"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    fig, ax = plt.subplots()

    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[0], color=colors[0])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], color= 'black', linewidth = 2, linestyle="--", label = "Ideal Path Length")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[1], color=colors[1])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Prior Distribution", color="black")
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[2], color=colors[2])
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label=names[3], color=colors[3])
    
    legend = ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure for the legend
    fig_legend = plt.figure  (figsize=(10, 1))  # Adjust the size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', ncol=len(names))
    ax_legend.axis('off') 

    fig_legend.savefig(f"Graph_Files/Graphs/trajectory_legend_7.png" )

    # Show the plot
    plt.close('all')

def alpha_given_mu_kappa(alpha, mu, kappa):
    p = (np.exp(kappa*np.cos(alpha-mu)))/(2*np.pi*i0(kappa))
    return p

def plot_distribution(axes, mu, kappa, label, line_color):
    alphas = np.linspace(-np.pi, np.pi, 1000)
    probability = alpha_given_mu_kappa(alphas, mu, kappa)
    # print(p_int)
    probability /= np.max(probability)
    axes.plot(alphas, probability, label = label, color = line_color)

def plot_current_v_original_distribution(runs, names, title, colors):
    fig, ax = plt.subplots()
    for i, run in enumerate(runs):
        if run.method == "krivic":
            plot_distribution(ax, run.mus[-1], run.kappas[-1], names[i], colors[i])
    plot_distribution(ax, 0, 1, 'Original Distribution', "black")
    # ax.legend()
    ax.grid()
    ax.set_xlabel('$\\gamma$ [rad]')
    ax.set_ylabel('$\\psi_d')
    ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    fig.savefig(f"Graph_Files/Graphs/{title}_learning_distribution.png", bbox_inches='tight')
    plt.close('all')

def plot_example_distribution():
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 10})
    plot_distribution(ax, 0, 1, '$\\mu = 0, \\kappa = 1$', "black")
    ax.vlines(np.pi/3, 0, alpha_given_mu_kappa(np.pi/3, 0, 1)/alpha_given_mu_kappa(0, 0, 1), linewidth=2, linestyles='--', color='tab:blue', label ="Sample" )
    ax.hlines(alpha_given_mu_kappa(np.pi/3, 0, 1)/alpha_given_mu_kappa(0, 0, 1),-np.pi, np.pi/3 , linewidth=2,linestyles='--', color='tab:blue' )
    ax.grid()
    ax.set_xlabel('$\\gamma$ [rad]')
    ax.set_ylabel('$\\psi_d$')
    ax.set_ylim((0, 1.1))
    ax.set_xlim(-np.pi, np.pi)

    ticks = np.arange(-np.pi, np.pi + np.pi/3, np.pi/3)
    tick_labels = [r'$-\pi$', r'$-\frac{2\pi}{3}$', r'$-\frac{\pi}{3}$', r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc="upper right")
    # ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    plt.show()
    fig.savefig(f"Graph_Files/Graphs/example_distribution.png", bbox_inches='tight')
    plt.close('all')
    plt.rcParams.update({'font.size': font_size})

def normalise_array_to_reference_value(input_array, val_0):
    return input_array-val_0

def circular_normalise_array_to_reference_value(input_array, val_0):
    flat_normalised = input_array-val_0
    # shifts up by pi, modulos to within the range, and then shifts back down
    circle_normalised = (flat_normalised + np.pi) % (2 * np.pi) - np.pi
    return circle_normalised

def list_directory_tree_with_os_walk(starting_directory):
    for root, directories, files in os.walk(starting_directory):
        print(f"Directory: {root}")
        for file in files:
            print(f"  File: {file}")

def plot_graph_from_saved_file(folder_path="", update=False ):
    if folder_path == "":
        folder_path = select_file(initialdir="Graph_Files/Files")

    title = folder_path[88:-4]
    print(title)
    # folder_path = "Graph_Files\Speed Dipole 0_16 Circle.csv"
    graph_df = pd.read_csv(folder_path)

    run_names = np.array(graph_df['run_files'])
    pickle_files = os.listdir("Graph_Files\Pickle Files")

    runs = []
    t0 = time.time()
    for run_name in run_names:
        string = f"{run_name[-26:]}.pkl"
        if  string in pickle_files:
            with open(f"Graph_Files\Pickle Files\{string}", 'rb') as file:
                ith_run = pickle.load(file)
        else:

            ith_run = run(run_name)
            ith_run.pickle_run(string)

        runs.append(ith_run)
    t1 = time.time()
    print(f"Run Making:{t1-t0}")

    names = np.array(graph_df['plot_names'])

    if "Speed" in title:
        numbers = [1, 10, 1, 10]
    elif "Precision" in title or "real_world circle" in title:
        numbers = [1, 3, 1, 3]
    elif "real_world straight" in title:
        numbers = [1,6,1,6]
    
    for i in range(2):
        names[i] = f"No Orientation Run {numbers[i]}"
    for i in range(2, 4):
        names[i] = f"Orientation Run {numbers[i]}"
    colors = np.array(graph_df['colours'])


    plot_comparison_graphs(runs, names, title, colors)

    graph_df = pd.DataFrame({"run_files": run_names,
                             'plot_names' : names,
                             "colours": colors})

    graph_df.to_csv(f"G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Files\{title}.csv", sep=',', index=False, encoding='utf-8')

    if update:
        update_report_figures_folder()

def make_table(runs, names_2, title, colors):
    names = names_2.copy()
    mus = []
    kappas = []
    learning_n = []
    for i, run in enumerate(runs):
        mus.append(round(run.mus[-1], 2))
        kappas.append(round(run.kappas[-1],2))
        learning_n.append(len(run.alphas))
        if names[i][0] =="N":
            names[i] = f"Wide {names[i][-1]}"
        else:
            names[i] = f"Narrow {names[i][-1]}"
        if names[i][-1] == "0":
            names[i] = f"{names[i][:-2]} 10"

    table_df = pd.DataFrame({"Run": names,
                             "$n_{A}$" : learning_n,
                             "$\\mu_T$ [rad]": mus,
                             "$\\kappa_T$": kappas})
    table_df.to_csv(f"G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Tables\{title}.csv", sep=',', index=False, encoding='utf-8')

def update_all_graphs():
    folder_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Files"
    sub_files = os.listdir(folder_path)
    n = len(sub_files)
    for i,sub_file in enumerate(sub_files):
        if "All 3" in sub_file or "Trajectories" in sub_file or "real_world circle Learning Comparison book" in sub_file:
            continue
        print(f"{i}/{n}")
        plot_graph_from_saved_file(f"{folder_path}/{sub_file}")
    
    make_legends(update=False)
    update_report_figures_folder()

def update_report_figures_folder():
    speed_path ="G:\My Drive\\004 Fourth Year\Fourth Year Project\Report Figures\Speed"
    graph_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Graphs"
    traj_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests\Graph_Files\Trajectories"
    precision_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\Report Figures\Precision"
    real_world_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\Report Figures\\real_world"
    method_path = "G:\My Drive\\004 Fourth Year\Fourth Year Project\Report Figures\Method"
    graphs_subfiles = os.listdir(graph_path)
    trajectory_subfiles = os.listdir(traj_path)

    def update_file(results_path):
        results_subfiles = os.listdir(results_path)
        for sub_file in results_subfiles:
            if sub_file in graphs_subfiles:
                shutil.copyfile(f"{graph_path}/{sub_file}", f"{results_path}/{sub_file}")
            elif sub_file in trajectory_subfiles:
                shutil.copyfile(f"{traj_path}/{sub_file}", f"{results_path}/{sub_file}")

    update_file(speed_path)
    update_file(precision_path)
    update_file(real_world_path)
    update_file(method_path)


if __name__ == "__main__":
    pass
    # comparisions()
    # graphs_indivual_run()
    # get_group_stats()
    # test_run_file_function()
    # compare_files(4)
    # plot_graph_from_saved_file(update=True)
    # create_trajectory_legend_figure()
    update_all_graphs()
    # plot_example_distribution()
    # make_legends()
    # update_report_figures_folder()
    # recurse_group_stats()
    # list_directory_tree_with_os_walk("G:\My Drive\\004 Fourth Year\Fourth Year Project\\003 Tests\Final Tests")
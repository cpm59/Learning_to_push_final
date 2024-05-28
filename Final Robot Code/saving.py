import numpy as np
import physical_classes
import supporting_functions
import controllers
import pandas as pd
import os
import time



class data_tracker:

    def __init__(self):
        self.time_step = []

        self.path_step = []

        self.learning = []

        self.run = []
        
        self.learned_data = []
             

    def update_learning_file(self, system, kro):
        self.learning.append({'time_step' : system.time,
                                'psi': kro.psi_s[-1][0],
                                'psi_feedback': kro.psi_s[-1][1],
                                'psi_feedforward': kro.psi_s[-1][2],
                                'error': kro.thetas[-1],
                                'orientation':system.orientations_robot_target[-1],
                                'step_size': system.step_size,
                                'delta': kro.delta,
                                'mu': kro.mu,
                                'kappa':kro.kappa})
        
    def update_path_file(self, system):
        self.path_step.append({'time_step' : system.time,
                                'path_point_x' : system.robot.path_point[-1][0],
                                'path_point_y':system.robot.path_point[-1][1],
                                'step_size': system.step_size,
                                })
        

    def update_run_file(self, run_time, step_size, threshold, implementation):
        row = {'Run_time': run_time,
                'Step_size': step_size,
                'threshold': threshold,
                'implementation': implementation}
        self.run.append(row)
        
    def update_time_step_file(self, system):
        self.time_step.append({'time_step' : system.time,
                               'time' : time.time() - system.start_time,
                          'Robot_x': system.robot.trajectory[-1][0],
                          'Robot_y' : system.robot.trajectory[-1][1],
                          'Target_x': system.target.trajectory[-1][0],
                          'Target_y': system.target.trajectory[-1][1],
                          'Target_theta': system.target.orientation[-1],
                          'robot_target_orientation': system.orientations_robot_target[-1],
                          'Distance to path': system.robot.distance_to_path_point,
                          'Goal_x': system.goal_point[-1][0],
                          'Goal_y': system.goal_point[-1][1],
                          'Distance_to_goal': system.distance_to_goal_point,
                          'gamma': system.gammas[-1]})
    
    def save_learned_data(self, system, kro):
        if system.method == "krivic":
            self.learned_data = {'alphas':kro.alphas,
                                 'orientations':kro.alpha_orientations}

        
    def save_files(self, name):
        """Saves the files collected"""
        path = f"Run_files/{name}"
        os.mkdir(path)
        
        self.run = pd.DataFrame(self.run)
        self.time_step = pd.DataFrame(self.time_step)
        self.path_step = pd.DataFrame(self.path_step)
        self.learning = pd.DataFrame(self.learning)
        self. learned_data=pd.DataFrame(self.learned_data)
        
        self.run.to_csv(f"{path}/run.csv", sep=',', index=False, encoding='utf-8')
        self.time_step.to_csv(f"{path}/timestep.csv", sep=',', index=False, encoding='utf-8')
        self.path_step.to_csv(f"{path}/pathstep.csv", sep=',', index=False, encoding='utf-8')
        self.learning.to_csv(f"{path}/learning.csv", sep=',', index=False, encoding='utf-8')
        self.learned_data.to_csv(f"{path}/learned_data.csv", sep=',', index=False, encoding='utf-8')

if __name__ == "__main__":
    dt = data_tracker()
    dt.update_run_file(1,1,1,"krivic")
    dt.save_files("test1")


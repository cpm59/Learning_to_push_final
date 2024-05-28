#!/usr/bin/env python
"""Runs the mujoco sims"""
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mjc
import mujoco.viewer
from scipy.special import i0
from scipy.optimize import minimize

import physical_classes
import controllers

class mujoco_tracker:

    def __init__(self):
        """Tracks any mujoco-specific stuff."""
        self.velocity_error = np.array([0.,0.])


def initialse_mujoco_model(system):
    """Starts the mujoco model. Returns the model and data objects. """
    robot_x = system.robot.trajectory[0][0]
    robot_y = system.robot.trajectory[0][1]
   
    xml = f"""
    <mujoco>

        <compiler angle="radian" meshdir="meshes"/>
 
        <!-- import our stl files -->
        <asset>
            <mesh file="large_convex.stl" />
        </asset>
 
        <worldbody>
        
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>  
        <geom type="plane" size="1 1 0.1" rgba=".9 .1 .1 1"/>
        <body name="robot" pos="{robot_x} {robot_y} 0.02">
            <joint type="free"/>
            <geom  type="cylinder" size="0.1 .02" rgba="0.1 .1 .9 1"/>
        </body>
        

        <body name="target" pos="0 0 0.02">
            <joint type="free"/>
            
            <geom type="cylinder" size="0.1 0.02" pos="0 0 0" rgba="0.1 .6 .1 1"/>
            
        </body>
        </worldbody>

    </mujoco>
    """

    # <body pos="0 0 0.02">
        #     <joint type="free"/>
        #     <geom name="target" type="mesh" mesh ="large_convex" rgba="0.1 .9 .1 1"/>
        # </body>


    # Load the MuJoCo XML model
    model = mjc.MjModel.from_xml_string(xml)
    data = mjc.MjData(model)

    # Starts a mujoco tracking object.
    m_tracker = mujoco_tracker()


    # Update the model with the assigned positions
    mjc.mj_kinematics(model, data)
    target_id = model.body('target').id
    system.target.orientation.append(np.array(np.arcsin(-1*data.geom_xmat[target_id,1]).copy()))

    return model, data, m_tracker

def update_system_data(system, model, data):
    robot_id = model.body('robot').id
    target_id = model.body('target').id

    system.robot.trajectory.append(np.array(data.geom_xpos[robot_id,:2].copy()))
    system.target.trajectory.append(np.array(data.geom_xpos[target_id,:2].copy()))
    system.target.orientation.append(np.array(np.arcsin(-1*data.geom_xmat[target_id,1]).copy()))
    
    system.robot.velocity.append(np.array([data.qvel[((robot_id-1)*6)].copy() ,data.qvel[((robot_id-1)*6)+1].copy()]))
    
    # need to add the orientation capture in here

def match_target_velocity(system, m_tracker,velocity=0.1, ki=80, kp=800, kd=100):
    """Finds the force required to match the target velocity for the robot."""

    
    error = system.current_direction*velocity - system.robot.velocity[-1]

    m_tracker.velocity_error[0] += ki*error[0]
    m_tracker.velocity_error[1] += ki*error[1]
    
    force = kp * error + kd*(-1*system.robot.velocity[-1]) + m_tracker.velocity_error

    return force

def run_mujoco_model(system, method):
    """Runs the model"""

    # mujoco Initialise steps
    model, data, m_tracker = initialse_mujoco_model(system)
    robot_id = model.body('robot').id


    controllers.set_path_point(system, method)
    system.find_distance_to_goal_point()
    system.robot.find_distance_to_path_point()

    # Launches the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # x = input("Wait for launch")
        for i in range(20000):
            
            # updates values
            update_system_data(system, model, data)
            system.find_distance_to_goal_point()
            system.robot.find_distance_to_path_point()
            

            # Checks if the goal has been achieved, or if the path point should be updated
            if system.distance_to_goal_point < 0.025:
                break
            if system.robot.distance_to_path_point < 7.5*system.step_size/10:
                controllers.set_path_point(system, method)
                if system.mode== "debug" and abs(system.kro.thetas[-1]) > 0.9* np.pi/2:
                    system.plot_current_step()
                    system.kro.plot_current_v_original_distribution()

            # finds the force to apply
            force = match_target_velocity(system, m_tracker)

            
            data.xfrc_applied[robot_id][0] = force[0]
            data.xfrc_applied[robot_id][1] = force[1]

            mjc.mj_step(model, data)

            viewer.sync()

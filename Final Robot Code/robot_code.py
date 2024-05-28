#!/usr/bin/env python
"""All functionality for robot control"""

import numpy as np
import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import PoseStamped

import physical_classes
import controllers
import supporting_functions
# rospy is included in ros noetic, and cannot be pip'd

def pose_to_array(pose_stamped):
    """Takes a posestamped object and returns a numpy array"""
    pose = pose_stamped.pose
    array = np.array([pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
    return array

class rospy_runner():

    def __init__(self ):
        """Starts all the ros nodes. The heirarchy has to be nodes > code, so system exists within here"""
        
        self.opti_robot_pose = 0
        self.opti_target_pose = 0
        self.robot_robot_pose = 0
        self.target_orientation = 0
        self.robot_poses = []
        
        rospy.init_node('Colin_code', anonymous=True)

        rospy.Subscriber('/cartesian_impedance_example_controller/robot_current_pose', PoseStamped, self.sub_robot_callback)
        rospy.Subscriber('/mocap_node/Pushing_tool/pose', PoseStamped, self.sub_optitrack_callback)
        rospy.Subscriber('/mocap_node/Pushing_target/pose', PoseStamped, self.sub_optitrack_target_callback)
        
        
        self.publisher = rospy.Publisher('cartesian_impedance_example_controller/equilibrium_pose_new', PoseStamped, queue_size=1000)


    def sub_robot_callback(self, msg):
        robot_pose = pose_to_array(msg)
        self.robot_robot_pose = robot_pose
        time_stamped_array = np.array([robot_pose[0],robot_pose[1],robot_pose[2], msg.header.stamp.secs, msg.header.stamp.nsecs])
        self.robot_poses.append(time_stamped_array)

    def sub_optitrack_callback(self, msg):
        robot_pose = pose_to_array(msg)
        robot_pose = np.array([robot_pose[0],-robot_pose[2],robot_pose[1],robot_pose[3],robot_pose[4],robot_pose[5],robot_pose[6]])
        self.opti_robot_pose = robot_pose
        
    def sub_optitrack_target_callback(self, msg):
        robot_pose = pose_to_array(msg)
        robot_pose = np.array([robot_pose[0],-robot_pose[2],robot_pose[1],robot_pose[3],robot_pose[4],robot_pose[5],robot_pose[6]])
        target_quaternion = robot_pose[3:]

        self.opti_target_pose = robot_pose

    def run(self):
        rospy.spin()

    def publish_pose(self, px, py, pz, ox, oy, oz, ow):
        pose_msg = PoseStamped()

        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "panda_link0"

        pose_msg.pose.position.x = px
        pose_msg.pose.position.y = py
        pose_msg.pose.position.z = pz
        pose_msg.pose.orientation.x = ox
        pose_msg.pose.orientation.y = oy
        pose_msg.pose.orientation.z = oz
        pose_msg.pose.orientation.w = ow

        self.publisher.publish(pose_msg)

        # rospy.loginfo(f"Published Pose:{px}, {py} {pz}")
        






        





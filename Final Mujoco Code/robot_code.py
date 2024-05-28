#!/usr/bin/env python
"""All functionality for robot control"""

import numpy as np
import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import PoseStamped

import physical_classes
import controllers
# rospy is included in ros noetic, and cannot be pip'd


class rospy_runner():

    def __init__(self):
        """Starts all the ros nodes. The heirarchy has to be nodes > code, so system exists within here"""
        self.system = system
        rospy.init_node('Colin_code', anonymous=True)

        rospy.Subscriber('/cartesian_impedance_example_controller/robot_current_pose', PoseStamped, self.sub_robot_callback)
        rospy.Subscriber('/cartesian_impedance_example_controller/robot_current_pose', PoseStamped, self.sub_optitrack_callback)

        self.publisher = rospy.Publisher('cartesian_impedance_example_controller/equilibrium_pose_new', PoseStamped, queue_size=1000)


    def sub_robot_callback(self, msg):


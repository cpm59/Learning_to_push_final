import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def import_data():
    num = 6
    opti = np.genfromtxt(f"Fourth Year Project\\004 Programming\Linux code\opti_test_data_{num}", delimiter=",")
    robot = np.genfromtxt(f"Fourth Year Project\\004 Programming\Linux code\\robot_test_data_{num}", delimiter=",")
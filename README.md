# Learning_to_push_final
The combined repositry for all pushing code

Final Robot code and Mujoco code are similar, but some of the robot code has changes to the most recent mujoco version. 

graph creation (loose file):

Makes most of the graphs for the final report from raw data (available in logbook).

In the robot code, there is:

main: main file for running tests. Runs threading, calibration, robot movement, and the inner movement loop. Calls for input data
robot_code: robot specific code; the rospy listener to optitrack /franka panda, and pose publisher to the franka panda. 
saving: saves the run data to pandas dataframes and csv files. 

Mujoco code:

main:  runs outer control loops. Designed to operate on robot and simulation, but this fell apart when robot work started picking up steam
mujoco_code: mujoco specific code. Intialises sim environment and objects, and reads data from the sim interface. Velocity integral controller in here as well. 

Shared code:

controllers.py; has all the pushing controllers. Final robot code has a few changes for operating on the robot but the maths should be the same
supporting_functions.py: Lots of little supporting functions that come in handy a lot.
physical_classes: object, robot and system classes for routine operations, and storing data.

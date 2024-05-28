import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gather_data():
    folders =  os.listdir("Run_files")
    folders.sort()

    for folder in folders:
        path = f"Run_files/{str(folder)}/Figures"
        if os.path.isdir('Figures'):
            pass
        else:
            os.mkdir(f"Run_files/{str(folder)}/Figures")
            plot_graphs(f"Run_files/{str(folder)}")

def plot_graphs(path):
    robot_x = pd.read_csv()
if __name__ == "__main__":
    gather_data()
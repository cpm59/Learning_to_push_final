#!/usr/bin/env python
"""implements the controller methods."""

import numpy as np
import matplotlib.pyplot as plt
import supporting_functions
from scipy.special import i0
from scipy.optimize import minimize
from scipy.integrate import quad
import time
import pandas as pd

def set_path_point(system, method, rr, step_size=0.075):
    """Takes in the data, and returns the next point for the robot to go to."""
    
    if method == "window":
        # Calls the pushing window method 
        direction_vector = pushing_window(system)
    
    elif method =="dipole":
        # Calls the dipole method
        direction_vector = dipole_direction_vector(system)

    elif method=="krivic":
        kro = system.kro
        direction_vector = kro.direction_vector(system)
        
        
    system.current_direction = supporting_functions.normalise_vector(direction_vector)
    
    # Sets a new goal point for the robot based on the set step size.
    path_point = system.robot.trajectory[-1] + step_size*system.current_direction
    system.robot.path_point.append(path_point)

    # Saves to path file
    system.saver.update_path_file(system)


def set_direction_vector(system, offset=0):
    """Sets the push and relocation vectors"""

     # Finds the push vector and goal vector
    system.find_push_vector()
    system.goal_vector()

    # Finds the relocate vector and checks the sign.
    relocate_vector = supporting_functions.rotate_vector(np.pi/2, system.push_vector)
    if supporting_functions.angle_between_vectors(system.goal_vector_absolute_frame, system.push_vector) < offset:
        relocate_vector *= -1
    
    system.relocate_vector = relocate_vector


def pushing_window(system, angle=5/90):
    """Implements the push vector window. Angle is 2*angle = arc width for pushing, in degrees eg 2/90"""

    # Sets the relocate and push vectors, and the goal vector
    set_direction_vector(system)
    
    # Finds the current heading error, and chooses to relocate or push
    theta_error = supporting_functions.angle_between_vectors(system.goal_vector_absolute_frame, system.push_vector)
    
    if abs(theta_error) < (np.pi/2)*angle:
        direction_vector = system.push_vector
        
    else:
        direction_vector = system.relocate_vector

    
    return direction_vector

def dipole_direction_vector(system):
    """Implements the Dipole pushing method"""

    set_direction_vector(system)
    system.relocate_vector = supporting_functions.rotate_vector(np.pi/2, system.goal_vector_absolute_frame)

    theta = supporting_functions.angle_between_vectors(system.goal_vector_absolute_frame, -1*system.push_vector)

    # direction_vector = system.goal_vector_absolute_frame*np.cos(2*theta)+system.relocate_vector*np.sin(2*theta)

    robot_goal_vector = supporting_functions.normalise_vector(system.goal_point[-1] - system.robot.trajectory[-1])
    phi_dipole = supporting_functions.angle_between_vectors(system.push_vector, robot_goal_vector)
    # alpha = abs(theta/phi_dipole)**2
    alpha = abs(theta/phi_dipole)

    if alpha > 15:
        alpha = 15
    if alpha < 1:
        alpha = 1

    direction_vector = system.goal_vector_absolute_frame*(np.cos(theta)**2-alpha*np.sin(theta)**2) + system.relocate_vector*(1+alpha)*(np.sin(theta)*np.cos(theta))
   
    return direction_vector



class krivics_object():

    def __init__(self):
        self.alpha_p = 0
        self.mu = 0
        self.kappa = 1
        self.alphas = []
        self.thetas= []
        self.integral_theta = []
        self.find_constant()
        self.last_update_index = 0
        self.alpha_orientations = []
        self.psi_s = []
        self.delta = np.pi

    def load_data(self):
        """Loads the most recent learned data file. The idea is this can be daisy chained"""

        most_recent_folder = supporting_functions.get_most_recent_file("Run_files")
        most_recent_learning = f"Run_files/{most_recent_folder}/learned_data.csv" 
        most_recent_learning = pd.read_csv(most_recent_learning, delimiter=",")
        alphas = list(most_recent_learning['alphas'])
        orientations = list(most_recent_learning['orientations'])
        self.alphas = alphas
        self.alpha_orientations = orientations

    def append_alpha(self, alpha):
        self.alphas.append(alpha)
        

    def find_constant(self):
        result, error = quad(self.krivics_integrand, 0, 100)
        self.normalising_constant = result

    def krivics_integrand(self, k):
        return i0(5.84*k)/(i0(k)**7)    

    def alpha_given_mu_kappa(self, alpha, mu, kappa):
        p = (np.exp(kappa*np.cos(alpha-mu)))/(2*np.pi*i0(kappa))
        return p
    
    def plot_distribution(self, axes, mu, kappa, label):
        alphas = np.linspace(-np.pi, np.pi, 1000)
        probability = self.alpha_given_mu_kappa(alphas, mu, kappa)
        axes.plot(alphas, probability, label = label)
    
    def plot_current_v_original_distribution(self):
        fig, ax = plt.subplots()
        self.plot_distribution(ax, self.mu, self.kappa, 'Current Distribution (mu={}, kappa={})'.format(round(self.mu,3), round(self.kappa, 3)))
        self.plot_distribution(ax, 0, 1, 'Original Distribution (mu=0, kappa=1)')
        ax.legend()
        ax.set_xlabel('Mu')
        ax.set_ylabel('Unnormalised Probability')
        ax.set_title('Comparison of Distributions')
        plt.show()

    def p_mu_kappa(self, mu, kappa):
        p = (1/self.normalising_constant) * (np.e**(5.84*kappa*np.cos(-1*mu)))/(2*np.pi*i0(kappa)**7)
        return p

    def likelihood(self, alphas, mu, kappa):
        return sum([np.log(self.alpha_given_mu_kappa(theta, mu, kappa)) for theta in alphas])
    
    def maximise_posterior(self, alphas, initial_mu, initial_kappa):
        def objective_function(params):
            mu, kappa = params
            return -1*self.unnormalised_posterior(alphas, mu, kappa)  # Minimize negative posterior

        # Perform optimization
        initial_guess = [initial_mu, initial_kappa]
        bounds =[(-np.pi, np.pi), (0.001, 50)]
        result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds)

        # Return optimized mu, kappa and optimized likelihood
        return result.x[0], result.x[1], -result.fun
    
    def unnormalised_posterior(self, alphas, mu, kappa):
        prior = self.p_mu_kappa(mu, kappa)
        likelihood = self.likelihood(alphas, mu, kappa)

        posterior = np.log(prior)+likelihood

        return posterior
    
    def set_krivics_vector(self, system):
        # Finds the push vector and goal vector
        system.find_push_vector()
        system.goal_vector()

        alpha = supporting_functions.angle_between_vectors(system.goal_vector_absolute_frame, system.push_vector)
    
        # Finds the relocate vector and checks the sign.
        relocate_vector = supporting_functions.rotate_vector(np.pi/2, system.push_vector)
        system.absolute_push_vector = system.push_vector
        if abs(alpha) > 60*np.pi/180:
            system.push_vector *= -1
        if alpha < self.mu:
            relocate_vector *= -1
    
        system.relocate_vector = relocate_vector
    
    def direction_vector(self, system):
        """ this is called everytime the system asks for a new direciton"""

        self.set_krivics_vector(system)

        orientation_vector = supporting_functions.rotate_vector(system.target.orientation[-1] , np.array([1,0]))
        orientation_robot_target = supporting_functions.angle_between_vectors(system.absolute_push_vector, orientation_vector)
        system.orientations_robot_target.append(orientation_robot_target)

        # Checks that the theta is decreasing
        if len (system.kro.thetas) >=2:
            if abs(system.kro.thetas[-2]) > abs(system.kro.thetas[-1]):

                # Appends alpha and the relevant orientation.
                self.append_alpha(self.thetas[-1])

                
                self.alpha_orientations.append(orientation_robot_target)

                # Choose line according to adaptation method!
                # delta = 180*np.pi/180
                delta = self.find_delta()

                self.delta = delta

                local_alphas = self.orientation_selection( self.alphas, self.alpha_orientations, delta)

                t1 = time.time()
                results = self.maximise_posterior(local_alphas, self.mu, self.kappa)
                t2 = time.time()

                if system.mode == "debug":
                    print(f"\nN = {len(self.alphas)}, Maximisation time = {t2-t1}")
                self.mu = results[0]
                self.kappa = results[1]

        # set_direction_vector(system, self.mu)
        
        theta_error = supporting_functions.angle_between_vectors(system.goal_vector_absolute_frame, system.push_vector)
        self.thetas.append(theta_error)
        self.integral_theta.append(theta_error)

        push_coefficient = self.alpha_given_mu_kappa(theta_error, self.mu, self.kappa)/self.alpha_given_mu_kappa(self.mu, self.mu, self.kappa)
        relocate_coefficient = np.sqrt(1-push_coefficient**2)

        direction_vector_1 = push_coefficient*system.push_vector + relocate_coefficient*system.relocate_vector
        psi_feedforward = supporting_functions.angle_between_vectors(system.push_vector,direction_vector_1 )

        integral_term = np.average(self.integral_theta)
        if  abs(integral_term) > 0.1:
            integral_term  = 0.1*integral_term/abs(integral_term)
        # psi_feedback = 0.0025*integral_term + 0.005*self.thetas[-1]
        psi_feedback = (0.02*integral_term + 0.1*self.thetas[-1])
        # psi_feedback = 0
        # psi_feedforward = 0
        psi = psi_feedforward + psi_feedback

        self.psi_s.append([psi, psi_feedback, psi_feedforward])
        direction_vector = supporting_functions.rotate_vector(psi, system.push_vector)

        # Saves the learning data
        system.saver.update_learning_file(system, self)

        return direction_vector
    
    def find_delta(self):

        deltas = np.linspace(0.1*np.pi/180, 22.5*np.pi/180, 5)
        kappas = []
        for delta in deltas:
            local_alphas = self.orientation_selection(self.alphas, self.alpha_orientations, delta)
            results = self.maximise_liklihood(local_alphas, self.mu, self.kappa)  
            kappa = results[1]
            kappas.append(kappa)
        print(kappas)
        delta = deltas[np.argmax(kappa)]
        return delta
    

    def orientation_selection(self, alphas, orientations, delta):
        """selects the orientations that are relevant to the current push"""
        orientations = np.array(orientations)
        relevant_alphas = []

        for i, orientation in enumerate(orientations):
            
            x = orientations[-1]
            x1 = x-delta
            x2 = x+delta
            if self.is_within_circular_range(orientation,x1,x2) == True:
                relevant_alphas.append(alphas[i])

        return relevant_alphas

    def is_within_circular_range(self, x, x1, x2):
        # Normalize the angles to be within the range -pi to pi
        x = np.mod(x + np.pi, 2 * np.pi) - np.pi
        x1 = np.mod(x1 + np.pi, 2 * np.pi) - np.pi
        x2 = np.mod(x2 + np.pi, 2 * np.pi) - np.pi

        # Check if x lies within the circular range x1 < x < x2
        if x1 < x2:
            return x1 < x < x2
        else:
            return x > x1 or x < x2


# kro = krivics_object()
# alphas = -1*np.ones(200)
# for alpha in alphas:
#     kro.append_alpha(alpha)
# mu, kappa, fun = kro.maximise_posterior(kro.alphas, kro.mu, kro.kappa)
# kro.mu = mu
# kro.kappa = kappa
# kro.plot_current_v_original_distribution()

# O = [0, 1, 2, 3, 4, 5, 6, 3]
# A = [0, 1, 2, 3, 4, 5, 6, 7]
# print(kro.orientation_selection(A, O, 1))
# print("finsihed")







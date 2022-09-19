#!/usr/bin/env python

from __future__ import division
import rospy
import carla
import time
import json
import os
import pkg_resources
from distutils.version import LooseVersion
from actors.weather import Weather
from enviornment import Enviornment
import sys
#from scenario import Scenario
import threading
from pdf import PDF
from controller import Controller
from search_space import SearchSpace
from scenario import Scenario
import numpy as np
import tensorflow as tf
import math
from agents.navigation.controller import VehiclePIDController
import random

class ParameterOptimisation:
    def __init__(self):
        rospy.init_node('ParameterOptimisation node initialiser') 
        try: 
            self.pdf = PDF()
            self.env = Enviornment()

            # Resume related code
            self.resume = True
            self.resume_data = None
            self.resume_settings = rospy.get_param('resume')
            self.path_to_saved_model = self.resume_settings['path_to_saved_model']
            self.path_to_all_models = self.resume_settings['path_to_all_models']
            self.path_to_data_json = self.resume_settings['path_to_data_json']
            self.path_to_multivariate = self.resume_settings['path_to_multivariate']
            ps_data = []
            self.ps_data_index = []
            self.parameter_a = []
            self.parameter_b = []
            self.parameter_c = []
            self.parameter_d = []
            self.parameter_e = []
            self.parameter_f = []
            self.parameter_g = []
            count = 0
            if os.path.isfile(self.path_to_multivariate):
                with open(self.path_to_multivariate) as json_file:
                    all_data = json.load(json_file)
                    self.parameter_a = all_data['a']
                    self.parameter_b = all_data['b']
                    self.parameter_c = all_data['c']
                    self.parameter_d = all_data['d']
                    self.parameter_e = all_data['e']
                    self.parameter_f = all_data['f']
                    self.parameter_g = all_data['g']
                    all_data = all_data['ps']
                    
                    print("Length of all data: {}".format(len(all_data)))
                    for index in range(len(all_data)):
                        # if all_data[index][0] <= 0 or all_data[index][1] <= 0 or all_data[index][2] <= 0 or all_data[index][3] <= 0 or all_data[index][4] <= 0 or all_data[index][5] <= 0 or all_data[index][6] <= 0 or all_data[index][7] <= 0:
                        #     continue
                        # self.ps_data_index.append(count)
                        sub_data = []
                        for sub_index in range(len(all_data[index])):
                            sub_data.append(round(all_data[index][sub_index],2))

                        ps_data.append(sub_data)
                        count += 1

            self.search_space = SearchSpace(ps_data=ps_data)
            exploration = None

            # Loading the json file for resume operation
            if os.path.isfile(self.path_to_data_json):
                with open(self.path_to_data_json) as json_file:
                    self.resume_data = json.load(json_file)
                    exploration = self.resume_data['exploration']
                    if 'episode' not in self.resume_data.keys():
                        self.resume = False
                    else:
                        print("WE ARE RESUMING!!!!!!")
            else:
                self.resume = False
                print("WE ARE NOT RESUMING, BUT WE ARE STARTING A BRAND NEW MODEL!!!!!!")
            
            # Setting the parameter for resume operation
            if self.resume:
                pass
                # Getting pedestrian speed and accelaration from saved data
                #self.pdf.speed_ped_dist = self.resume_data['ped_speed_dist']
                #self.pdf.acceleration_ped_dist = self.resume_data['ped_accel_dist']
                #self.pdf.weather_dist = self.resume_data['weather_dist']

            # Add paprameters to search space
            self.add_params_to_search_space()
            self.scenario = Scenario(client = self.env.client, world = self.env.world, pdf = self.pdf, search_space = self.search_space)
            self.reward_settings = rospy.get_param('rewards')
            
            # Setting the dictionary to store the plotting data
            self.plotting_data = {
                'exploration': [], 
                'loss_per_episode': [], 
                'episode': None, 'reward': [], 
                'no_explored_step_batch': [], 
                'rss_data_per_episode': [],
                'scenario_data_per_episode': [],
                'action_per_episode': []
            }
            self.start_time_episode = time.time()
            print("Start time: {}".format(self.start_time_episode))

            # Controller is a Policy Gradient LSTM Network 
            self.controller = Controller(search_space = self.search_space, resume = self.resume, model_name = self.path_to_saved_model, exploration = exploration)
            self.process()
            
        except:
            e = sys.exc_info()[0]
            print(e)

        rospy.on_shutdown(self.shutdown)   
        rospy.spin()


    def add_params_to_search_space(self):
        
        '''
        self.search_space.add_parameters(name="trigger_dist", values=[8.19, 12.43, 4.04, 3.85, 9.03, 12.00, 8.92, 10.24, 7.50, 8.15])
        self.search_space.add_parameters(name="cut_in_vel", values=[9.84,11.18,4.99,6.69,10.62,3.10,7.86,13.45,11.85,3.71])
        self.search_space.add_parameters(name="start_to_cutin_time", values=[1,2,3,4,5,6,7,8])
        self.search_space.add_parameters(name="cut_end_vel", values=[9.36,13.53,10.27,4.02,9.68,3.35,13.02,4.34,13.14,12.03])
        self.search_space.add_parameters(name="cutin_to_cutend_time", values=[1,2,3,4,5,6,7,8,9])
        self.search_space.add_parameters(name="adv_final_vel", values=[9.69,6.57,8.11, 7.92, 5.32, 8.01, 8.90, 11.32, 10.35, 12.28])
        self.search_space.add_parameters(name="cutend_to_final_time", values=[1,2,3,4,5,6,7,8,9,10,11,12])
        self.search_space.add_parameters(name="ego_init", values=[13.52, 7.26, 8.87, 1.56, 10.5, 8.97, 10.83, 8.95, 10.89, 8.23])
        '''
        
        self.search_space.add_parameters(name="trigger_dist", values=self.parameter_a)
        self.search_space.add_parameters(name="cut_in_vel", values=self.parameter_b)
        self.search_space.add_parameters(name="start_to_cutin_time", values=self.parameter_c)
        self.search_space.add_parameters(name="cut_end_vel", values=self.parameter_d)
        self.search_space.add_parameters(name="cutin_to_cutend_time", values=self.parameter_e)
        self.search_space.add_parameters(name="adv_final_vel", values=self.parameter_f)
        self.search_space.add_parameters(name="cutend_to_final_time", values=self.parameter_g)
        
        #Non challenging scenario at non_challenging_08_08_22 on episode 3 - 16.5_16.5_6.0_13.0_4.0_10.5_4.0




    def process(self):

        self.start_time_scenario = time.time()
        print("Start time: {}".format(self.start_time_scenario))
        con_settings = rospy.get_param('nas_controller')
        pg_settings = rospy.get_param('policy_gradient')
        plot_settings = rospy.get_param('plot')

        # Initilizing the values
        range_start = 0
        data_file_elapsed_time = 0
        exploration = pg_settings['exploration']

        # Setting the parameter for resume operation
        if self.resume:
            range_start = int(self.resume_data['episode'])+1
            data_file_elapsed_time = self.resume_data['elapsed_time']
            exploration = self.resume_data['exploration']

        # Getting the initial random state/action
        state = self.search_space.get_random_parameter()
        
        # Reshaping state/action to the imput shape of the NN
        state = state.reshape(con_settings['batch_size'], con_settings['time_steps'], con_settings['features'])
        state = tf.cast(state, tf.float32)
        print("Initial state: {}".format(state))
        
        for episode in range(range_start, con_settings['epochs']): 
            try:
                # Logic for brake the looping abruptly.
                stop_looping = rospy.get_param('stop_looping')
                if stop_looping:
                    break
                
                start_time = time.time()
                print("\n\n--------------------------Episode: {}---------------------------".format(episode))
                       
                action = self.controller.get_action(state) 
                print("Selected action: {}".format(action))
                skip = False
                reward = 0

                 # Run the action             
                reward, skip, rss_plot_data, scenario_plot_data = self.scenario.generate(actions = action, episode = episode)
                print("Reward is: {}".format(reward))

                self.env.world.tick()
                self.scenario.reset()
                self.env.world.tick()

                # Checking skip set to True 
                if skip is not True:

                    action = tf.keras.backend.get_value(action)

                    # In our sample action is equal state
                    state = action.reshape(con_settings['batch_size'], con_settings['time_steps'], self.search_space.size)

                    # Casting to float
                    state = tf.cast(state, tf.float32)
                    
                    # Save the data
                    self.controller.remember(state, reward)

                else:
                    print("We are skipping this episode as something goes wrong in the scenario generation")

                 # Exploration mechanism
                if episode%pg_settings['update_exploration'] == 0:
                    if episode >= pg_settings['exploration_hard_stop']:
                        self.controller.exploration = 0
                        exploration = 0
                    else:
                        exploration = self.controller.exploration_decay_progress()

                # Save plotting data
                self.plotting_data['exploration'].append(exploration)
                self.plotting_data['episode'] = episode
                self.plotting_data['reward'].append(reward)
                self.plotting_data['rss_data_per_episode'].append(rss_plot_data)
                self.plotting_data['scenario_data_per_episode'].append(scenario_plot_data)
                action = tf.keras.backend.get_value(action)
                action = np.array(tf.cast(action, tf.float32))
                self.plotting_data['action_per_episode'].append("{}_{}_{}_{}_{}_{}_{}".format(str(action[0][0][0].item()), str(action[0][0][1].item()), str(action[0][0][2].item()), str(action[0][0][3].item()), str(action[0][0][4].item()), str(action[0][0][5].item()), str(action[0][0][6].item())))
                  
                if episode%pg_settings['update_every'] == 0 and episode != 0:
                    
                    # Update the policy parameter \theta
                    self.controller.update_policy()

                    print("here1")
                    
                    # Save plotting data
                    self.plotting_data['loss_per_episode'].append(float(self.controller.total_pg_loss))

                    # This determines no. of explored actions in a batch 
                    self.plotting_data['no_explored_step_batch'].append(self.controller.no_explored_step_batch)

                    file_location = plot_settings['path_to_plot_data'].format(episode)
                    with open(file_location, 'w') as file_handle:
                        json.dump(self.plotting_data, file_handle)
                    self.plotting_data = {
                        'exploration': [], 
                        'loss_per_episode': [], 
                        'episode': None, 
                        'reward': [], 
                        'no_explored_step_batch': [], 
                        'rss_data_per_episode': [],
                        'scenario_data_per_episode': [],
                        'action_per_episode': []
                    }

                    # Reset the variables of the controller for next batch
                    self.controller.reset()

                print("here2")

                # Display statistics of each episode
                end_time = time.time()
                episode_time =  end_time - start_time 
                total_time_spend_sofar = (start_time-self.start_time_episode)+data_file_elapsed_time
                print("\n\nTotal elapsed time: {} hr(s)".format(round((total_time_spend_sofar/60)/60),2))
                print("Episode time: {} s".format(episode_time))
                print("Current exploration value: {}".format(float(exploration)))
                
                # Save the model and resume data      
                if episode%self.resume_settings['save_model'] == 0:
                    
                    # Save the model
                    self.controller.save_model(self.path_to_saved_model, self.path_to_all_models.format(episode))
                    
                    # Save necessary data to resume 
                    data = {
                        'model': self.path_to_saved_model,
                        'episode': episode,
                        'elapsed_time': total_time_spend_sofar,
                        'exploration' : exploration
                    }

                    # Save the json file
                    with open(self.path_to_data_json, 'w') as outfile:
                        json.dump(data, outfile)

                print("here3")
                
            except:
                e = sys.exc_info()[0]
                print(e)
                print("Exception occured and skipping this episode")
                print("Resetting...")
                self.scenario.reset()                
                pass
        

    def shutdown(self):
        self.scenario.destroy()
        

if __name__ == '__main__':
    try:
        ParameterOptimisation()
    except rospy.ROSInterruptException:
	    rospy.logerr('Could not start ParameterOptimisation node.')
        

if __name__ == '__main__':
    try:
        ParameterOptimisation()
    except rospy.ROSInterruptException:
	    rospy.logerr('Could not start ParameterOptimisation node.')

#!/usr/bin/env python

from __future__ import division
import rospy
import carla
import time
import json
import os
from actors.ego_vehicle import EgoVehicle
from actors.vehicle import Vehicle
from actors.walker import Walker
import sys
import random
import numpy as np
from pdf import PDF
import math 
import matplotlib
import re

# For the error: Exception ignored in: <bound method Image.del of <tkinter.PhotoImage object at 0x7f1b5f86a710>> Traceback (most recent call last):
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import tensorflow as tf
from rss import RSS
from agents.navigation.controller import VehiclePIDController
import lanelet2
from lanelet2.core import Lanelet, LineString3d, Point2d, Point3d, getId, BoundingBox2d, BasicPoint2d

args_lateral_dict = {'K_P': 1, 'K_D': 0.0, 'K_I': 0} 
args_longitudinal_dict = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0}
max_throt = 0.75
max_brake = 0.3
max_steer = 0.8

class Scenario:

    def __init__(self, client = None, world = None, pdf = None, search_space = None):
        self.client = client
        self.world = world
        self.pdf = pdf
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.actor_list = []
        self.walker_list = []
        self.walker_controller_list = []
        self.parameters = rospy.get_param('parameter_range')
        self.scenario_settings = rospy.get_param('scenario')
        self.reward_settings = rospy.get_param('rewards')
        carla_settings = rospy.get_param('carla_settings')
        self.search_space = search_space
        self.rss = RSS()
        self.tm = self.client.get_trafficmanager()
        self.tm.set_global_distance_to_leading_vehicle(0.0)
        self.tm.global_percentage_speed_difference(-40.0) 
        self.tm.set_synchronous_mode(carla_settings['synchronous_mode'])
        self.tm_port = self.tm.get_port()
        self.scenario_plot_data = {}
        self.no_collision_timestep = 0
        

    '''
    action = [[Ego vehicle position, no. of pedestrian, pedestrian postion, pedestrian longitudinal position]]
    '''
    def generate(self, actions = None, episode = None): 

        #print("Generating the scenario from: {}".format(actions))
        self.collision_flag = False
        self.no_collision_timestep = 0
        reward = 0
        skip = False

        self.scenario_plot_data = {
            'speed_veh': [],
            'speed_ego': [],
            'timesteps': None,
            's_t': None,
            'lane_change_start': None,
            'lane_change_end': None,
            'euclidean_distance': []
        }


        
        # trigger_dist = actions[0][0].item()
        # cut_in_vel = actions[0][1].item()
        # start_to_cutin_time = actions[0][2].item()
        # cut_end_vel = actions[0][3].item()
        # cutin_to_cutend_time = actions[0][4].item()
        # adv_final_vel = actions[0][5].item()
        # cutend_to_final_time = actions[0][6].item()
        # ego_init_speed = actions[0][7].item()
        

        # trigger_dist = actions[0]
        # cut_in_vel = actions[1]
        # start_to_cutin_time = actions[2]
        # cut_end_vel = actions[3]
        # cutin_to_cutend_time = actions[4]
        # adv_final_vel = actions[5]
        # cutend_to_final_time = actions[6]
        # ego_init_speed = actions[7]


        actions = tf.keras.backend.get_value(actions)
        actions = np.array(tf.cast(actions, tf.float32))
        trigger_dist = actions[0][0][0].item()
        cut_in_vel = actions[0][0][1].item()
        start_to_cutin_time = actions[0][0][2].item()
        cut_end_vel = actions[0][0][3].item()
        cutin_to_cutend_time = actions[0][0][4].item()
        adv_final_vel = actions[0][0][5].item()
        cutend_to_final_time = actions[0][0][6].item()


        
        #*** Use time instead of distance

        # Temporary fix
        #trigger_dist += 1.0
        direction = -1 # 1 or -1
        start_count = 0
        cut_in_count = 0
        final_count = 0
        lane_change = False
        reached_loc = False
        distance = 0
        adv_init_speed = 5
        

        # Creation of Ego vehicle
        self.ego_vehicle = EgoVehicle(client = self.client, world = self.world, spawn_points = self.spawn_points, long_pos = -0.5)
        self.ego_vehicle.collision_sensor.listen(lambda event: self.ego_collision(event))
        self.ego_vehicle.player.set_autopilot(True,self.tm_port)
        self.actor_list.append(self.ego_vehicle)
        self.vehicle = Vehicle(client = self.client, world = self.world, spawn_points = self.spawn_points, long_pos = 0)
        self.actor_list.append(self.vehicle)
        self.world.tick()

        # If the RSS safe distance is less than actual distance, we need add some value to the list to compute the reward.
        rss_failure = 0 
        ttc_failure = 0 
        timesteps = 0
        timesteps_active_total = 0
        timestep_failure = []
        rss_plot_data = None
       
        if skip is not True:
            start_time = time.time()

        self.current_pos = self.vehicle.player.get_transform().location
        self.past_pos = self.vehicle.player.get_transform().location
        self.controller = VehiclePIDController(self.vehicle.player,
                                    args_lateral=args_lateral_dict,
                                    args_longitudinal=args_longitudinal_dict,
                                    max_throttle=max_throt,
                                    max_brake=max_brake,
                                    max_steering=max_steer)
        #control = carla.VehicleControl()
        #control.throttle = 0.5
        #self.vehicle.player.apply_control(control)
        adv_start_to_cutin_speed_profile = self.pdf.generate_velocity(starting_speed = adv_init_speed, final_speed = cut_in_vel, time = start_to_cutin_time) 
        adv_cutin_to_cutend_speed_profile = self.pdf.generate_velocity(starting_speed = cut_in_vel, final_speed = cut_end_vel, time = cutin_to_cutend_time) 
        adv_cutend_to_final_speed_profile = self.pdf.generate_velocity(starting_speed = cut_end_vel, final_speed = adv_final_vel, time = cutend_to_final_time) 
        lane_change_finished = False
        scenario_finished = False
        ego_init_loc = None
        episode_time = 0
        first_time = False
        proceed = False
        exit_count = 0
        rss_real_dist = []
        timestep = 0
        cartesian_data = {
            'ego': [],
            'adv': []
        }
        self.stop = False
        while True:
            vel_ego = self.ego_vehicle.player.get_velocity()
            if abs(vel_ego.x) > 0.2 and first_time == False:
                start_time = time.time()
                proceed = True
                first_time = True
            else:
                self.world.tick()

            if proceed == True:
                loc_ego = self.ego_vehicle.player.get_location()
                if start_count == 3:
                    veh_init_loc = self.vehicle.player.get_location()
                    ego_init_loc = self.ego_vehicle.player.get_location()
                    print(veh_init_loc)
                if start_count > 3:
                    vehicle_loc = self.vehicle.player.get_location()
                    distance = vehicle_loc.distance(loc_ego)
                
                current_waypoint = self.world.get_map().get_waypoint(self.vehicle.player.get_location())
                #The vehicle is location.x is increasing the negative direction, so the ahead lication.x is always less than previous location. 
                #So we need to you less than even though logically we expected to say ahead location is greater than previous.
                if  distance >= trigger_dist and reached_loc == False and abs(vehicle_loc.x) > abs(loc_ego.x):
                    right_lane = current_waypoint.get_right_lane()
                    self.waypointsList = right_lane.next(50)[0].next_until_lane_end(0.3)
                    if cut_in_count >= len(adv_cutin_to_cutend_speed_profile):
                        self.vel_ref = adv_cutin_to_cutend_speed_profile[len(adv_cutin_to_cutend_speed_profile)-1]
                    else:
                        self.vel_ref = adv_cutin_to_cutend_speed_profile[cut_in_count]
                    cut_in_count += 1

                    control_signal = self.controller.run_step(self.vel_ref,self.waypointsList[0])  
                    self.vehicle.player.apply_control(control_signal)
                    self.vel_ref *= direction
                    self.vehicle.player.set_velocity(carla.Vector3D(x=self.vel_ref,y=0,z=0))
                    self.world.tick()
                    lane_change = True
                    reached_loc = True
                    self.save_scenario_data(timestep)
                    self.scenario_plot_data['lane_change_start'] = timestep
                    timestep += 1

                    #Getting cartesian data
                    loc_ego = self.ego_vehicle.player.get_location()
                    loc_v1 = self.vehicle.player.get_location()
                    cartesian_data['ego'].append({'x':loc_ego.x, 'y': loc_ego.y})
                    cartesian_data['adv'].append({'x':loc_v1.x, 'y': loc_v1.y})     
                    print("The adversary vehicle changed the lane")               
                elif lane_change == True:
                    for i in range(len(self.waypointsList)-1):
                        if lane_change_finished == True:
                            if final_count >= len(adv_cutend_to_final_speed_profile):
                                self.vel_ref = adv_cutend_to_final_speed_profile[len(adv_cutend_to_final_speed_profile)-1]
                            else:
                                self.vel_ref = adv_cutend_to_final_speed_profile[final_count]
                            final_count += 1
                        else:
                            self.vel_ref = adv_cutin_to_cutend_speed_profile[cut_in_count]
                            cut_in_count += 1

                        if self.stop == False:
                            control_signal = self.controller.run_step(self.vel_ref,self.waypointsList[i+1])  
                            self.vehicle.player.apply_control(control_signal)
                            self.vel_ref *= direction
                            self.vehicle.player.set_velocity(carla.Vector3D(x=self.vel_ref,y=0,z=0))
                        self.world.tick()

                        # Actual distance
                        loc_ego = self.ego_vehicle.player.get_location()
                        loc_v1 = self.vehicle.player.get_location()
                        actual_dist = loc_ego.distance(loc_v1)

                        # RSS distance
                        vel_ego = self.ego_vehicle.player.get_velocity()
                        vel_veh = self.vehicle.player.get_velocity()
                        rss_dist = self.rss.calculate_rss_safe_dist(abs(vel_ego.x), abs(vel_veh.x))
                        rss_real_dist.append({'real': actual_dist, 'rss': rss_dist})
                        
                        y_diff = abs(abs(self.waypointsList[i+1].transform.location.y) - abs(veh_init_loc.y))
                        if y_diff > 3 and lane_change_finished == False:
                            lane_change_finished = True
                            self.scenario_plot_data['lane_change_end'] = timestep
                        
                        '''
                        y_diff = abs(abs(self.vehicle.player.get_location().y) - abs(veh_init_loc.y))
                        if y_diff > 0.5 and lane_change_finished == False:
                            print("Lane_change finished, y_diff: {}".format(y_diff))
                            lane_change_finished = True
                            self.scenario_plot_data['lane_change_end'] = timestep
                        '''

                        # Once labe changed, this loop will be playing
                        end_time = time.time()
                        episode_time =  end_time - start_time 
                        ego_dist_travelled = loc_ego.distance(ego_init_loc)
                        cartesian_data['ego'].append({'x':loc_ego.x, 'y': loc_ego.y})
                        cartesian_data['adv'].append({'x':loc_v1.x, 'y': loc_v1.y})

                        # This is the criteria for end of scenario.
                        #if self.collision_flag is True or episode_time > self.scenario_settings['scenario_end']['max_time']:
                        if episode_time > self.scenario_settings['scenario_end']['max_time']:
                            print("Episode about to finish")
                            scenario_finished = True
                            break

                        # Just in case ego vehicle never started
                        if exit_count > 10000 and first_time == False:
                            scenario_finished = True
                            break

                        if self.no_collision_timestep > 7:
                            self.ego_vehicle.player.set_velocity(carla.Vector3D(x=0.0,y=0,z=0))
                            self.vehicle.player.set_velocity(carla.Vector3D(x=0.0,y=0,z=0))
                            self.stop = True


                        self.save_scenario_data(timestep)
                        timestep += 1

                    lane_change = False    
                else:
                    if start_count >= len(adv_start_to_cutin_speed_profile):
                        self.vel_ref = adv_start_to_cutin_speed_profile[len(adv_start_to_cutin_speed_profile)-1]
                    else:
                        self.vel_ref = adv_start_to_cutin_speed_profile[start_count]
                    control_signal = self.controller.run_step(self.vel_ref,current_waypoint.next(100)[0])  
                    self.vehicle.player.apply_control(control_signal)
                    self.vel_ref *= direction
                    self.vehicle.player.set_velocity(carla.Vector3D(x=self.vel_ref,y=0,z=0))
                    self.world.tick()
                    start_count += 1
                    self.save_scenario_data(timestep)
                    timestep += 1

                    #Getting cartesian data
                    loc_ego = self.ego_vehicle.player.get_location()
                    loc_v1 = self.vehicle.player.get_location()
                    cartesian_data['ego'].append({'x':loc_ego.x, 'y': loc_ego.y})
                    cartesian_data['adv'].append({'x':loc_v1.x, 'y': loc_v1.y})

                end_time = time.time()
                episode_time =  end_time - start_time 
                if self.collision_flag is True or episode_time > self.scenario_settings['scenario_end']['max_time']:
                    scenario_finished = True

                if scenario_finished == True:
                    break

                exit_count +=1

        reward = self.compute_reward(rss_real_dist)

        # Find s & t
        self.commpute_s_t(cartesian_data)
        
        return reward, skip, rss_real_dist, self.scenario_plot_data

    def commpute_s_t(self, cartesian_data):
        ls_ego = None
        prev_p_ego = None
        s_t = []
        s_ego = []
        s_ego.append(0)
        s_total = 0
        
        for index in range(len(cartesian_data['ego'])):
            p2_ego = Point3d(getId(), cartesian_data['ego'][index]['x'], cartesian_data['ego'][index]['y'], 0)
            if prev_p_ego is None:
                prev_p_ego = Point3d(getId(), cartesian_data['ego'][index]['x'], cartesian_data['ego'][index]['y'], 0)
            elif ls_ego is None and prev_p_ego is not None:
                ls_ego = LineString3d(getId(), [prev_p_ego, p2_ego])
                
                p_dist = lanelet2.geometry.distance(prev_p_ego, p2_ego)
                s_total += p_dist
                s_ego.append(s_total) 
                prev_p_ego = p2_ego
            else:
                ls_ego.append(p2_ego)

                p_dist = lanelet2.geometry.distance(prev_p_ego, p2_ego)
                s_total += p_dist
                s_ego.append(s_total) 
                prev_p_ego = p2_ego

        _len = len(cartesian_data['ego'])-1

        p_last_ego = Point3d(getId(), cartesian_data['ego'][_len]['x']+200, cartesian_data['ego'][_len]['y'], 0)
        ls_ego.append(p_last_ego)


        prev_p_adv = None
        s_adv_total = 0
        ego_s = 0
        for index in range(len(cartesian_data['adv'])):
            if prev_p_adv is None:
                prev_p_adv = Point3d(getId(), cartesian_data['adv'][0]['x'], cartesian_data['adv'][0]['y'], 0)
                continue
            p_adv = BasicPoint2d(cartesian_data['adv'][index]['x'], cartesian_data['adv'][index]['y'])
            projected_point_on_ls = lanelet2.geometry.project(lanelet2.geometry.to2D(ls_ego), p_adv)
            t = math.sqrt( (p_adv.x - projected_point_on_ls.x)**2 + (p_adv.y - projected_point_on_ls.y)**2 )
            
            # Average width of the car is 1.90. The point is at the center of the car, we need to cosider half of both cars
            # Basically, we can round to 2.0.
            t -= 2.0
            p2_adv = Point3d(getId(), cartesian_data['adv'][index]['x'], cartesian_data['adv'][index]['y'], 0)
            p_adv_dist = lanelet2.geometry.distance(prev_p_adv, p2_adv)
            s_adv_total += p_adv_dist
            
            if index < len(s_ego):
                ego_s = s_ego[index]

            #s_t.append({'s':s_adv_total, 's_ego': ego_s,'t':t, 'timestep': index})
            s_t.append({'s':s_adv_total, 's_ego': ego_s,'t':t, 'timestep': index, 'ego_x': cartesian_data['ego'][index]['x'], 'ego_y': cartesian_data['ego'][index]['y'], 'adv_x': cartesian_data['adv'][index]['x'], 'adv_y': cartesian_data['adv'][index]['y']})
            prev_p_adv = p2_adv
        
        #print(s_t)
        self.scenario_plot_data['s_t'] = s_t

    def save_scenario_data(self, timestep):
        vel_veh = self.vehicle.player.get_velocity()
        vel_ego = self.ego_vehicle.player.get_velocity()
        self.scenario_plot_data['speed_veh'].append(abs(vel_veh.x))
        self.scenario_plot_data['speed_ego'].append(abs(vel_ego.x))
        self.scenario_plot_data['timesteps'] = timestep

        # Actual distance
        loc_ego = self.ego_vehicle.player.get_location()
        loc_v1 = self.vehicle.player.get_location()
        actual_dist = loc_ego.distance(loc_v1)
        self.scenario_plot_data['euclidean_distance'].append(actual_dist)

    
    def ego_collision(self, event):
        self.collision_flag = True
        self.no_collision_timestep += 1
        print("Collision detected")

    def compute_reward(self, rss_real_dist):
        reward =  self.reward_settings['rss']['min']
        if self.collision_flag == True:
            reward = self.reward_settings['collision']
        else:
            unsafe = 0
            if len(rss_real_dist) > 0:
                for index in range(len(rss_real_dist)):
                    if rss_real_dist[index]['real'] < rss_real_dist[index]['rss']:
                        unsafe += 1
                if unsafe == 0:
                    print("No unsafe timesteps")
                reward = self.normalise_to_x(unsafe, 0, len(rss_real_dist), self.reward_settings['rss']['min'], self.reward_settings['rss']['max'])
                
        return reward



    # Weather presets: ['Clear Noon', 'Clear Sunset','Cloudy Noon', 'Cloudy Sunset', 'Default', 'Hard Rain Noon', 
    # 'Hard Rain Sunset', 'Mid Rain Sunset', 'Mid Rainy Noon', 'Soft Rain Noon', 'Soft Rain Sunset', 
    # 'Wet Cloudy Noon','Wet Cloudy Sunset','Wet Noon','Wet Sunset']
    def find_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    def normalise(self, value = None, minimum = None, maximum = None):
        
        # min = 0
        # max = 10
        # y = (x - min) / (max - min), normalise between 0 and 1.
        return round(((value - minimum) / (maximum-minimum)), 2)

    def normalise_to_x(self, value = None, minimum = None, maximum = None, newrange_a = None, newrange_b = None):
        
        # Using this formula: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        # Actual range is : [0, 1] - [min, max]
        # New range is: [-0.1, 0.1] - [a,b]
        # Equation: (b-a)*((x-min)/(max-min))+a
        return ((newrange_b - newrange_a)*((value-minimum)/(maximum-minimum))+newrange_a)

    def reset(self):
        self.walker_controller_list = []
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        self.walker_list = []

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = None

if __name__ == '__main__':
    Scenario()

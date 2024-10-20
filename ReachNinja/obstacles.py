import numpy as np
import time
import os
import shutil
import datetime
import logging
import tkinter as tk
import pygame
import os
from marker import Marker


class Obstacle(Marker):
    def __init__(self, gameshape, tau, exploding_perc, vel_max = 0.5, vel_min = 1, acc = 100, theta_max = -30, theta_min = -150, max_obs_time = 1, max_unobs_time = 0.15, isRegular = True, isStatic=False,
                                                isMagnetic = False, rand_gen=np.random):
        Marker.__init__(self,'Target')
        self.shape = gameshape
        self.isRegular = isRegular
        self.doReplace = False
        self.isStatic = isStatic
        self.isMagnetic = isMagnetic
        self.rand_gen = rand_gen
        self.resetObstacle(tau, exploding_perc, vel_max= vel_max, vel_min = vel_min, acc = acc, theta_max = theta_max, theta_min = theta_min, max_obs_time = 1, max_unobs_time = 0.15)
        

    def resetObstacle(self, tau, exploding_perc, vel_max = 0.5, vel_min = 1, acc = np.array([0, 100]), theta_max = -30, theta_min = -150, max_obs_time = 1, max_unobs_time = 0.15):
        self.resetMarker('Target')
        self.x = np.around(self.rand_gen.rand(1)*self.shape[1])
        if vel_min == vel_max and vel_max == 0:
            stationary = True
        else:
            stationary = False
        if stationary:
            if self.isStatic:
                self.x = np.around(self.shape[1]-100)
                self.y = np.around(self.shape[0]-100)
            else:
                self.y = np.around(np.clip(self.rand_gen.rand(1),0.1,0.9)*self.shape[0])
                self.y = self.y[0]
        else:
            self.y = 0
    
        # Changing for now
        self.velocity_scale = 400
        self.velocity = np.clip(self.rand_gen.rand(1),vel_min,vel_max)*self.velocity_scale
        #print(f"initial velocity is velocity is {self.velocity}")
        self.theta = self.rand_gen.randint(theta_min, theta_max)*np.pi/180
        self.velocity = [np.cos(self.theta), np.sin(self.theta)]*self.velocity
        self.acceleration = acc
        self.loc = np.array([self.x, (self.shape[0] - self.y)])
        #self.last_time = curr_time
        self.tau = tau
        self.inframe = True
        self.max_unobs_time = max_unobs_time
        self.max_obs_time = max_obs_time
        
        # self.setPartialObservable()
        self.perc_obs = 0 #np.clip(np.random.rand(),0.5,1)  
        self.start_time = time.time()
        self.marker_color = (0,0,255)
        if self.isRegular:
            self.setRegularObstacle()
        else:
            self.setExplodingObstacle()


    def updatePosition(self, playerLoc, playerRad, magnetic_coef):    
        
        vel_change = np.array([0,0])
        acc_change = np.array([0,0])
        
        if self.isMagnetic:
            field_vector = self.loc - playerLoc
            field_dist = np.clip(int(np.linalg.norm(field_vector)),1,1000)

            if field_dist < self.radius*10:
                if field_dist <= 0.01:
                    field_dist = 0.01
                field_dir = (field_vector/field_dist)
                acc_change = (magnetic_coef*playerRad/(field_dist^2))*field_dir

            if self.obstacle_type == 'Exploding':
                acc_change = -acc_change

        self.velocity = np.clip(self.velocity + self.tau*(self.acceleration+acc_change),-500,500)
        self.loc = (self.loc + self.tau*self.velocity)
        
        if self.loc[0] >= 0 and self.loc[0] < self.shape[1] and self.loc[1] >= 0 and self.loc[1] < self.shape[0]:
            self.inframe = True
        else:
            self.inframe = False
        return self.loc

    def checkCollision(self, player, scaling_factor):
        return np.linalg.norm(self.loc - player.loc) < (self.radius + player.radius)*scaling_factor
            
    def setRegularObstacle(self):
        self.obstacle_type = 'Regular'

    def setExplodingObstacle(self):
        self.obstacle_type = 'Exploding'
        self.marker_color = (0, 0, 0)

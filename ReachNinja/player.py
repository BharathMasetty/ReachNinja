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
from obstacles import Obstacle


class Player(Marker):
    def __init__(self, gameshape, damping, damping_mat, mirror, id = 1):
        super().__init__("Player")
        self.id = id
        self.shape = gameshape
        self.attempt = 0    # TO CHANGE
        self.setRadius()
        self.loc = np.array([gameshape[1]/2, gameshape[0]/2])
        self.last_time = -1
        self.setScore()
        self.setStartTime()
        self.setPartialObservable()
        self.perc_obs = 0 #np.clip(np.random.rand(),0.5,1)  
        self.obstacle_count = 0
        self.exploding_perc = 0.3
        self.marker_color = (255,0,0)
        self.rotation_angle = 0 #degrees
        self.vel = np.array([0,0])
        self.acc = np.array([0, 0])
        self.old_loc_wt = 0.3
        self.new_loc_wt = 0.7
        self.gameType = 0
        self.check_targets = False
        self.damping = damping 
        self.damping_mat = damping_mat
        self.mirror = mirror

    def resetObsTime(self, max_obs_time = 1, max_unobs_time = 0.15):
        print(f'{max_obs_time} {max_unobs_time}')
        self.max_obs_time = max_obs_time
        self.max_unobs_time = max_unobs_time
    
    def setShape(self, shape = np.array((480, 640))):
        self.shape = shape

    def updateAcceleration(self):
        self.acc = self.damping_mat.dot(self.vel)


    def updatePosition(self, new_loc, curr_time):
        # print(f"Old: {self.loc}, new: {new_loc}, dist: {np.linalg.norm(new_loc - self.loc)}")
        # if np.linalg.norm(new_loc - self.loc) > 5:
        #     new_loc = self.loc
        av_loc = self.loc + self.vel*(curr_time - self.last_time)
        # av_loc = new_loc
        if (new_loc != self.loc).all():
            av_loc = (av_loc*self.old_loc_wt + new_loc*self.new_loc_wt)
        av_loc[0] = np.clip(av_loc[0], 0, self.shape[1])
        av_loc[1] = np.clip(av_loc[1], 0, self.shape[0])

        # print(f"vel: {self.vel}")
        self.updateAcceleration()
        self.vel = (av_loc - self.loc)/(curr_time - self.last_time) + self.acc*(curr_time - self.last_time)
        # print(f"velnew: {self.vel}")
        self.loc = av_loc
        # print(f"newloc: {self.loc}")
        self.last_time = curr_time

    def setScore(self):
        self.score = 0

    def setPlayTime(self, play_time = 60):
        self.play_time = play_time#int(input("How long will you play for"))

    def rotateMarkerPosition(self, shape):
        # print(shape)
        marker_pose = self.loc - np.array((shape[1], shape[0]))/2
        c, s = np.cos(self.rotation_angle*np.pi/180), np.sin(self.rotation_angle*np.pi/180)
        R = np.array(((c, -s), (s, c)))
        marker_pose = R.dot(marker_pose) + (np.array((shape[1], shape[0])))/2
        return marker_pose.astype(int)

    def setPlayID(self):
        self.play_id = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def newGameInit(self, curr_time):
        self.setScore()
        self.setPlayTime()
        self.setStartTime(curr_time)
        self.setPartialObservable()
        self.setPlayID()
        self.attempt += 1
            


    

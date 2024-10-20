import numpy as np
import time
import os
import shutil
import datetime
import logging
import tkinter as tk
import pygame
import os

class Marker:
    def __init__(self, marker_type = 'Unlabeled'):
        self.resetMarker()
        self.min_radius = 5
        self.max_radius = 10
        self.perc_obs = 0
        self.max_unobs_time = 0.15
        self.max_obs_time = 1
        self.unobs_start = -1
        self.marker_color = (0,255,0)
        self.visible = True

    def resetMarker(self, marker_type = 'Unlabeled'):
        self.type = marker_type
        self.setRadius()
        self.setObservable()
        self.setStartTime()
        self.obs_start = time.time()


    def setRadius(self, rad = 10):
            self.radius = rad

    def setObservable(self):
        self.observable = True

    def setPartialObservable(self):
        self.observable = False

    def changeObsPerc(self, new_perc):
        self.perc_obs = new_perc

    def checkObservable(self, curr_time):
        if self.max_unobs_time == 0:
            self.observable = True

        if self.observable == False:
            if ((curr_time - self.obs_start) > self.max_obs_time and self.unobs_start == -1) or \
                    ((curr_time - self.unobs_start) < self.max_unobs_time and self.obs_start == -1):
                self.visible = False
                self.obs_start = -1
                if self.unobs_start == -1:
                    self.unobs_start = curr_time                  
            else:
                self.visible = True
                self.unobs_start = -1
                if self.obs_start == -1:
                    self.obs_start = curr_time
                    
        else:
            self.visible = True
            self.unobs_start = -1
            if self.obs_start == -1:
                self.obs_start = curr_time
            
        
        return self.visible



    def setStartTime(self, start_time = -1):
        self.start_time = start_time

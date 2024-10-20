import cv2
import numpy as np
import time
import os
import shutil
import datetime
import logging
import tkinter as tk
import pygame
import os


class Gamelog:
    def __init__(self, player, game_id):
        self.player_id = player.id
        self.game_id = game_id
        self.player_folder = f"LogFiles/Player_{self.player_id}"
        self.createDefaultLogFolder()
        
        
    def createDefaultLogFolder(self):
        os.makedirs(self.player_folder, exist_ok = True)
        self.calibration_log_folder = f"{self.player_folder}/Calibration_{self.game_id}"
        self.savefolder = self.calibration_log_folder
        os.makedirs(self.calibration_log_folder, exist_ok = True)
        self.frame_id = 1

    def newGameLog(self, attempt, init_line, max_obstacle_count = 5):
        
        self.attempt = attempt        
        self.image_log_folder = f"{self.player_folder}/Game_{self.attempt}_{self.game_id}"
        self.savefolder = self.image_log_folder
        os.makedirs(self.image_log_folder, exist_ok = True)
        self.createLogger()
        self.logger.info(init_line)
        self.addHeaders(max_obstacle_count, init_line) 
        self.writeLogLine()

    def createLogger(self):
        """
        Method to return a custom logger with the given name and level
        """
        self.logger_name = f'{self.player_folder}/Log_{self.attempt}_{self.game_id}.log'
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        format_string = ("%(message)s")
        log_format = logging.Formatter(format_string)
        
        # Creating and adding the file handler
        self.file_handler = logging.FileHandler(self.logger_name, mode='a')
        self.file_handler.setFormatter(log_format)
        self.logger.addHandler(self.file_handler)
        self.frame_id = 1
        
    def addHeaders(self, max_obstacle_count, init_line):
        
        self.logLine = ['Frame','Time', 'Attempt', \
                        'Play ID','Marker #s', \
                        'Player:X', 'Player:Y', \
                        'Player:Int', 'Player:Score', \
                        'Player:Obs', 'Player:GameType'] 
        for i in range(1, max_obstacle_count+1):
            self.logLine.append(f"M{i+1}:Type")
            self.logLine.append(f"M{i+1}:Rad")
            self.logLine.append(f"M{i+1}:PX")
            self.logLine.append(f"M{i+1}:PY")
            self.logLine.append(f"M{i+1}:VX")
            self.logLine.append(f"M{i+1}:VY")
            self.logLine.append(f"M{i+1}:AX")
            self.logLine.append(f"M{i+1}:AY")

    def startPlayerLine(self, curr_time, player, obstacle_count, game_type, intervention_type = 0):
        self.frame_id += 1
        self.logLine = [self.frame_id, curr_time - player.start_time, player.attempt, player.play_id, obstacle_count, \
                        player.loc[0], player.loc[1], intervention_type, player.score, player.visible, game_type]

    def addObstacleLine(self, curr_obstacles):
        for o in curr_obstacles:
            self.logLine.append(o.obstacle_type)
            self.logLine.append(o.radius)
            self.logLine += [o.loc[0], o.loc[1]]
            self.logLine += [o.velocity[0], o.velocity[1]]
            self.logLine += [o.acceleration[0], o.acceleration[1]]

    def writeLogLine(self):
        self.logLine = list(map(str,self.logLine))
        self.logger.info(",".join(self.logLine))
        self.clearLogLine()

    def clearLogLine(self):
        self.LogLine = []

    def saveImage(self, frame):
        cv2.imwrite(f"{self.savefolder}/{self.frame_id:010d}.png",frame)







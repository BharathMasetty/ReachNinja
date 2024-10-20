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
import sys
from marker import Marker
from player import Player
from obstacles import Obstacle
from gamelog import Gamelog
import json

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Game:
    def __init__(self, play_time = 40, camera_port = 0):
        pygame.init()
        self.win_sound = [pygame.mixer.Sound(resource_path("Sounds/pop1.wav")),
                          pygame.mixer.Sound(resource_path("Sounds/pop4.wav")),
                          pygame.mixer.Sound(resource_path("Sounds/pop6.wav"))]
        self.error_sound = pygame.mixer.Sound(resource_path("Sounds/boom1.wav"))

        pygame.font.init()
        all_fonts = pygame.font.get_fonts()
        self.textsize ={"large": pygame.font.SysFont(all_fonts[3], 200),
                        "medium": pygame.font.SysFont(all_fonts[3], 100),
                        "mid":pygame.font.SysFont(all_fonts[3],60),
                        "small": pygame.font.SysFont(all_fonts[3], 40),
                        "tiny": pygame.font.SysFont(all_fonts[3], 33)}
        # self.textsize = {"large": pygame.font.Font('freesansbold.ttf',200),
        #             "medium": pygame.font.Font('freesansbold.ttf',100),
        #             "small":pygame.font.Font('freesansbold.ttf',40),
        #             "tiny": pygame.font.Font('freesansbold.ttf',20)}

        self.textcolor = {"gray":   (100,100,100),
                          "white":  (255,255,255),
                          "black":  (0,0,0),
                          "red":    (255,0,0),
                          "green":  (0,255,0),
                          "blue":   (0,0,255)}
        self.camera_port = camera_port
        self.play_time = play_time
        self.game_type = None 
        self.wait_time = 5
        self.obstacle_count = 0
        self.intervention = 0
        self.blob_area = 1
        self.damping = 0 # 0.5 # -0.5
        self.damping_mat = self.damping*np.array(((-1, -1),(-1, 1)))
        self.mirror = False
        self.scaling_factor = 10
        self.display_score = 0
        self.old_score = 0
        self.saveframe = False
        self.game_params = json.load(open(resource_path('GameParams.json'),'r'))
        self.initializeGameType()
        self.initializeGameFrame()
        self.setGameID()
        self.resetBounds()

    def initializeGameType(self, exploding_perc = 0.33, max_unobs_time = 0.2, max_obs_time = 0.8, 
                                vel_max = 0.5, vel_min = 1, acc = 100, theta_max = -30, theta_min = -150,
                                min_obstacles = 1, max_obstacles = 4, damping = 0, mirror = False):
        self.exploding_perc = exploding_perc
        self.max_unobs_time = max_unobs_time
        self.max_obs_time = max_obs_time
        self.acceleration = np.array([0,acc])
        self.velocity_min = vel_max
        self.velocity_max = vel_min
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.damping = damping # 0.5 # -0.5
        self.damping_mat = self.damping*np.array(((-1, -1),(-1, 1)))
        self.mirror = mirror
        

    def initializeGameFrame(self):

        self.video_capture = cv2.VideoCapture(self.camera_port)
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # self.game_display = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
        # screen_width = pygame.display.Info().current_w
        # screen_height = pygame.display.Info().current_h

        print(f"Width is {screen_width} and height is {screen_height}")

        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.screen_size_tuple = (screen_width, screen_height)
        self.original_frame_tuple = (frame_width, frame_height)
        self.aspectratio = self.original_frame_tuple[0]/self.original_frame_tuple[1]#frame_width/frame_height

        self.frame_size_tuple = (int(self.screen_size_tuple[1]*self.aspectratio),self.screen_size_tuple[1])

        self.width_ratio = self.original_frame_tuple[0]/(self.screen_size_tuple[1]*self.aspectratio)
        self.height_ratio = self.original_frame_tuple[1]/self.screen_size_tuple[1]

        self.frame_offset = int((self.screen_size_tuple[0] - int(self.screen_size_tuple[1]*self.aspectratio))/2)

        self.game_display = pygame.display.set_mode(self.screen_size_tuple)

        self.scaling_factor = 1/self.height_ratio

        self.calibrate_button = pygame.Rect(self.frame_offset + self.frame_size_tuple[0]/5, 3*self.frame_size_tuple[1]/4, self.frame_size_tuple[0]/5, self.frame_size_tuple[0]/15)
        self.start_button = pygame.Rect(self.frame_offset + 3*self.frame_size_tuple[0]/5, 3*self.frame_size_tuple[1]/4, self.frame_size_tuple[0]/5, self.frame_size_tuple[0]/15)
        # self.button_color = (self.textcolor["grey"]
        self.player_id_textbox = pygame.Rect(100, 100, 50, 50)
        # self.textbox_color = (255,255,255)

        pygame.display.set_caption('Reach Ninja')

    def setGameID(self):
        self.game_id = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def run(self):
        
        clock = pygame.time.Clock()
        crashed = False
        self.game_mode = None
        key = cv2.waitKey(1)
        key = key & 0xFF

        player_id = 1
        self.player = Player((self.original_frame_tuple[1], self.original_frame_tuple[0]), self.damping, self.damping_mat,self.mirror, player_id)
        self.init_line = f"Screen: {self.screen_size_tuple}; Original: {self.original_frame_tuple}; Modified: {self.frame_size_tuple}"
        self.gamelog = Gamelog(self.player, self.game_id)

        while not crashed:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    crashed = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.game_mode = 'StartPlay'
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    self.game_mode = 'Calibrate'
                    self.current_time = time.time()  
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    self.game_mode = None
                    self.player.start_time = -1
                    # self.player.attempt -= 1 # TO CHANGE
                    print('Exiting... ')

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.start_button.collidepoint(event.pos):
                        self.game_mode = 'StartPlay'
                    elif self.calibrate_button.collidepoint(event.pos):
                        self.game_mode = 'Calibrate'
                        self.current_time = time.time()  
                    else:
                        self.pickColor()
                
            if self.game_mode == None:
                self.displayDefault()

            elif self.game_mode == 'StartPlay' and self.player.start_time == -1:
                print(f'Starting... ')
                self.game_type = 'Curriculum'
                self.frame_id = 1
                self.waitGame(time.time())
                self.initializeNewGame()
                self.gamelog.newGameLog(self.player.attempt, self.init_line)
                

            elif self.game_mode == 'InPlay' and (time.time() - self.player.start_time) <= self.play_time:
                
                self.current_time = time.time()
                self.saveframe = True
                self.updateGameFrame()
                self.updateGamelog()
                
            elif self.game_mode == 'InPlay':
                self.game_mode = None
                self.saveframe = False
                pygame.time.delay(2000)
                self.player.setStartTime()

            elif self.game_mode == 'Calibrate':
                print('Calibrating...')
                self.frame_id = 1
                self.saveframe = False
                self.gamelog.savefolder = self.gamelog.calibration_log_folder
                self.calibrateGame()
                self.game_mode = None

            pygame.display.update()
            clock.tick(60)

        self.video_capture.release()
        cv2.destroyAllWindows()
        pygame.quit()


    def videoFrameToPyGameDisplay(self, frame, saveframe = False, savelocation = []):
        
        frame, center, blob_area = self.blobDetect(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if saveframe == True:
            cv2.imwrite(f"{self.calibration_log_folder}/{self.frame_id:010d}.png",frame)

        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, self.frame_size_tuple)
        return frame

    def updateGameFrame(self):
        ret, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        self.stepGame(frame)
        self.game_display.fill([255,255,255])
        pygame.draw.rect(self.game_display, (0,0,0), (0,0,self.frame_offset, self.frame_size_tuple[1]))
        pygame.draw.rect(self.game_display, (0,0,0), (self.frame_offset + self.frame_size_tuple[0],0,self.frame_offset, self.frame_size_tuple[1]))
        
        # print(self.player.loc, self.height_ratio)
        playerloc = (int(self.player.loc[0]/self.width_ratio) + self.frame_offset, int(self.player.loc[1]/self.height_ratio))
        if self.player.checkObservable(self.curr_time):
            pygame.draw.circle(self.game_display, self.player.marker_color, playerloc, int(self.scaling_factor*self.player.radius))

        for o in self.curr_obstacle:
            obstacleloc = (int(o.loc[0]/self.width_ratio) + self.frame_offset, int(o.loc[1]/self.height_ratio))
            pygame.draw.circle(self.game_display, o.marker_color, obstacleloc, int(self.scaling_factor*o.radius))
            
        
        self.messageDisplay(f"{int(self.player.score)}", (self.frame_offset/2, self.frame_size_tuple[1]/2), self.textcolor["gray"])
        text_rect = self.getTextRect(f"{int(self.player.score)}")
        self.messageDisplay("Score:", (self.frame_offset/2, self.frame_size_tuple[1]/2 - text_rect[3]), self.textcolor["gray"], "small")

        self.messageDisplay(f"{int(self.play_time - (self.curr_time - self.player.start_time))}s", \
                                                        (self.frame_size_tuple[0] + int(self.frame_offset*3/2), self.frame_size_tuple[1]/2), self.textcolor["gray"])
        text_rect = self.getTextRect(f"{int(self.player.score)}") # TO CHANGE
        self.messageDisplay('Time:', (self.frame_size_tuple[0] + int(self.frame_offset*3/2), self.frame_size_tuple[1]/2 - text_rect[3]), self.textcolor["gray"], "small")


        disp_frames = 10
        if self.display_score > 0 and self.display_score <= disp_frames:
            score_change = self.player.score - self.old_score
            if score_change > 0:
                dispmsg = f"{int(score_change)}"

            else:
                dispmsg = f"-{int(np.absolute(score_change))}"

            self.messageDisplay(dispmsg, playerloc, self.textcolor["black"], "tiny")
            self.display_score += 1
            if self.display_score == disp_frames:
                self.old_score = self.player.score
                self.display_score = 0

        pygame.display.update()

    def stepGame(self, rawframe):

        self.curr_time = time.time()

        frame, center, blob_area = self.blobDetect(rawframe)

        if blob_area >= self.blob_area:
            new_loc = center
        else:
            new_loc = self.player.loc

        self.player.updatePosition(np.array(new_loc), self.curr_time)

        new_obstacle = []
        resetobstacle_count = False
        for o in self.curr_obstacle:
            o.updatePosition(self.curr_time)
            if o.checkCollision(self.player, self.scaling_factor/2):
                addscore = self.updateScore(o)
                if o.obstacle_type == 'Exploding':
                    pygame.mixer.Sound.play(self.error_sound)
                    pygame.time.delay(500)
                else:
                    try:
                        p = max( i for i,o in enumerate(addscore - np.array([0, 8, 15])) if o > 0)
                    except:
                        p = 0
                    pygame.mixer.Sound.play(self.win_sound[p])
                resetobstacle_count = True
                self.display_score = 1
            elif not o.inframe:
                resetobstacle_count = True
            else:
                new_obstacle.append(o)

        self.curr_obstacle = new_obstacle

        if resetobstacle_count:
            self.newobstacle_count(self.curr_obstacle)

        while len(self.curr_obstacle) < self.obstacle_count:
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                        self.current_time, self.exploding_perc, \
                                                        self.velocity_max, self.velocity_min, \
                                                        self.acceleration, self.theta_max, self.theta_min,\
                                                        self.max_obs_time, self.max_unobs_time))

    def blobDetect(self, frame):
        center = [0, 0]
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        blob_area = 0
        if len(contours) > 0:
            blob = max(contours, key=lambda el: cv2.contourArea(el))
            M = cv2.moments(blob)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, center, self.player.radius, (0,0,255), -1)
            cv2.drawContours(frame, [blob],-1, (0,255,0), 3)
            blob_area = cv2.contourArea(blob)

        if self.saveframe:
            self.gamelog.saveImage(frame)

        return frame, center, blob_area

    def resetBounds(self,lower = [170, 180,70], upper = [180, 255,255]):
        self.lower = np.array(lower, dtype = "uint8")
        self.upper = np.array(upper, dtype = "uint8")

    def pickColor(self):
        ret, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        pos = pygame.mouse.get_pos()
        x = int((pos[0] - self.frame_offset)*self.width_ratio)
        y = int(pos[1]*self.height_ratio)
        image_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        pixel = image_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper_new =  np.clip(np.array([pixel[0] + 10, pixel[1] + 80, pixel[2] + 100]), 0, 255)
        lower_new =  np.clip(np.array([pixel[0] - 10, pixel[1] - 80, pixel[2] - 100]), 0, 255)

        self.resetBounds(lower_new, upper_new)
        frame, center, blob_area = self.blobDetect(frame)
        self.blob_area = blob_area*0.1

    def waitGame(self, call_time):
        while ((time.time() - call_time) < self.wait_time):
            self.game_display.fill([255,255,255])
            text_center = ((self.frame_size_tuple[0]/2) + self.frame_offset,(self.frame_size_tuple[1]/2))
            self.messageDisplay(f"{int(self.wait_time - (time.time() - call_time))}", text_center, self.textcolor["black"])
            pygame.display.update()

    def textObjects(self, text, font, color):
        text_surface = font.render(text, True, color)
        return text_surface, text_surface.get_rect()

    def getTextRect(self, text, textsize_type = 'medium', color = (0,0,0)):
        font = self.textsize[textsize_type]
        text_surface = font.render(text, True, color)
        return text_surface.get_rect()

    def messageDisplay(self, text, text_center, color, textsize_type = "medium"):
        # large_text = pygame.font.Font('freesansbold.ttf',200)
        # medium_text = pygame.font.Font('freesansbold.ttf',100)
        # small_text = pygame.font.Font('freesansbold.ttf',20)
        
        text_surf, text_rect = self.textObjects(text, self.textsize[textsize_type], color)
        text_rect.center = text_center
        self.game_display.blit(text_surf, (text_rect[0], text_rect[1]))
        
    def newobstacle_count(self, curr_obstacle = []):
        self.check_targets = False
        for o in curr_obstacle:
            if o.obstacle_type == 'Regular':
                self.check_targets = True
        if self.check_targets == False:
            if len(curr_obstacle)+1 < self.max_obstacles+1:
                self.obstacle_count = np.random.randint(len(curr_obstacle)+1,self.max_obstacles+1) 
            else:
                self.obstacle_count = min(len(curr_obstacle) + 1, self.max_obstacles + 1)
        else:
            self.obstacle_count = np.random.randint(self.min_obstacles,self.max_obstacles+1)


    def updateScore(self, marker):
        if marker.obstacle_type == 'Regular':
            addscore = 1
            # Size Update
            addscore = addscore * (marker.min_radius/marker.radius)
            # velocityUpdate
            if self.velocity_min != 0:
                addscore = addscore * (np.linalg.norm(marker.velocity)/marker.velocity_scale)/self.velocity_min 
            #markerNumberUpdate 
            addscore = addscore + (self.obstacle_count - 1)/self.max_obstacles
            #obsUpdate
            if not self.player.checkObservable(self.current_time):
                addscore = addscore * 1.5
        else:
            addscore = -1

        self.player.score = self.player.score + int(addscore*10)
        return int(addscore*10)

    def initializeNewGame(self):
        self.gamelog.game_type = self.game_type
        self.game_mode = 'InPlay'
        self.display_score = 0
        self.old_score = 0
        self.current_time = time.time()
        self.player.newGameInit(self.current_time)
        
        print(f'Attempt {self.player.attempt}')

        if self.game_type == 'Curriculum' and self.player.attempt <= 40:
            self.initializeGameType(**self.game_params[str(self.player.attempt)])
            self.player.resetObsTime(self.game_params[str(self.player.attempt)]['max_obs_time'],
                                        self.game_params[str(self.player.attempt)]['max_unobs_time'])

        self.newobstacle_count()
        self.curr_obstacle = []
        while len(self.curr_obstacle) < self.obstacle_count:
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                self.current_time, self.exploding_perc, \
                                                self.velocity_max, self.velocity_min, \
                                                self.acceleration, self.theta_max, self.theta_min,\
                                                self.max_obs_time, self.max_unobs_time))

    def displayDefault(self, saveframe = False):
        ret, frame = self.video_capture.read()
        rawframe = cv2.flip(frame, 1)
        frame = self.videoFrameToPyGameDisplay(frame)
        self.game_display.fill([0,0,0])
        self.game_display.blit(frame, (self.frame_offset,0))
        
        if self.game_mode == None:
            pygame.draw.rect(self.game_display, self.textcolor["gray"], self.start_button)
            self.messageDisplay("Start Game", (self.start_button.center), self.textcolor["black"], textsize_type = "small")
            pygame.draw.rect(self.game_display, self.textcolor["gray"], self.calibrate_button)
            self.messageDisplay("Calibrate", (self.calibrate_button.center), self.textcolor["black"], textsize_type = "small")
        
        pygame.display.update()

    def updateGamelog(self):
        self.gamelog.startPlayerLine(self.current_time, self.player, (self.obstacle_count + 1), self.game_type)
        self.gamelog.addObstacleLine(self.curr_obstacle)
        self.gamelog.writeLogLine()
        self.gamelog.clearLogLine()

    def calibrateGame(self):
        call_time = time.time()

        print('Calibration Start')
        while (time.time() - call_time) < self.wait_time:
            self.saveframe = False
            self.displayDefault()
            text_center = ((self.frame_size_tuple[0]/2) + self.frame_offset,(self.frame_size_tuple[1]/2))
            self.messageDisplay(f"{self.wait_time - int(time.time() - call_time)}", text_center, self.textcolor["black"])
            pygame.display.update()

        call_time = time.time()
        while (time.time() - call_time) < self.wait_time:
            self.saveframe = True
            self.gamelog.frame_id += 1
            self.displayDefault()
            text_center = ((self.frame_size_tuple[0]/2) + self.frame_offset,(self.frame_size_tuple[1]/2))
            self.messageDisplay(f"{self.wait_time - int(time.time() - call_time)}", text_center, self.textcolor["green"])
            pygame.display.update()

        print('Calibration End')

        self.saveframe = False

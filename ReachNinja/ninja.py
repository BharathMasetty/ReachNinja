import math
import numpy as np
from marker import Marker
from player import Player
from obstacles import Obstacle
import time
import os
import shutil
import datetime
# import logging
import pygame
import os
import sys
import copy
from operator import itemgetter 
from ninjaParams import target_task,test_task, marker_source_tasks
from gym import spaces
import gym
import datetime
import logging
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class gameState:
    def __init__(self, maxBlue, maxBlack):
        
        self.maxBlue = maxBlue
        self.maxBlack = maxBlack
        self.playerPos =np.empty(2)
        self.playerVel = np.empty(2)
        self.playerAcc = np.empty(2)
        self.bluePos = np.ones((self.maxBlue, 2))*1000
        self.blueVel = np.zeros((self.maxBlue, 2))*1000
        self.blackPos = np.ones((self.maxBlack, 2))*1000
        self.blackVel = np.zeros((self.maxBlack, 2))*1000


class Ninja(gym.Env):

    def __init__(self, params, action_type='cont', log=False, save_path = '../results/post_training_logs/RandomAgent/', actionNoise=3):
        super(Ninja, self).__init__()

        # Need State DIm, action Dims
        self.actionNoise = actionNoise
        self.log_name = 0
        self.params = params
        self.mode = params.mode
        self.termination = params.termination
        self.actionChoice = params.actionChoice
        self.isStatic = params.isStatic
        self.isStationary = params.isStationary
        self.numBlueObstacles = params.numBlue
        self.numBlackObstacles = params.numBlack
        self.visualization = params.visualization
        self.width = params.width
        self.height = params.height
        self.blue_gen =  RandomState(MT19937(SeedSequence(0)))
        self.black_gen =  RandomState(MT19937(SeedSequence(1)))        
        self.initializeGameFrame()
        #self.actionSpace = np.array([-3*params.actionStep, -params.actionStep, 0, params.actionStep, 3*params.actionStep])
        self.actionSpace = np.array([-params.actionStep, 0, params.actionStep])
        self.positionNorm = np.ones(2)
        self.velNorm = np.ones(2)
        self.accNorm = np.asarray([params.actionStep, params.markerAcc])
        self.observation = gameState(self.params.maxBlue, self.params.maxBlack)
        self.defaultPos = 0
        self.defaultVel = 0 
        self.acceleration = np.array([0, params.markerAcc])
        self.timeStep = 0
        self.maxNumSteps = params.maxNumSteps
        # state = self.getState()
        state = np.empty(44)        
        self.state_dimension = state.size
        self.action_dimension = self.actionSpace.size**2
        self.maxAction = params.maxAction
        self.actionMap()
        self.action_type = action_type
        self.log = log
        self.save_path = save_path
        self.max_possible_score = 0
        self.min_possible_score = 0
        # logging.basicConfig(level=logging.INFO)
        self.player_velocities_log = np.empty((self.maxNumSteps,2))
        self.rewards_log = np.empty(self.maxNumSteps)
        self.initialize(params)
        self.save_path = save_path
        print("NOISE LEVEL: ", self.actionNoise)

    # Initialiaies the game based on the passed task parameters
    def initialize(self, params):
        
        # Game Parameters
        if self.isStationary:
            self.velocity_max = 0.0
            self.velocity_min = 0.0
            self.acceleration = np.array([0, 0])
        else:
            self.velocity_max = params.velMax
            self.velocity_min = params.velMin
            self.acceleration = np.array([0, params.markerAcc])

        self.theta_max = params.thetaMax
        self.theta_min = params.thetaMin

        # Other Parameters
        self.exploding_perc = 0.33
        self.maxUnobsTime = 0.5
        self.maxObsTime = 1 
        self.min_obstacles = 1
        self.scalingFactor = 10
        self.tau = 0.01 # Seconds between updates
        self.tauObs = 0.01
        self.episodeCount = 0
        self.noMarkers = False
        if self.numBlueObstacles == 0 and self.numBlackObstacles == 0:
            self.noMarkers = True
        
        self.lastDistance = 2000
        # Pygame Initialization
        pygame.init()
        self.blue = 0  
        self.black = 0  
        pygame.font.init()
        all_fonts = pygame.font.get_fonts()
        self.textsize ={"large": pygame.font.SysFont(all_fonts[0], 200),
                        "medium": pygame.font.SysFont(all_fonts[0], 100),
                        "mid":pygame.font.SysFont(all_fonts[0],60),
                        "small": pygame.font.SysFont(all_fonts[0], 40),
                        "tiny": pygame.font.SysFont(all_fonts[0], 10)}

        self.textcolor = {"gray":   (100,100,100),
                          "white":  (255,255,255),
                          "black":  (0,0,0),
                          "red":    (255,0,0),
                          "green":  (0,255,0),
                          "blue":   (0,0,255)}

        # self.reset()
        self.setnormalizationConstants()        
        clock = pygame.time.Clock()
        
        # Initializing Gym Spaces
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(self.action_dimension)
        elif self.action_type == 'cont':
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # print(self.action_space)

        obs_low = np.ones(self.state_dimension)*-1
        obs_high = np.ones(self.state_dimension)*1
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        #self.reward_range = (-np.inf, np.inf)

    # Takes a single step in the game
    def step(self, action):
        
        # self.action = action
        # print(action)
        # self.action = action*self.params.maxAction
        # velocityChange = self.action - self.observation.playerVel
        # self.observation.playerVel = self.action
        velocityChange = action*self.params.actionStep
        # print(action)
        # Signal dependent noise
        actionNoise = np.multiply(np.array(velocityChange),np.random.normal(loc=0, scale=self.actionNoise, size=(2)))
        # print(actionNoise, velocityChange)
        velocityChange += actionNoise
      
        # print(velocityChange)
        self.observation.playerVel += velocityChange
        self.observation.playerVel = np.clip(self.observation.playerVel, -self.maxAction, self.maxAction)
        self.observation.playerAcc = velocityChange
        # print(self.observation.playerAcc)
        positionChange = self.observation.playerVel*self.tau
        pos = self.observation.playerPos + positionChange
        
        
        # Keeping within frame
        self.observation.playerPos = np.clip(pos, [0,0], self.original_frame_tuple)
        # x_pos = max(0, min(pos[0], self.original_frame_tuple[0]))
        # y_pos = max(0, min(pos[1], self.original_frame_tuple[1]))
        
        self.stepReward = 0
        blueDistanceReward = 0
        
        # Capping velocity at walls
        if self.observation.playerPos[0] in [0, self.original_frame_tuple[0]]: 
            self.observation.playerVel[0] = 0.0
            self.observation.playerAcc[0] = 0.0
            #self.stepReward += -0.1   
        if self.observation.playerPos[1] in [0, self.original_frame_tuple[1]]: 
            self.observation.playerVel[1] = 0.0
            self.observation.playerAcc[1] = 0.0
            #self.stepReward += -0.1   

        # Updating player location
        self.player.loc = self.observation.playerPos
        
        # Updating obstacle situation
        regularIndex = 0;
        bombIndex = 0
        for i in range(len(self.curr_obstacle)):
        
            self.curr_obstacle[i].updatePosition(self.player.loc, self.player.radius, self.params.magneticCoeff)\

            if self.curr_obstacle[i].checkCollision(self.player, self.scaling_factor):
                addscore = self.updateScore(self.curr_obstacle[i])
                self.stepReward += addscore
                self.curr_obstacle[i].doReplace = True
                self.display_score = 1
            
            elif not self.curr_obstacle[i].inframe:
                self.curr_obstacle[i].doReplace = True
            
            if self.curr_obstacle[i].doReplace:
                isReg = True
                mag = self.params.isBlueMagnetic
                self.blue += 1
                gen = self.blue_gen
                if self.curr_obstacle[i].obstacle_type == 'Exploding':
                    isReg = False
                    gen = self.black_gen
                    mag = self.params.isBlackMagnetic
                    self.blue -= 1
                    self.black += 1

                self.curr_obstacle[i] = Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                        self.tauObs, self.exploding_perc, \
                                                        self.velocity_max, self.velocity_min, \
                                                        self.acceleration, self.theta_max, self.theta_min,isRegular=isReg, \
                                                        isStatic=self.isStatic, isMagnetic=mag, rand_gen=gen)
            
                if isReg: self.max_possible_score += self.updateScore(self.curr_obstacle[i])
                else: self.min_possible_score += self.updateScore(self.curr_obstacle[i])

            if self.curr_obstacle[i].obstacle_type == 'Regular':
                self.observation.bluePos[regularIndex,:] = self.curr_obstacle[i].loc
                self.observation.blueVel[regularIndex,:] = self.curr_obstacle[i].velocity
                regularIndex += 1

            elif self.curr_obstacle[i].obstacle_type == 'Exploding':
                self.observation.blackPos[bombIndex,:] = self.curr_obstacle[i].loc
                self.observation.blackVel[bombIndex,:] = self.curr_obstacle[i].velocity
                bombIndex += 1
        
        self.n_blue = regularIndex+1
        self.n_black = bombIndex+1

        # update player score
        speed_score = self.getVelocityReward()
        
        state = self.getState()
        
        if self.log:
            # logging.info(list(np.append([self.observation.playerVel[0], self.observation.playerVel[1], self.stepReward], state)))
            self.player_velocities_log[self.timeStep] = np.array([self.observation.playerVel[0], self.observation.playerVel[1]])
            self.rewards_log[self.timeStep] = self.stepReward

        # Updating episodic reward
        

        # Updateing time Step count
        self.timeStep += 1
        if self.timeStep == self.maxNumSteps and self.termination == 'Time': 
            self.done = True
            self.episodeCount += 1

        # print(state[-1])
        if self.visualization: self.updateVisualization() 
        # self.stepReward += self.get_distance_reward(state)
        self.player.score = self.player.score + self.stepReward 
        self.reward = self.player.score
        reward = self.stepReward 
        done = self.done

        if done and self.log:
            self.save_logs()

        return state, reward, done, {}

    def save_logs(self):
        name = str(self.log_name)
        np.savez_compressed(self.save_path+name,
            velocities=self.player_velocities_log,
            rewards = self.rewards_log,
            num_blue_markers=self.blue,
            num_black_markers=self.black,
            max_score=self.max_possible_score,
            min_score=self.min_possible_score)



    def createNewObstacle(self, isRegular):
        newObstacle = Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                        self.tauObs, self.exploding_perc,\
                                                        self.velocity_max, self.velocity_min, \
                                                        self.acceleration, self.theta_max, self.theta_min, isRegular)
        return newObstacle
    
    def updateVisualization(self):
        
        self.game_display.fill([255,255,255])
        
        pygame.draw.rect(self.game_display, (0,0,0), (0,0,self.frame_offset, self.frame_size_tuple[1]))
        pygame.draw.rect(self.game_display, (0,0,0), (self.frame_offset + self.frame_size_tuple[0],0,self.frame_offset, self.frame_size_tuple[1]))

        playerloc = (int(self.player.loc[0]/self.width_ratio) + self.frame_offset, int(self.player.loc[1]/self.height_ratio))
        self.gameplayerloc = playerloc

        pygame.draw.circle(self.game_display, self.player.marker_color, playerloc, int(self.scaling_factor*self.player.radius))

        for o in self.curr_obstacle:
            obstacleloc = (int(o.loc[0]/self.width_ratio) + self.frame_offset, int(o.loc[1]/self.height_ratio))
            pygame.draw.circle(self.game_display, o.marker_color, obstacleloc, int(self.scaling_factor*o.radius))
        
        self.messageDisplay(f"{int(self.player.score)}", (self.frame_offset/2, self.frame_size_tuple[1]/2), self.textcolor["gray"], "tiny")
        text_rect = self.getTextRect(f"{int(self.player.score)}")
        self.messageDisplay("Return", (self.frame_offset/2, self.frame_size_tuple[1]/2 - text_rect[3]), self.textcolor["gray"], "tiny")

        ## FRAME RATE HERE
        disp_frames = 20
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

    
    def run(self):
        self.reset()
        while not self.done:
            act = self.action_space.sample()
            state, r, _, _ = self.step(act)
 

    
    # resets the game to Initialization stage
    def reset(self):
        self.oldScore = 0
        self.displayScore = 0
        self.obstacleCount = 0
        self.timeStep = 0
        self.max_possible_score = 0
        self.min_possible_score = 0
        # print("Num blue markers: ", self.blue ,"num black markers: ",self.black)
        self.blue=0
        self.black = 0
        self.player_velocities_log = np.empty((self.maxNumSteps, 2))
        self.rewards_log = np.empty(self.maxNumSteps)
        self.lastDistance = 2000
        self.player = Player((self.original_frame_tuple[1], self.original_frame_tuple[0]), 0, np.zeros((2,2)), "Agent")
        # MDP variables
        self.reward = 0
        self.observation = gameState(self.params.maxBlue, self.params.maxBlack)
        self.initializeNewGame()
        self.observation.playerPos = self.player.loc
        self.observation.playerVel = np.zeros(2)
        self.observation.playerAcc = np.zeros(2)
        self.done = False
        newState = self.getState()
        self.blue_gen =  RandomState(MT19937(SeedSequence(0)))
        self.black_gen =  RandomState(MT19937(SeedSequence(1)))  
        self.log_name+=1

        return newState
    
    # keyboard gameplay
    def get_keys_to_action(self):
        
        action = [1,1]
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key ==  pygame.K_LEFT: 
                    action[0] = 0
                    print("Left Key pressed")
                if event.key == pygame.K_RIGHT: 
                    action[0] = 2
                    print("Right Key pressed")
                if event.key == pygame.K_UP: 
                    action[1] = 0
                    print("UP Key pressed")
                if event.key == pygame.K_DOWN: 
                    action[1] = 2
                    print("Down Key pressed")
                    
            if event.type == pygame.KEYUP:
                if event.key ==  pygame.K_LEFT: action[0] = 1
                if event.key ==  pygame.K_RIGHT: action[0] = 1
                if event.key ==  pygame.K_UP: action[0] = 1
                if event.key ==  pygame.K_DOWN: action[0] = 1
                
        return action

    def initializeGameFrame(self):
        
        self.scale = 0.80
        screen_width = int(self.width/self.scale)
        # screen_width = self.width
        screen_height = self.height
        
        frame_width = int(self.width)
        frame_height = int(screen_height)
        
        self.screen_size_tuple = (screen_width, screen_height)
        self.original_frame_tuple = (frame_width, frame_height)

        # print(self.screen_size_tuple)
        # print(self.original_frame_tuple)
        self.aspectratio = self.original_frame_tuple[0]/self.original_frame_tuple[1]#frame_width/frame_height
        self.frame_size_tuple = (int(self.screen_size_tuple[1]*self.aspectratio),self.screen_size_tuple[1])
        self.width_ratio = self.original_frame_tuple[0]/(self.screen_size_tuple[1]*self.aspectratio)
        self.height_ratio = self.original_frame_tuple[1]/self.screen_size_tuple[1]
        self.frame_offset = int((self.screen_size_tuple[0] - int(self.screen_size_tuple[1]*self.aspectratio))/2)
        self.scaling_factor = 1 
        
        if self.visualization:
            self.game_display = pygame.display.set_mode(self.screen_size_tuple)
            self.player_id_textbox = pygame.Rect(100, 100, 50, 50)
            pygame.display.set_caption('Reach Ninja ' + 'Noise Level: ' + str(self.actionNoise))
    
    def setnormalizationConstants(self):
        self.positionNorm = np.asarray(self.original_frame_tuple)
        self.velNorm = np.asarray([abs(self.maxAction), abs(self.maxAction)])
        self.accNorm = np.asarray([abs(self.maxAction), abs(self.maxAction)])
            
        
    
    def initializeNewGame(self):
        #self.gamelog.game_type = self.game_type
        #self.game_mode = 'InPlay'
        self.display_score = 0
        self.old_score = 0
        self.current_time = time.time()
        self.player.newGameInit(self.current_time)
        self.curr_obstacle = []
        for i in range(self.numBlueObstacles):
            
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                self.tauObs, self.exploding_perc, \
                                                self.velocity_max, self.velocity_min, \
                                                self.acceleration, self.theta_max, self.theta_min,isRegular=True, isStatic=self.isStatic, isMagnetic=self.params.isBlueMagnetic,
                                                rand_gen=self.blue_gen))
        for i in range(self.numBlackObstacles):
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                self.tauObs, self.exploding_perc, \
                                                self.velocity_max, self.velocity_min, \
                                                self.acceleration, self.theta_max, self.theta_min,isRegular=False, isStatic=self.isStatic, isMagnetic=self.params.isBlackMagnetic,
                                                rand_gen=self.black_gen))

        regularIndex = 0;
        bombIndex = 0    
        for o in self.curr_obstacle:

            if o.obstacle_type == 'Regular':
                self.observation.bluePos[regularIndex,:] = o.loc
                self.observation.blueVel[regularIndex,:] = o.velocity
                regularIndex += 1
                self.max_possible_score += self.updateScore(o)

            elif o.obstacle_type == 'Exploding':
                self.observation.blackPos[bombIndex,:] = o.loc
                self.observation.blackVel[bombIndex,:] = o.velocity
                self.min_possible_score += self.updateScore(o)
                bombIndex += 1 
   

    def getVelocityReward(self):
        speed_change = np.linalg.norm(self.observation.playerAcc)
        return -1e-4*speed_change

    def get_distance_reward(self,state):
        if (state[42] >= 0.95 or state[43]>= 0.95):
            # print("Penalized for touching wall")
            return -1.0
        else:
            return 0.0


    def updateScore(self, marker):
        if marker.obstacle_type == 'Regular':
            addscore = 1
            addscore =  addscore*np.linalg.norm(marker.velocity)/(marker.velocity_scale*self.velocity_min)
            addscore = (addscore+0.8)*10
        else:
            addscore = -10
        return float(addscore)

    def textObjects(self, text, font, color):
        text_surface = font.render(text, True, color)
        return text_surface, text_surface.get_rect()

    def getTextRect(self, text, textsize_type = 'medium', color = (0,0,0)):
        font = self.textsize[textsize_type]
        text_surface = font.render(text, True, color)
        return text_surface.get_rect()
    
    def messageDisplay(self, text, text_center, color, textsize_type = "tiny"):

        text_surf, text_rect = self.textObjects(text, self.textsize[textsize_type], color)
        text_rect.center = text_center
        self.game_display.blit(text_surf, (text_rect[0], text_rect[1]))


    def getState(self):
        
        bluePositions, blackPositions, blueVelocities, blackVelocities = self.getRelPos()
        bluePositions[self.numBlueObstacles:, :] = 0.
        blueVelocities[self.numBlueObstacles:, :] = 0.
        blackPositions[self.numBlackObstacles:, :] = 0.
        blackVelocities[self.numBlackObstacles:, :] = 0.
        acceleration = self.getRelAcc()
        center = 0.5*np.asarray(self.original_frame_tuple)
        distance_from_center = np.abs(center-self.player.loc)/center
        # print(distance_from_center)
        # distance_from_center = np.linalg.norm(self.player.loc - 0.5*np.asarray(self.original_frame_tuple))/np.linalg.norm(0.5*np.asarray(self.original_frame_tuple)) 
        
        s = np.append(bluePositions, blueVelocities)
        s = np.append(s, blackPositions)
        s = np.append(s, blackVelocities)
        s = np.append(s, acceleration)
        s = np.append(s, distance_from_center)
        return s

    def getRelPos(self):
        playerPos = self.observation.playerPos

        self.blueSortIndices=[]
        self.blackSortIndices = []
        
        relBlueData = np.empty((self.observation.bluePos.shape[0], 5))
        relBlackData = np.empty((self.observation.blackPos.shape[0], 5))

        relBlueData[:, 1:3] = np.divide(self.observation.bluePos - self.observation.playerPos, self.positionNorm)
        relBlackData[:, 1:3] = np.divide(self.observation.blackPos - self.observation.playerPos, self.positionNorm)
        relBlueData[:, 3:] =  np.divide(self.observation.blueVel - self.observation.playerVel, self.velNorm)
        relBlackData[:, 3:] =  np.divide(self.observation.blackVel - self.observation.playerVel, self.velNorm)
        # relBluePos = np.divide(self.observation.bluePos - playerPos, self.positionNorm)
        # relBlackPos = np.divide(self.observation.blackPos - playerPos, self.positionNorm)
        relBlueData[:, 0] = np.linalg.norm(relBlueData[:, 1:3], axis=1)
        relBlackData[:, 0] = np.linalg.norm(relBlackData[:, 1:3], axis=1)
               
        sortedBlueData = np.array(sorted(relBlueData, key= lambda x: x[0]))
        sortedBlackData = np.array(sorted(relBlackData, key= lambda x: x[0]))


        return sortedBlueData[:, 1:3], sortedBlackData[:, 1:3], sortedBlueData[:, 3:], sortedBlackData[:, 3:]
    
    
    def getRelAcc(self):
        playerAcc = self.observation.playerAcc
        relAcc = np.divide(self.acceleration-playerAcc, [self.params.actionStep, self.params.actionStep])
        return relAcc

    def getActionSpace(self):
        return np.array([1,0,-1]), self.action.size

    def render(self):
        if not self.visualization:
            self.visualzation = True
            self.initializeGameFrame()
    
    def actionMap(self):
        actionDict = {}
        numActions = len(self.actionSpace)
        for actIndex1 in range(numActions):
            for actIndex2 in range(numActions):
                actionDict[numActions*actIndex1+actIndex2] = [actIndex1, actIndex2]
        
        self.getAction = actionDict

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    task = marker_source_tasks[-1]
    for i in range(8):
        print(marker_source_tasks[i].numBlue, marker_source_tasks[i].numBlack,  marker_source_tasks[i].magneticCoeff)
    task.visualization=True
    for i in range(10):
        game = Ninja(task,  log=False, actionNoise=4)
        game.run()
        time.sleep(10.0)

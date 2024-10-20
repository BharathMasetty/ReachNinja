import math
import numpy as np
from marker import Marker
from player import Player
from obstacles import Obstacle
import time
import os
import shutil
import datetime
import logging
import pygame
import os
import sys
import copy
from operator import itemgetter 
from ninjaParams import S0Params, S1Params, S2Params, S3Params, S4Params, S5Params
from gym import spaces
import gym
import datetime
import logging
from tqdm import tqdm

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
        self.bluePos = np.ones((self.maxBlue, 2))*100
        self.blueVel = np.zeros((self.maxBlue, 2))*100
        self.blackPos = np.ones((self.maxBlack, 2))*100
        self.blackVel = np.zeros((self.maxBlack, 2))*100


class Ninja(gym.Env):

    def __init__(self, params, action_type='discrete', log=False, save_path = '../results/post_training_logs/RandomAgent/', data_path='../results/imitation_learning_m_turk/Player_3_10-1-121_12-28-11.npz'):
        super(Ninja, self).__init__()

        # Need State DIm, action Dims
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
        state = self.getState()
        self.state_dimension = state.size
        self.action_dimension = self.actionSpace.size**2
        self.maxAction = params.maxAction
        self.actionMap()
        self.action_type = action_type
        self.log = log
        self.save_path = save_path
        self.max_possible_score = 0
        self.min_possible_score = 0
        logging.basicConfig(level=logging.INFO)
        self.initialize(params)
        self.save_path = save_path
        data = np.load(data_path)
        self.positions = data['human_positions']
        self.states = data['human_states']
        print(self.positions.shape)
        self.num_positions = self.positions.shape[0]
        self.current_position_idx = 0 


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

        self.reset()
        self.setnormalizationConstants()        
        clock = pygame.time.Clock()
        
        # Initializing Gym Spaces
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(self.action_dimension)
        elif self.action_type == 'cont':
            self.action_space = spaces.Box(low=-self.params.actionStep, high=self.params.actionStep, shape=(2,), dtype=np.float32)

        print(self.action_space)

        obs_low = np.ones(self.state_dimension)*-1
        obs_high = np.ones(self.state_dimension)*1
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        #self.reward_range = (-np.inf, np.inf)

    # Takes a single step in the game
    def step(self, action):
        
        
        current_pos  = self.positions[self.current_position_idx]
        # print(self.states[self.current_position_idx])

        num_blue = int(current_pos[22])
        num_black = int(current_pos[23])
        # print(num_blue, num_black)

        self.curr_obstacle = []

        for i in range(num_blue):
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                        self.tauObs, self.exploding_perc, \
                                                        self.velocity_max, self.velocity_min, \
                                                        self.acceleration, self.theta_max, self.theta_min,isRegular=True, isStatic=self.isStatic, isMagnetic=False))
            self.curr_obstacle[-1].loc =  np.array(current_pos[2*i+2:2*i+4])
            # print(self.curr_obstacle[-1].loc)
        
        black_start = 12
        for i in range(num_black):
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                        self.tauObs, self.exploding_perc, \
                                                        self.velocity_max, self.velocity_min, \
                                                        self.acceleration, self.theta_max, self.theta_min,isRegular=False, isStatic=self.isStatic, isMagnetic=False))
            
            self.curr_obstacle[-1].loc =  np.array(current_pos[2*i+black_start:2*i+black_start+2])
            # print(self.curr_obstacle[-1].loc)

        self.current_position_idx += 1

        self.player.loc = np.array(current_pos[0:2])
        # print(self.player.loc)

        if self.current_position_idx == self.num_positions-1:
            self.done = True

        print("time, ", current_pos[-1])


        self.updateVisualization()

        return self.done


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
     
        # for i in tqdm(range(self.num_positions)):
        #     act = self.action_space.sample()
        #     _ = self.step(act)
        while not self.done:
            # for event in pygame.event.get():
            #     if event.type == pygame.KEYDOWN:
            time.sleep(0.01)
            act = self.action_space.sample()
            _ = self.step(act)
                                
            
    
    # resets the game to Initialization stage
    def reset(self):
       self.oldScore = 0
       self.displayScore = 0
       self.obstacleCount = 0
       self.timeStep = 0
       self.max_possible_score = 0
       self.min_possible_score = 0
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
        
       name = str(datetime.datetime.now())
       if self.log:
           fileh = logging.FileHandler(self.save_path+name+'.log', 'a')
           formatter = logging.Formatter('%(message)s')
           fileh.setFormatter(formatter)
           log = logging.getLogger()
           for hdlr in log.handlers[:]:
               log.removeHandler(hdlr)
           
           log.addHandler(fileh)
           
           logging.info(str(self.params.paramName))
           logging.info("player vx, player vy, stepreward, blue velocities, black velocities")

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
        screen_height = self.height
        
        frame_width = int(self.width)
        frame_height = int(screen_height)
        
        self.screen_size_tuple = (screen_width, screen_height)
        self.original_frame_tuple = (frame_width, frame_height)

        print(self.screen_size_tuple)
        print(self.original_frame_tuple)
        self.aspectratio = self.original_frame_tuple[0]/self.original_frame_tuple[1]#frame_width/frame_height
        self.frame_size_tuple = (int(self.screen_size_tuple[1]*self.aspectratio),self.screen_size_tuple[1])
        self.width_ratio = self.original_frame_tuple[0]/(self.screen_size_tuple[1]*self.aspectratio)
        self.height_ratio = self.original_frame_tuple[1]/self.screen_size_tuple[1]
        self.frame_offset = int((self.screen_size_tuple[0] - int(self.screen_size_tuple[1]*self.aspectratio))/2)
        self.scaling_factor = 1 
        
        if self.visualization:
            self.game_display = pygame.display.set_mode(self.screen_size_tuple)
            self.player_id_textbox = pygame.Rect(100, 100, 50, 50)
            pygame.display.set_caption('Reach Ninja')
    
    def setnormalizationConstants(self):
        self.positionNorm = np.asarray(self.original_frame_tuple)
        if self.actionChoice == 'acceleration':
            self.velNorm = np.asarray([abs(self.maxAction), abs(self.maxAction)])
        elif self.actionChoice == 'jerk': 
            self.velNorm = np.asarray([700, 700])
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
                                                self.acceleration, self.theta_max, self.theta_min,isRegular=True, isStatic=self.isStatic, isMagnetic=self.params.isBlueMagnetic))
        for i in range(self.numBlackObstacles):
            self.curr_obstacle.append(Obstacle((self.original_frame_tuple[1], self.original_frame_tuple[0]), \
                                                self.tauObs, self.exploding_perc, \
                                                self.velocity_max, self.velocity_min, \
                                                self.acceleration, self.theta_max, self.theta_min,isRegular=False, isStatic=self.isStatic, isMagnetic=self.params.isBlackMagnetic))

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
   

    def updateScore(self, marker):
        if marker.obstacle_type == 'Regular':
            addscore = 1
            if self.params.paramName in ['S1', 'S0', 'S2']:
                addscore = 1000

            if self.params.contReward:
                relativeVelocity = np.linalg.norm(self.observation.blueVel[0,:] - self.observation.playerVel)
                velocityReward   = -relativeVelocity*0.5
                addscore += velocityReward
            
            else:
                addscore =  addscore*(np.linalg.norm(marker.velocity)/marker.velocity_scale)/self.velocity_min
                addscore = (addscore+0.8)*10
        else:
            addscore = -10
            if self.params.paramName in ['S1', 'S0', 'S2']:
                addscore = -1000

        return float(addscore)

    def textObjects(self, text, font, color):
        text_surface = font.render(text, True, color)
        return text_surface, text_surface.get_rect()

    def getTextRect(self, text, textsize_type = 'medium', color = (0,0,0)):
        font = self.textsize[textsize_type]
        text_surface = font.render(text, True, color)
        return text_surface.get_rect()
    
    def messageDisplay(self, text, text_center, color, textsize_type = "tiny"):
        # large_text = pygame.font.Font('freesansbold.ttf',200)
        # medium_text = pygame.font.Font('freesansbold.ttf',100)
        # small_text = pygame.font.Font('freesansbold.ttf',20)

        text_surf, text_rect = self.textObjects(text, self.textsize[textsize_type], color)
        text_rect.center = text_center
        self.game_display.blit(text_surf, (text_rect[0], text_rect[1]))


    def getState(self):
        
        bluePositions, blackPositions, playerPosition = self.getRelPos()
        blueVelocities, blackVelocities = self.getRelVel()
        bluePositions[self.numBlueObstacles:, :] = 0.
        blueVelocities[self.numBlueObstacles:, :] = 0.
        blackPositions[self.numBlackObstacles:, :] = 0.
        blackVelocities[self.numBlackObstacles:, :] = 0.
        acceleration = self.getRelAcc()
        distance_from_center = np.linalg.norm(playerPosition - 0.5*np.asarray(self.original_frame_tuple))/np.linalg.norm(0.5*np.asarray(self.original_frame_tuple)) 
        
        # State Space for stationary static environment
        if self.isStatic == True and self.actionChoice == 'velocity':
            s = np.append(bluePositions, blackPositions)
            s = np.append(s, distance_from_center)

        # State space for non stationary environment
        else:
            s = np.append(bluePositions, blueVelocities)
            s = np.append(s, blackPositions)
            s = np.append(s, blackVelocities)
            s = np.append(s, acceleration)
            s = np.append(s, distance_from_center)
        return s

    def getRelPos(self):
        playerPos = self.observation.playerPos
        relBluePos = copy.deepcopy(self.observation.bluePos)
        blueDistances = []
        self.blueSortIndices=[]
        self.blackSortIndices = []
       
        relBluePos = np.divide(self.observation.bluePos - playerPos, self.positionNorm)
        relBlackPos = np.divide(self.observation.blackPos - playerPos, self.positionNorm)
        blueDistances = np.linalg.norm(relBluePos, axis=1)
        blackDistances = np.linalg.norm(relBlackPos, axis=1)
        
        """
        for index in range(self.numBlueObstacles):
            relBluePos[index,:] = np.divide(self.observation.bluePos[index, :] - playerPos, self.positionNorm)
            blueDistances.append(np.linalg.norm(relBluePos[index,:]))
            
        relBlackPos = copy.deepcopy(self.observation.blackPos)
        blackDistances = []
        for index in range(self.numBlackObstacles):
            relBlackPos[index, :] = np.divide(self.observation.blackPos[index, :] - playerPos, self.positionNorm)
            blackDistances.append(np.linalg.norm(relBluePos[index,:]))
        """
        playerposnorm = playerPos
        sortedRelBluePoses = copy.deepcopy(self.observation.bluePos)
        sortedRelBlackPoses = copy.deepcopy(self.observation.blackPos)
        
        dist = copy.deepcopy(blueDistances)
        for i in range(len(blueDistances)):
            smallestIndex =  np.argmin(dist)
            sortedRelBluePoses[i, :] = relBluePos[smallestIndex, :]
            dist = np.delete(dist, smallestIndex)
            self.blueSortIndices.append(smallestIndex)

        dist = copy.deepcopy(blackDistances)
        for i in range(len(blackDistances)):
            smallestIndex =  np.argmin(dist)
            sortedRelBlackPoses[i, :] = relBlackPos[smallestIndex, :]
            dist = np.delete(dist, smallestIndex)
            self.blackSortIndices.append(smallestIndex)
    
        return sortedRelBluePoses, sortedRelBlackPoses, playerposnorm
    
    def getRelVel(self):
        playerVel = self.observation.playerVel
        relBlueVel = copy.deepcopy(self.observation.blueVel)
        i=0
        for index in self.blueSortIndices:
            relBlueVel[i,:] = np.divide(self.observation.blueVel[index, :] - playerVel, self.velNorm)
            i=i+1

        relBlackVel = copy.deepcopy(self.observation.blackVel)
        i=0
        for index in self.blackSortIndices:
            relBlackVel[i, :] = np.divide(self.observation.blackVel[index, :] - playerVel, self.velNorm)
            i=i+1

        return relBlueVel, relBlackVel
    
    def getRelAcc(self):
        playerAcc = self.observation.playerAcc
        relAcc = playerAcc - self.acceleration
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
    
    # path = '../results/C.npz'
    path = '../results/Imitation_learning_low_speed/EXP/0.npz'
    game = Ninja(S5Params, log=False, action_type='cont', data_path=path)
    game.run()

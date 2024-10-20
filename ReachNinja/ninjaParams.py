

""" 
ReachNinjaParams
setup new source tasks here
"""

class ReachNinjaParams():
    def __init__(self, viz=False, actionChoice='acceleration', 
            actionStep = 100.0, maxAction = 600.0, 
            isStationary = False, isStatic = False,
            numBlue = 3, numBlack = 2, variableMarkers = False,
            velMax = 1.0, velMin = 0.5, thetaMax = -30, thetaMin = -150,
            MarkerAcc = 100, termination = 'Time',
            contReward = False, blueMagnetic = True, blackMagnetic = True,
            paramName = 'target', magneticCoeff = 5000, mode = 'auto',
            targetScore = 1500, height=480, width=640):
        
        self.paramName = paramName

        # Visualization 
        self.visualization = viz
        self.isStationary = isStationary
        self.isStatic = isStatic
        
        # Jerk/ Acceleration / Velocity
        self.actionChoice = actionChoice
        self.actionStep = actionStep
        self.maxAction = maxAction

        # Number of Markers
        self.numBlue = numBlue
        self.numBlack = numBlack
        self.variableMarkers = variableMarkers
        self.maxBlue = 5
        self.maxBlack = 5

        # Minimum and maximum marker velocities
        self.velMin = velMin
        self.velMax = velMax

        # Min and max projectile angle for markers
        self.thetaMin = thetaMin
        self.thetaMax = thetaMax

        # Marker downward acceleration
        self.markerAcc = MarkerAcc

        # Termination condition
        # 'NoRespawn -- terminates after first set of markers go away'
        # 'Time' --  Runs for a fixed number of timesteps
        self.termination = termination
        
        # Magnetic Params
        self.isBlueMagnetic = blueMagnetic
        self.isBlackMagnetic = blackMagnetic

        # Continuous Reward
        self.contReward = contReward

        # Default parameters
        self.mode = mode
        self.maxNumSteps = 3000
        self.magneticCoeff = magneticCoeff
        
        self.targetScore = targetScore

        self.width = width
        self.height = height



## Velcity related source tasks - 2

## Num Marker Related source tasks - 2

## Magnetic effect related source tasks 2 


# Source Task - 0
"""
    Task Description: 
"""
S0Params = ReachNinjaParams(viz = False,
                            actionChoice = 'acceleration',
                            actionStep = 5.0,
                            maxAction = 100.0,
                            isStationary = False,
                            isStatic = False,
                            velMin = 0.05,
                            velMax = 0.05,
                            numBlue = 1,
                            numBlack = 1,
                            variableMarkers = False,
                            thetaMax = -89.9,
                            thetaMin = -90.0,
                            MarkerAcc = 0.,
                            termination = 'NoRespawn',
                            blueMagnetic = True,
                            blackMagnetic = True,
                            contReward = True,
                            paramName='S0',
                            magneticCoeff = -2000,
                            mode = 'auto',
                            targetScore = 900)




# Source Task - 1
"""
    Task Description: 
    One blue and one black marker spawn with equal vertical velocity.
    Markers move up slower then target task. 
    Reward is more continuous to encourage the red marker to hit blue marker slowly.
    Episode ends when red markers either of the obstacles or when the obstacles leave the space.
    No Magnetic effect
"""
S1Params = ReachNinjaParams(viz = False,
                            actionChoice = 'acceleration',
                            actionStep = 10.0,
                            maxAction = 200.0,
                            isStationary = False,
                            isStatic = False,
                            velMin = 0.05,
                            velMax = 0.05,
                            numBlue = 1,
                            numBlack = 1,
                            variableMarkers = False,
                            thetaMax = -89.9,
                            thetaMin = -90.0,
                            MarkerAcc = 0.,
                            termination = 'NoRespawn',
                            blueMagnetic = False,
                            blackMagnetic = False,
                            contReward = True,
                            paramName='S1',
                            mode = 'auto',
                            targetScore = 1000)


# Source Task - 2
"""
    Task Description
    Same as source task 1 except for the following changes:
    Marker move vertically at the velocity observerd target task
    Reward is sparse
"""
S2Params = ReachNinjaParams(viz = False,
                            actionChoice = 'acceleration',
                            actionStep = 10.0,
                            maxAction = 200.0,
                            isStationary = False,
                            isStatic = False,
                            numBlue = 1,
                            numBlack = 1,
                            variableMarkers = False,
                            velMin = 0.5,
                            velMax = 0.5,
                            thetaMax = -89.9,
                            thetaMin = -90.0,
                            MarkerAcc = 100.0,
                            termination = 'NoRespawn',
                            blueMagnetic = False,
                            blackMagnetic = False,
                            contReward = True,
                            magneticCoeff = 1000,
                            paramName='S2',
                            mode = 'auto',
                            targetScore = 1000)



# Source Task - 3
"""
    Terminates after a fixed time
    Non vertical projectile similar to target task
    Magnetic effect on blue marker
"""
S3Params = ReachNinjaParams(viz = False,
                            actionChoice = 'acceleration',
                            actionStep = 10.0,
                            maxAction = 200.0,
                            isStationary = False,
                            isStatic = False,
                            numBlue = 1,
                            numBlack = 1,
                            variableMarkers = False,
                            velMin = 0.1,
                            velMax = 0.3,
                            thetaMax = -30,
                            thetaMin = -150,
                            MarkerAcc = 50.0,
                            termination = 'Time',
                            blueMagnetic = True,
                            magneticCoeff = 1000,
                            blackMagnetic = True,
                            contReward = False,
                            paramName='S3',
                            mode = 'auto',
                            targetScore = 200)


# Source Task - 4

"""
    Same as task 3 with magnetic effect on black markers
"""
S4Params = ReachNinjaParams(viz = True,
                            actionChoice = 'acceleration',
                            actionStep = 10.0,
                            maxAction = 200.0,
                            isStationary = False,
                            isStatic = False,
                            numBlue = 1,
                            numBlack = 1,
                            variableMarkers = False,
                            velMin = 0.1,
                            velMax = 0.5,
                            thetaMax = -30,
                            thetaMin = -150,
                            MarkerAcc = 100.0,
                            termination = 'Time',
                            blueMagnetic = True,
                            blackMagnetic = True,
                            contReward = False,
                            paramName='S4',
                            mode = 'auto',
                            targetScore = 200)


# Source task - 5
"""
    Target task without partial observability
"""
S5Params = ReachNinjaParams(viz =  True,
                            actionChoice = 'acceleration',
                            actionStep = 1000.0,
                            maxAction = 600.0,
                            isStationary = False,
                            isStatic = False,
                            numBlue = 3,
                            numBlack =2,
                            variableMarkers = False,
                            velMin = 0.5,
                            velMax = 1.0,
                            thetaMax = -30,
                            thetaMin = -150,
                            MarkerAcc = 100.0,
                            termination = 'Time',
                            blueMagnetic = False,
                            blackMagnetic = False,
                            contReward = False,
                            paramName='S5',
                            mode = 'auto',
                            targetScore = 500,
                            magneticCoeff=5000.,
                            width = 640,
                            height = 480)



SourceTasks = [S1Params, S2Params, S3Params, S4Params, S5Params]

marker_configs = [[3,2]]
magnetic_configs = [1500, 3000, 5000]
velocity_cofigs = [[0.5, 0.8], [0.8, 1.0]]

## Action Space for CMDP and H-CMDP
cmdp_source_tasks = [ReachNinjaParams() for i in range(6)]
i = 0
for marker in marker_configs:
    for mag in magnetic_configs:
        for vel in velocity_cofigs:
            task = cmdp_source_tasks[i]
            task.numBlue = marker[0]
            task.numBlack = marker[1]
            task.velMin = vel[0]
            task.velMax = vel[1]
            task.magneticCoeff = mag
            print(i, task.velMin, task.velMax, task.magneticCoeff)
            i+=1


target_task = ReachNinjaParams()

baseline_task = ReachNinjaParams()
baseline_task.magneticCoeff = 0.0
baseline_task.isBlackMagnetic = False
baseline_task.isBlueMagnetic = False

cmdp_source_tasks.append(baseline_task)
cmdp_source_tasks.append(target_task)


test_task = ReachNinjaParams()
test_task.actionStep = 100.0


# Marker Tasks
marker_configs = [[1,0], [3, 3]]
magnetic_configs = [1500, 3000, 5000]
marker_source_tasks = [ReachNinjaParams() for i in range(6)]
i=0
for marker in marker_configs:
    # print(marker)
    for magnet in magnetic_configs:
        task = marker_source_tasks[i]
        task.numBlue = marker[0]
        task.numBlack = marker[1]
        task.magneticCoeff = magnet
        # print(i, task.numBlue, task.numBlack, task.magneticCoeff)
        i+=1

marker_source_tasks.append(baseline_task)
marker_source_tasks.append(target_task)


# Final Static CL
def create_source_task(markers=[], mag=0):
    task = ReachNinjaParams()
    task.magneticCoeff = mag
    task.numBlue = markers[0]
    task.numBlack = markers[1]
    return task


final_cl = []
final_cl.append(create_source_task([3, 0], 0))
# final_cl.append(create_source_task([3, 0], 1000))
final_cl.append(create_source_task([4, 1], 3000))
# final_cl.append(create_source_task([3, 2], 41000))
final_cl.append(target_task)

unconstrained_task = ReachNinjaParams()


unconstrained_basetask = ReachNinjaParams(magneticCoeff=0.0)

unconstrained_a = ReachNinjaParams(magneticCoeff=1500.0)

unconstrained_b = ReachNinjaParams(magneticCoeff=3500.0)

actionStepParams = ReachNinjaParams()
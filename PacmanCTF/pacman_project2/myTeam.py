# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
DEF_TIME = 60
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SmartAttackAgent', second = 'SmartDefenderAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class SmartAttackAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def __init__(self, index):

    super().__init__(index)
    self.capacity = 6
    self.lastReturned = 0
    self.defendTime = 0
    self.winningAmount = 0
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """

    '''
    You should change this in your own agent.
    '''

    # Used only for pacman agent hence agentindex is always 0.
    def maxLevel(gameState, depth, alpha, beta,act):

      currDepth = depth + 1
      if gameState.isOver() or currDepth == 1:  # Terminal Test
        return self.evaluate(gameState,act)
      maxvalue = -999999
      actions = gameState.getLegalActions(self.index)
      alpha1 = alpha
      for action in actions:
        successor = gameState.generateSuccessor(self.index, action)
        maxvalue = max(maxvalue, minLevel(successor, currDepth, 1, alpha1, beta,action))
        if maxvalue > beta:
          return maxvalue
        alpha1 = max(alpha1, maxvalue)
      return maxvalue

    # For all ghosts.
    def minLevel(gameState, depth, agentIndex, alpha, beta,act):

      minvalue = 999999
      if gameState.isOver():  # Terminal Test
        return self.evaluate(gameState,act)
      actions = gameState.getLegalActions(agentIndex)
      beta1 = beta
      for action in actions:
        successor = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == 3:
          minvalue = min(minvalue, maxLevel(successor, depth, alpha, beta1,action))
          if minvalue < alpha:
            return minvalue
          beta1 = min(beta1, minvalue)
        elif agentIndex == 1:
          minvalue = min(minvalue, minLevel(successor, depth, agentIndex + 2, alpha, beta1,action))
          if minvalue < alpha:
            return minvalue
          beta1 = min(beta1, minvalue)
      return minvalue

    # Alpha-Beta Pruning
    actions = gameState.getLegalActions(self.index)

    currentScore = -999999
    returnAction = actions[0]
    alpha = -999999
    beta = 999999
    for action in actions:
      nextState = gameState.generateSuccessor(self.index, action)
      # Next level is a min level. Hence calling min for successors of the root.
      score = minLevel(nextState, 0, 1, alpha, beta, action)
      # Choosing the action which is Maximum of the successors.
      if score > currentScore:
        returnAction = action
        currentScore = score
      # Updating alpha value at root.
      if score > beta:

        return returnAction
      alpha = max(alpha, score)

    return returnAction


    #miniMaxScores = [(action,self.evaluate(gameState,action)) for action in actions]

    #bestAction = max(miniMaxScores, key=lambda item:item[1])
    #self.evaluate(gameState,actions[0])


    #return bestAction[0]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor



  def evaluate(self, succ,action):
    """
    Computes a linear combination of features and feature weights
    """

    pos = succ.getAgentPosition(self.index)
    opponents = self.getOpponents(succ)
    pellets = self.getCapsules(succ)
    eatFood = self.getFood(succ)
    enemies = [succ.getAgentState(i) for i in self.getOpponents(succ)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    defenderMode = False
    state = succ.getAgentState(self.index)
    scaredTimes = []
    isChased = False
    isThreatened = False


    if succ.isOver():
      return 100000

    for enemy in opponents:
      scaredTimes.append(succ.getAgentState(enemy).scaredTimer)


    enemyPositions = []
    for enemy in opponents:
      enemyPositions.append(succ.getAgentPosition(enemy))

    enemyGhostPositions = []
    for enemy in opponents:
      if not succ.getAgentState(enemy).isPacman and not succ.getAgentState(enemy).scaredTimer > 8:
        enemyGhostPositions.append(succ.getAgentPosition(enemy))

    enemyInvaderPositions = []
    for enemy in opponents:
      if succ.getAgentState(enemy).isPacman:
        enemyInvaderPositions.append(succ.getAgentPosition(enemy))

    enemyGhostDistances = []
    for enemyPos in enemyGhostPositions:
      distance = self.distancer.getDistance(pos,enemyPos)
      if distance < 7:
        isChased = True
      #if distance < 8:
        #isThreatened = True
      enemyGhostDistances.append(self.distancer.getDistance(pos,enemyPos))

    enemyInvaderDistances = []
    for enemyPos in enemyInvaderPositions:
      enemyInvaderDistances.append(self.distancer.getDistance(pos,enemyPos))

    foodPositions = []
    for i in range(0,eatFood.width):
      for j in range(0,eatFood.height):
        if(eatFood[i][j]) is True:
          foodPositions.append((i,j))


    pelletDistances = []
    for pellet in pellets:
      pelletDistances.append(self.distancer.getDistance(pos,pellet))

    foodDistances = []
    for food in foodPositions:
      foodDistances.append(self.distancer.getDistance(pos,food))


    sumPelletDistances = 0
    for distance in pelletDistances:
      if sum(scaredTimes) > 0:
        sumPelletDistances -= distance
      else:
        sumPelletDistances += distance

    sumFoodDistance = 0
    for distance in foodDistances:
      if distance > 0:
        sumFoodDistance += distance


    features = util.Counter()
    weights = util.Counter()
    if action == Directions.STOP:
      features['stop'] = 1
    else:
      features['stop'] = 0


    if len(enemyGhostDistances) == 0 and not state.isPacman and not min(scaredTimes) > 0:
      defenderMode = True
    if len(enemyInvaderDistances) > 0:
      print(min(enemyInvaderDistances))
      if min(enemyInvaderDistances) < 7 and not state.scaredTimer >= 3:
        defenderMode = True


    if self.red:
      self.winningAmount = succ.getScore()
    else:
      self.winningAmount = -1*succ.getScore()



    if succ.data.timeleft < 200 and self.winningAmount>0:
      defenderMode = True

    weights['stop'] = -90000
    features['invaderDistance'] = 0
    if defenderMode:
      if len(enemyInvaderDistances) > 0:
        features['invaderDistance'] = min(enemyInvaderDistances)*100
    features['succScore'] = -len(self.getFood(succ).asList())
    features['distanceToFood'] = min(foodDistances)
    if len(pelletDistances) > 0:
      features['powerPellet'] = min(pelletDistances)
    features['backHome'] = self.cashIn(pos, succ.getAgentState(self.index), succ, isChased, isThreatened)
    if len(enemyGhostDistances) > 0:
      if not defenderMode:
        features['ghostDistance'] = min(enemyGhostDistances)
    else:
      features['ghostDistance'] = 0
    if isChased:
      weights['ghostDistance'] = 200000000000000000
    if isThreatened:
      weights['ghostDistance'] = 500
    else:
      weights['ghostDistance'] = 50
    weights['invaderDistance'] = -100
    features['deadEnd'] = self.isDeadEnd(succ,pos)
    weights['deadEnd'] = 0
    if isChased:
      weights['deadEnd'] = -9999
    if not defenderMode:
      weights['succScore'] = 4000
    if not defenderMode:
      weights['distanceToFood'] = -100
    else:
      weights['distanceToFood'] = 0
    if not defenderMode:
      weights['powerPellet'] = -5000
    else:
      weights['powerPellet'] = 0
    if isChased:
      weights['powerPellet'] *= 100
    weights['backHome'] = -1
    if isThreatened:
      weights['powerPellet'] *= 50

    features['backHome'] += self.goBack(pos,isChased,isThreatened,succ)


    print("------------")
    print(defenderMode)
    print(features)
    print(weights)
    print("------------")
    return features*weights

  def cashIn(self, pos, agentState, gameState, isChased, isThreatened):
    cap = self.capacity + 1
    if isChased:
      cap = 1
    if isThreatened:
      cap = 3
    if agentState.numCarrying >= cap:
      return self.getMazeDistance(pos,gameState.getInitialAgentPosition(self.index))
    else:
      return 0

  def isDeadEnd(self,gameState, pos):
    wallCount = 0
    if gameState.hasWall(pos[0] - 1, pos[1]):
      wallCount += 1
    if gameState.hasWall(pos[0] + 1, pos[1]):
      wallCount += 1
    if gameState.hasWall(pos[0], pos[1] - 1):
      wallCount += 1
    if gameState.hasWall(pos[0], pos[1] + 1):
      wallCount += 1

    if wallCount >=3:
      return 1
    else:
      return 0

  def goBack(self,pos,isChased,isThreatened,state):
    if isChased:
      return self.getMazeDistance(pos,state.getInitialAgentPosition(self.index)) * 3000000
    elif isThreatened:
      return self.getMazeDistance(pos,state.getInitialAgentPosition(self.index)) * 500
    else:
      return 0

class SmartDefenderAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def __init__(self, index):

    super().__init__(index)
    self.capacity = 2
    self.winningAmount = 0

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """

    '''
    You should change this in your own agent.
    '''

    # Used only for pacman agent hence agentindex is always 0.
    def maxLevel(gameState, depth, alpha, beta):

      currDepth = depth + 1
      if gameState.isOver() or currDepth == 1:  # Terminal Test
        return self.evaluate(gameState)
      maxvalue = -999999
      actions = gameState.getLegalActions(self.index)
      alpha1 = alpha
      for action in actions:
        successor = gameState.generateSuccessor(self.index, action)
        maxvalue = max(maxvalue, minLevel(successor, currDepth, 1, alpha1, beta))
        if maxvalue > beta:
          return maxvalue
        alpha1 = max(alpha1, maxvalue)
      return maxvalue

    # For all ghosts.
    def minLevel(gameState, depth, agentIndex, alpha, beta):

      minvalue = 999999
      if gameState.isOver():  # Terminal Test
        return self.evaluate(gameState)
      actions = gameState.getLegalActions(agentIndex)
      beta1 = beta
      for action in actions:
        successor = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == 3:
          minvalue = min(minvalue, maxLevel(successor, depth, alpha, beta1))
          if minvalue < alpha:
            return minvalue
          beta1 = min(beta1, minvalue)
        elif agentIndex == 1:
          minvalue = min(minvalue, minLevel(successor, depth, agentIndex + 2, alpha, beta1))
          if minvalue < alpha:
            return minvalue
          beta1 = min(beta1, minvalue)
      return minvalue

    # Alpha-Beta Pruning
    actions = gameState.getLegalActions(self.index)
    currentScore = -999999
    returnAction = ''
    alpha = -999999
    beta = 999999
    for action in actions:
      nextState = gameState.generateSuccessor(self.index, action)
      # Next level is a min level. Hence calling min for successors of the root.
      score = minLevel(nextState, 0, 1, alpha, beta)

      # Choosing the action which is Maximum of the successors.
      if score > currentScore:
        returnAction = action
        currentScore = score
      # Updating alpha value at root.
      if score > beta:

        return returnAction
      alpha = max(alpha, score)

    return returnAction


    #miniMaxScores = [(action,self.evaluate(gameState,action)) for action in actions]

    #bestAction = max(miniMaxScores, key=lambda item:item[1])
    #self.evaluate(gameState,actions[0])


    #return bestAction[0]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor




  def evaluate(self, succ):

    pos = succ.getAgentPosition(self.index)
    opponents = self.getOpponents(succ)
    pellets = self.getCapsules(succ)
    myPellets = self.getCapsulesYouAreDefending(succ)
    eatFood = self.getFood(succ)
    enemies = [succ.getAgentState(i) for i in self.getOpponents(succ)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    scaredTimes = []
    stealMode = True
    protectionMode = False


    pelletWeight = 0

    if succ.isOver():
      return 100000

    for enemy in opponents:
      scaredTimes.append(succ.getAgentState(enemy).scaredTimer)


    enemyGhostPositions = []
    for enemy in opponents:
      if not succ.getAgentState(enemy).isPacman:
        enemyGhostPositions.append(succ.getAgentPosition(enemy))

    enemyInvaderPositions = []
    for enemy in opponents:
      if succ.getAgentState(enemy).isPacman:
        enemyInvaderPositions.append(succ.getAgentPosition(enemy))

    if len(invaders) > 0:
      protectionMode = True
      stealMode = False

    foodPositions = []
    for i in range(0,eatFood.width):
      for j in range(0,eatFood.height):
        if(eatFood[i][j]) is True:
          foodPositions.append((i,j))

    enemyGhostDistances = []
    for enemyPos in enemyGhostPositions:
      enemyGhostDistances.append(self.distancer.getDistance(pos,enemyPos))

    enemyInvaderDistances = []
    for enemyPos in enemyInvaderPositions:
      enemyInvaderDistances.append(self.distancer.getDistance(pos,enemyPos))

    pelletDistances = []
    for pellet in pellets:
      pelletDistances.append(self.distancer.getDistance(pos,pellet))

    myPelletDistances = []
    for pellet in myPellets:
      myPelletDistances.append(self.distancer.getDistance(pos,pellet))

    foodDistances = []
    for food in foodPositions:
      foodDistances.append(self.distancer.getDistance(pos,food))

    sumEnemyGhostDistances = 0
    for i in range(0,len(enemyGhostDistances)):
      if scaredTimes[i] >2:
        sumEnemyGhostDistances += enemyGhostDistances[i]

    sumEnemyInvaderDistances = 0
    for i in range(0,len(enemyInvaderDistances)):
      sumEnemyInvaderDistances += enemyInvaderDistances[i]

    sumPelletDistances = 0
    for distance in pelletDistances:
      if sum(scaredTimes) > 0:
        sumPelletDistances -= distance
      else:
        sumPelletDistances += distance

    sumMyPelletDistances = 0
    for distance in myPelletDistances:
      sumMyPelletDistances += distance

    sumFoodDistance = 0
    for distance in foodDistances:
      if distance > 0:
        sumFoodDistance += distance

    for distance in enemyGhostDistances:
      if distance < 9:
        stealMode = False
        protectionMode = True
        break
    else:
      if protectionMode != True:
        stealMode = True
    if self.red:
      self.winningAmount = succ.getScore()
    else:
      self.winningAmount = -1*succ.getScore()
    features = util.Counter()
    weights = util.Counter()
    if succ.data.timeleft < 150 and self.winningAmount>0:
      stealMode = False
      protectionMode = True
    features['succScore'] = -len(self.getFood(succ).asList())
    features['distanceToFood'] = min(foodDistances)
    if len(enemyInvaderDistances) > 0:
      features['invaderDistance'] = min(enemyInvaderDistances)*100
    if len(pelletDistances) > 0:
      features['powerPellet'] = min(pelletDistances)
    features['backHome'] = self.cashIn(pos,succ.getAgentState(self.index),succ)

    if protectionMode:
      features['backHome'] += 100

    weights['succScore'] = 1000

    weights['distanceToFood'] = 0
    if stealMode:
      weights['distanceToFood'] = -100

    weights['invaderDistance'] = -100
    if stealMode:
      weights['powerPellet'] = -100
    weights['backHome'] = -1
    if protectionMode and succ.getAgentState(self.index).isPacman:
      weights['backHome'] *= 1000
    return features*weights

  def cashIn(self, pos, agentState,gameState):
    if agentState.numCarrying >= self.capacity:
      return self.getMazeDistance(pos,gameState.getInitialAgentPosition(self.index))
    else:
      return 0

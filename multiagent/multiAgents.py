# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # we don't want the ghost to stop, so we penalize it
        if action == Directions.STOP:
            score -= 5

        foodList = newFood.asList()
        score = successorGameState.getScore()
        if len(foodList) == 0:
            return score + 10000000


        # Reward for eating food
        oldFoodCount = len(currentGameState.getFood().asList())
        if len(foodList) < oldFoodCount:
            score += 20

        # Using manhattan distance, find the closest food, and add to score inversely proportional to distance
        if foodList is not None:
            minFoodDist = min(manhattanDistance(newPos, f) for f in foodList)
            score += 10 / (minFoodDist + 1)

        # Penalize being close to ghosts, reward being close to scared ghosts
        for ghost in newGhostStates:
            ghost_dist = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer == 0:
                if ghost_dist <= 1:
                    return -99999999999999
                score -= 10 / (ghost_dist) 
            else:
                score += 5 / (ghost_dist + 1)
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()

        def minimaxhelper(agentIndex, depth, state):

            # Base Cases:
            # 1. Game over (win/lose)
            # 2. Reached the maximum search depth
            # 3. No legal actions available for the current agent
            bestValue = None
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            elif not state.getLegalActions(agentIndex):
                return self.evaluationFunction(state)

            if agentIndex == 0:
                bestValue = - 2**63 - 1
                for action in state.getLegalActions(agentIndex):
                    '''
                    # Recursively call minimaxhelper for the next agent (agentIndex + 1 or 1 if it's the first ghost)
                    # The depth remains the same if it's a ghost's turn in the same ply,
                    # or increments if it's the next Pacman's turn in the next ply.
                    '''
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimaxhelper(1, depth, successor)
                    bestValue = max(bestValue, value)
                return bestValue
            else:
                nextAgent = (agentIndex + 1) % numAgents
                if nextAgent == 0:
                    nextDepth = depth + 1 
                else:
                    nextDepth = depth
                bestValue = 2**63 +1
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimaxhelper(nextAgent, nextDepth, successor)
                    bestValue = min(bestValue, value)
                return bestValue

        bestAction = None
        best_Value = - 2**63 - 1
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimaxhelper(1, 0, successor)
            if value > best_Value:
                best_Value = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

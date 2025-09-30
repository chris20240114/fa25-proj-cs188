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
                    Here we recursively call the minimaxhelper for the next agent 
                    (agentIndex + 1 or 1 if it's the first ghost)
                    The depth remains the same if it's a ghost's turn in the same ply,
                    or increments if it's the next Pacman's turn in the next ply.
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


    #Alpha-Beta Pruning pseudocode

    '''
    a = max's best option on path to root
    b = min's best option on path to root
    '''
    '''
    def max-value(state, a, b):
        if state is a terminal node:
            return utility of state
        v = neg infinity
        for each successor of state:
            v = max(v, value(successor, a, b))
            if v > b:
                return v
            a = max(a, v)
        return v

    def min-value(state, a, b):
        if state is a terminal node:
            return utility of state
        v = pos infinity
        for each successor of state:
            v = min(v, value(successor, a, b))
            if v < a:
                return v
            b = min(b, v)
        return v
    '''

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        import math
        INF = math.inf
        NEG_INF = -math.inf

        def max_value(agentIndex, depth, state, a, b):
            v = NEG_INF
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, ab_val((agentIndex + 1) % numAgents, depth + ((agentIndex + 1) // numAgents), successor, a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def min_value(agentIndex, depth, state, a, b):
            v = INF
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, ab_val((agentIndex + 1) % numAgents, depth + ((agentIndex + 1) // numAgents), successor, a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

        def ab_val(agentIndex, depth, state, a, b):

            # checks for win/loss states or if the maximum search depth has been reached, or if agent has no legal actions, returns evalfunc's value in this case
            # determines if the current agentIndex belongs to the maximizing player or a minimizing player to determine whether to call max_value or min_value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            elif not state.getLegalActions(agentIndex):
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                return max_value(agentIndex, depth, state, a, b)
            return min_value(agentIndex, depth, state, a, b)
            
        bestAction = None
        alpha = NEG_INF
        beta = INF
        # choosing the best action for the maximizing player here (pacman)
        # initial call to ab_val for each legal action of the maximizing player (Pacman) and updates alpha and bestAction accordingly
        # iterate through all legal actions to find optimal action using ab_pruning and returns best action found
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            abval = ab_val(1 % numAgents, 0, succ, alpha, beta)
            if abval > alpha or bestAction is None:
                alpha = abval
                bestAction = action

        return bestAction   


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
        numAgents = gameState.getNumAgents()
        import math
        INF = math.inf
        NEG_INF = -math.inf

        def max_value(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            elif not state.getLegalActions(agentIndex):
                return self.evaluationFunction(state)

            # elif agentIndex != 0:
            #     raise Exception("max_value called for a minimizing agent")

            next_agent = (agentIndex + 1) % numAgents
            if next_agent == 0:
                nextDepth = depth + 1
            else: 
                nextDepth = depth

            if agentIndex == 0:
                v = NEG_INF
                for action in state.getLegalActions(agentIndex):
                    succ = state.generateSuccessor(agentIndex, action)
                    val = max_value(next_agent, nextDepth, succ)
                    if val > v:
                        v = val
                return v
            else:
                total = 0
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    total += max_value(next_agent, nextDepth, successor)
                return float(total) / len(state.getLegalActions(agentIndex))

        bestAction = None
        best_Value = NEG_INF
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = max_value(1 % numAgents, 0, successor)
            if value > best_Value or bestAction is None:
                best_Value = value
                bestAction = action
        return bestAction
            



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My function starts with the current game score and then
    adjusts it based on:
        1. The distance to the nearest food pellet (closer is better).
        2. The number of remaining food pellets (fewer is better).
        3. The distance to the nearest non-scared ghost (further is better). A very close
      ghost results in a huge penalty.
        4. The distance to the nearest scared ghost (closer is better, for hunting).
        5. The number of remaining power capsules (fewer is better).
        6. Power capsule strategic value
        7. Win/loss state handling
        8. Food clustering consideration for efficiency
    
    It's sort of similar to the idea we had in the reflex agent, but a lot less arbitrary than that.
    The function should guide Pacman to eat all the food while avoiding ghosts, and to hunt scared ghosts when the opportunity arises
    More commentary in the code itself included.
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    pacman_position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    if food_list:
        food_distances = [util.manhattanDistance(pacman_position, food) for food in food_list]
        min_food_distance = min(food_distances)
        # Closer food increases score significantly
        score += 10.0 / (min_food_distance + 1)
        closest_foods = sorted(food_distances)[:min(3, len(food_distances))]
        avg_close_food_dist = sum(closest_foods) / len(closest_foods)
        score += 5.0 / (avg_close_food_dist + 1)

    score -= 4 * len(food_list)

    # Separate ghosts into scared and active (not scared)
    scared_ghosts = []
    active_ghosts = []
    for ghost in ghost_states:
        if ghost.scaredTimer > 0:
            scared_ghosts.append(ghost)
        else:
            active_ghosts.append(ghost)

    if active_ghosts:
        ghost_distances = [util.manhattanDistance(pacman_position, ghost.getPosition()) 
                          for ghost in active_ghosts]
        min_ghost_distance = min(ghost_distances)
        # Heavily penalize being too close to an active ghost
        if min_ghost_distance <= 1:
            score -= 5000
        elif min_ghost_distance == 2:
            score -= 500
        elif min_ghost_distance <= 3:
            score -= 100
        else:
            score += 2 * min_ghost_distance
    
    # Scared ghost hunting with time awareness
    if scared_ghosts:
        for ghost in scared_ghosts:
            ghost_dist = util.manhattanDistance(pacman_position, ghost.getPosition())
            scared_time = ghost.scaredTimer
            
            # Only hunt if we have enough time to reach the ghost
            if scared_time > ghost_dist:
                # Reward for being close to a scared ghost
                score += 100.0 / (ghost_dist + 1)
                # Extra reward based on how much time is left
                score += scared_time * 2
            else:
                # Penalize if we can't reach the ghost in time
                score -= 10
    if capsules:
        capsule_distances = [util.manhattanDistance(pacman_position, cap) for cap in capsules]
        min_capsule_distance = min(capsule_distances)
        
        # Capsules are valuable when ghosts are nearby
        if active_ghosts:
            closest_ghost_dist = min([util.manhattanDistance(pacman_position, ghost.getPosition()) 
                                     for ghost in active_ghosts])
            if closest_ghost_dist < 5:
                score += 50.0 / (min_capsule_distance + 1)
        # Otherwise, capsules are less critical
    score -= 20 * len(capsules)

    return score
    

# Abbreviation
better = betterEvaluationFunction

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


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        min_ghost_dist = 9999999999
        for i in range(len(newGhostStates)):
            ghost = newGhostStates[i]
            dist = manhattanDistance(ghost.getPosition(), newPos)
            if not newScaredTimes[i] and dist <= 1:
                return -999999999999
            if dist < min_ghost_dist:
                min_ghost_dist = dist

        if min_ghost_dist:
            min_ghost_dist = 2 / min_ghost_dist

        min_dist = 9999999999
        for food in newFood.asList():
            dist = manhattanDistance(food, newPos)
            if dist < min_dist:
                min_dist = dist

        if min_dist:
            min_dist = 1 / min_dist

        return successorGameState.getScore() + min_dist - min_ghost_dist


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        return self.calculate_max(gameState, 0)[1]

    def calculate_max(self, gameState, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), '')

        max = -9999999999
        best_action = None
        for action in gameState.getLegalActions(0):
            nxt_state = gameState.generateSuccessor(0, action)
            value = self.calculate_min(nxt_state, depth + 1, 1)
            if value > max:
                max = value
                best_action = action

        return (max, best_action)

    def calculate_min(self, gameState, depth, index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        min = 9999999999
        for action in gameState.getLegalActions(index):
            nxt_state = gameState.generateSuccessor(index, action)
            if index == gameState.getNumAgents() - 1:
                value = self.calculate_max(nxt_state, depth)[0]
            else:
                value = self.calculate_min(nxt_state, depth, index + 1)

            if value < min:
                min = value

        return min


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.calculate_max(gameState, 0, -9999999999, 9999999999)[1]

    def calculate_max(self, gameState, depth, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), '')

        max = -9999999999
        best_action = None
        for action in gameState.getLegalActions(0):
            nxt_state = gameState.generateSuccessor(0, action)
            value = self.calculate_min(nxt_state, depth + 1, 1, alpha, beta)

            if value > max:
                max = value
                best_action = action

            if max > beta:
                return (max, best_action)

            if max > alpha:
                alpha = max

        return (max, best_action)

    def calculate_min(self, gameState, depth, index, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        min = 9999999999
        for action in gameState.getLegalActions(index):
            nxt_state = gameState.generateSuccessor(index, action)
            if index == gameState.getNumAgents() - 1:
                value = self.calculate_max(nxt_state, depth, alpha, beta)[0]
            else:
                value = self.calculate_min(nxt_state, depth, index + 1, alpha, beta)

            if value < min:
                min = value

            if min < alpha:
                return min

            if min < beta:
                beta = min

        return min


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.calculate_max(gameState, 0)[1]

    def calculate_max(self, gameState, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), '')

        max = -9999999999
        best_action = None
        for action in gameState.getLegalActions(0):
            nxt_state = gameState.generateSuccessor(0, action)
            value = self.calculate_ghost_expectation(nxt_state, depth + 1, 1)
            if value > max:
                max = value
                best_action = action

        return (max, best_action)

    def calculate_ghost_expectation(self, gameState, depth, index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        p = 0
        action_count = len(gameState.getLegalActions(index))
        for action in gameState.getLegalActions(index):
            nxt_state = gameState.generateSuccessor(index, action)
            if index == gameState.getNumAgents() - 1:
                value = self.calculate_max(nxt_state, depth)[0]
            else:
                value = self.calculate_ghost_expectation(nxt_state, depth, index + 1)

            p += value / action_count

        return p


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    min_ghost_dist = 9999999999
    for i in range(len(ghostStates)):
        dist = manhattanDistance(ghostPositions[i], pacmanPosition)
        if not scaredTimers[i] and dist <= 1:
            return -999999999999
        if dist < min_ghost_dist:
            min_ghost_dist = dist

    if min_ghost_dist:
        min_ghost_dist = 2 / min_ghost_dist

    min_dist = 9999999999
    for food in foods.asList():
        dist = manhattanDistance(food, pacmanPosition)
        if dist < min_dist:
            min_dist = dist

    if min_dist:
        min_dist = 1 / min_dist

    capsules = currentGameState.getCapsules()
    caps_num = 0
    if capsules:
        caps_num = 20 / len(capsules)

    scared = 0
    for sc in scaredTimers:
        scared += sc

    return currentGameState.getScore() + min_dist - min_ghost_dist + caps_num \
           - len(foods.asList()) + 5 * scared


# Abbreviation
better = betterEvaluationFunction

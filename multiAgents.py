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

        # this successor's original game score
        score = successorGameState.getScore()

        # move towards nearest food, use reciprocal of distance
        # so that closer food has larger weight
        food_positions = newFood.asList()
        if food_positions:
            distance_to_nearest_food = min(manhattanDistance(newPos, food) for food in food_positions)
            # add 1 for avoiding division by zero
            score += 1.0 / (distance_to_nearest_food + 1)

        for i in range(len(newGhostStates)):
            ghostPos = newGhostStates[i].getPosition()
            distance_to_ghost = manhattanDistance(newPos, ghostPos)
            scared_timer = newScaredTimes[i]
            # if ghost is scared, move towards it, else avoid it
            if scared_timer > 0:
                    score += 1.0 / (distance_to_ghost + 1)
            else:
                if distance_to_ghost < 2:
                    score -= 1000

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
        pacman_legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_score = float('-inf')
        for action in pacman_legal_actions:
            successor_game_state = gameState.generateSuccessor(0, action)
            score = self.recursive_minimax(successor_game_state, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def recursive_minimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legal_actions = state.getLegalActions(agentIndex)
        if agentIndex == 0:
            # max of recursive generator if pacman's turn
            return max(
                self.recursive_minimax(state.generateSuccessor(agentIndex, action), depth, 1) for action in legal_actions)
        else:
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                newDepth = depth - 1
            else:
                newDepth = depth
            # min of recursive generator if ghost's turn
            return min(
                self.recursive_minimax(state.generateSuccessor(agentIndex, action), newDepth, nextAgent) for action in legal_actions
            )


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        pacman_legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_score = float('-inf')
        alpha = float('-inf') # best value max can guarantee throughout recursion
        beta = float('inf') # best value min can guarantee throughout recursion
        for action in pacman_legal_actions:
            successor_game_state = gameState.generateSuccessor(0, action)
            score = self.ab_minimax(successor_game_state, self.depth, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        return best_action

    def ab_minimax(self, state, depth, agentIndex, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legal_actions = state.getLegalActions(agentIndex)
        if agentIndex == 0:
            v = float('-inf')
            for action in legal_actions:
                v = max(v, self.ab_minimax(state.generateSuccessor(agentIndex, action), depth, 1, alpha, beta))
                # pacman finds a value that's greater than smallest value
                # a ghost has encountered so far, so his value will never traverse
                # up the tree
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                newDepth = depth - 1
            else:
                newDepth = depth
            v = float('inf')
            for action in legal_actions:
                v = min(v, self.ab_minimax(state.generateSuccessor(agentIndex, action), newDepth, nextAgent, alpha, beta))
                # ghost finds a value that's less than greatest value
                # pacman has encountered so far, so his value will never traverse
                # up the tree
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

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
        pacman_legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_score = float('-inf')
        for action in pacman_legal_actions:
            successor_game_state = gameState.generateSuccessor(0, action)
            score = self.recursive_expectimax(successor_game_state, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def recursive_expectimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legal_actions = state.getLegalActions(agentIndex)
        if agentIndex == 0:
            # max of recursive generator if pacman's turn
            return max(
                self.recursive_expectimax(state.generateSuccessor(agentIndex, action), depth, 1) for action in legal_actions)
        else:
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                newDepth = depth - 1
            else:
                newDepth = depth
            # average of recursive generator values if ghost's turn
            return sum(
                self.recursive_expectimax(state.generateSuccessor(agentIndex, action), newDepth, nextAgent) for action in legal_actions
            ) / len(legal_actions)

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

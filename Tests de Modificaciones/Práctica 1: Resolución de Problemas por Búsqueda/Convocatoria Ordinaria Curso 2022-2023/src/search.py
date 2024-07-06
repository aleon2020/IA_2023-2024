# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# PSEUDOCÓGIDO UTILIZADO PARA REALIZAR LAS PREGUNTAS 1, 2, 3 Y 4
# (proporcionado en el enunciado de la práctica):
# frontier = {startNode}
# expanded = {}
# while frontier is not empty:
    # node = frontier.pop()
    # if isGoal(node):
        # return path_to_node
    # if node not in expanded:
        # expanded.add(node)
        # for each child of node´s children:
            # frontier.push(child)
# return failed

#______________________________________________________________________________
# QUESTION 1

# PREGUNTA 1: BÚSQUEDA EN PROFUNDIDAD (DFS)

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función depthFirstSearch():
# En primer lugar, se han declarado las siguientes variables:
# frontier: Crea una pila (orden Last In First Out).
# expanded: Almacena los nodos ya expandidos.
# actionList: Almacena las acciones que va realizando el agente.
# Una vez hecho esto, se inserta el nodo inicial en la frontera
# junto a la lista de acciones (actionList) vacía.
# Siempre que la frontera NO esté vacía, se extrae un nodo de la misma.
# Si el nodo extraído es el nodo objetivo, 
# devuelve la lista de acciones realizadas hasta ese momento.
# Si el nodo que hemos extraído NO se había expandido previamente,
# éste es insertado en la lista de nodos expandidos.
# Cada hijo del nodo extraído es insertado en la frontera.
# Si no existe ninguna solución, devuelve una lista vacía.

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** MY CODE STARTS HERE ***"
    frontier = util.Stack()
    expanded = []
    actionList = []
    frontier.push((problem.getStartState(), actionList))
    while not frontier.isEmpty():
        node, actions = frontier.pop()
        if problem.isGoalState(node):
            return actions
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                frontier.push((child[0], actions + [child[1]]))
    return []
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 2

# PREGUNTA 2: BÚSQUEDA EN ANCHURA (BFS)

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función breadthFirstSearch():
# En primer lugar, se han declarado las siguientes variables:
# frontier: Crea una cola (orden First In First Out).
# expanded: Almacena los nodos ya expandidos.
# actionList: Almacena las acciones que va realizando el agente.
# Una vez hecho esto, se inserta el nodo inicial en la frontera
# junto a la lista de acciones (actionList) vacía.
# Siempre que la frontera NO esté vacía, se extrae un nodo de la misma.
# Si el nodo extraído es el nodo objetivo, 
# devuelve la lista de acciones realizadas hasta ese momento.
# Si el nodo que hemos extraído NO se había expandido previamente,
# éste es insertado en la lista de nodos expandidos.
# Cada hijo del nodo extraído es insertado en la frontera.
# Si no existe ninguna solución, devuelve una lista vacía.

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** MY CODE STARTS HERE ***"
    frontier = util.Queue()
    expanded = []
    actionList = []
    frontier.push((problem.getStartState(), actionList))
    while not frontier.isEmpty():
        node, actions = frontier.pop()
        if problem.isGoalState(node):
            return actions
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                frontier.push((child[0], actions + [child[1]]))
    return []
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 3

# PREGUNTA 3: BÚSQUEDA DE COSTE UNIFORME (UCS)

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función uniformCostSearch():
# En primer lugar, se han declarado las siguientes variables:
# frontier: Crea una cola con prioridad (orden natural).
# expanded: Almacena los nodos ya expandidos.
# actionList: Almacena las acciones que va realizando el agente.
# Una vez hecho esto, se inserta el nodo inicial en la frontera
# junto a la lista de acciones (actionList) vacía.
# Siempre que la frontera NO esté vacía, se extrae un nodo de la misma.
# Si el nodo extraído es el nodo objetivo, 
# devuelve la lista de acciones realizadas hasta ese momento.
# Si el nodo que hemos extraído NO se había expandido previamente,
# éste es insertado en la lista de nodos expandidos.
# Cada hijo del nodo extraído es insertado en la frontera con su coste.
# Si no existe ninguna solución, devuelve una lista vacía.

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** MY CODE STARTS HERE ***"
    frontier = util.PriorityQueue()
    expanded = []
    actionList = []
    frontier.push((problem.getStartState(), actionList), problem)
    while not frontier.isEmpty():
        node, actions = frontier.pop()
        if problem.isGoalState(node):
            return actions
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                frontier.push((child[0], actions + [child[1]]), problem.getCostOfActions(actions + [child[1]]))
    return []
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 4

# PREGUNTA 4: BÚSQUEDA A* (ASS)

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función aStarSearch():
# En primer lugar, se han declarado las siguientes variables:
# frontier: Crea una cola con prioridad (orden natural).
# expanded: Almacena los nodos ya expandidos.
# actionList: Almacena las acciones que va realizando el agente.
# Una vez hecho esto, se inserta el nodo inicial en la frontera
# junto a la lista de acciones (actionList) vacía y la heurística de cada una.
# Siempre que la frontera NO esté vacía, se extrae un nodo de la misma.
# Si el nodo extraído es el nodo objetivo, 
# devuelve la lista de acciones realizadas hasta ese momento.
# Si el nodo que hemos extraído NO se había expandido previamente,
# éste es insertado en la lista de nodos expandidos.
# Cada hijo del nodo extraído es insertado en la frontera con su coste y su heurística.
# Si no existe ninguna solución, devuelve una lista vacía.

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** MY CODE STARTS HERE ***"
    frontier = util.PriorityQueue()
    expanded = []
    actionList = []
    frontier.push((problem.getStartState(), actionList), heuristic(problem.getStartState(), problem))
    while not frontier.isEmpty():
        node, actions = frontier.pop()
        if problem.isGoalState(node):
            return actions
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                frontier.push((child[0], actions + [child[1]]), problem.getCostOfActions(actions + [child[1]]) + heuristic(child[0], problem))
    return []
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

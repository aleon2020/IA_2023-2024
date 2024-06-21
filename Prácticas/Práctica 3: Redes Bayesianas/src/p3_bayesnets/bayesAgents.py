# bayesAgents.py
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


import bayesNet as bn
import game
from game import Actions, Agent, Directions
import inference
import layout
import factorOperations
import itertools
import operator as op
import random
import functools
import util

from hunters import GHOST_COLLISION_REWARD, WON_GAME_REWARD
from layout import PROB_BOTH_TOP, PROB_BOTH_BOTTOM, PROB_ONLY_LEFT_TOP, \
    PROB_ONLY_LEFT_BOTTOM, PROB_FOOD_RED, PROB_GHOST_RED

X_POS_VAR = "xPos"
FOOD_LEFT_VAL = "foodLeft"
GHOST_LEFT_VAL = "ghostLeft"
X_POS_VALS = [FOOD_LEFT_VAL, GHOST_LEFT_VAL]

Y_POS_VAR = "yPos"
BOTH_TOP_VAL = "bothTop"
BOTH_BOTTOM_VAL = "bothBottom"
LEFT_TOP_VAL = "leftTop"
LEFT_BOTTOM_VAL = "leftBottom"
Y_POS_VALS = [BOTH_TOP_VAL, BOTH_BOTTOM_VAL, LEFT_TOP_VAL, LEFT_BOTTOM_VAL]

FOOD_HOUSE_VAR = "foodHouse"
GHOST_HOUSE_VAR = "ghostHouse"
HOUSE_VARS = [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR]

TOP_LEFT_VAL = "topLeft"
TOP_RIGHT_VAL = "topRight"
BOTTOM_LEFT_VAL = "bottomLeft"
BOTTOM_RIGHT_VAL = "bottomRight"
HOUSE_VALS = [TOP_LEFT_VAL, TOP_RIGHT_VAL, BOTTOM_LEFT_VAL, BOTTOM_RIGHT_VAL]

OBS_VAR_TEMPLATE = "obs(%d,%d)"

BLUE_OBS_VAL = "blue"
RED_OBS_VAL = "red"
NO_OBS_VAL = "none"
OBS_VALS = [BLUE_OBS_VAL, RED_OBS_VAL, NO_OBS_VAL]

ENTER_LEFT = 0
ENTER_RIGHT = 1
EXPLORE = 2

#______________________________________________________________________________
# OPTIONAL QUESTION

# PREGUNTA OPCIONAL

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función constructHeartAttackBayesNet():
# En primer lugar, se crea una lista que contiene todas las variables que 
# aparecen en la red de Bayes (variableList).
# Después, se crea una lista de tuplas relativa a cada una de las flechas 
# de dicha red de Bayes (edgeTuplesList).
# Además de esto, también se crea un dominio para cada variable, el cual contiene 
# todos los posibles valores de dichas variables, en este caso, True (yes) o False (no).
# Por último, una vez creada la red de Bayes vacía, y teniendo en cuenta 
# todos los datos anteriores, se crea un factor para cada una de las 
# variables que componen dicha red bayesiana, donde se indican las probabilidades
# de todos los casos posibles.
# Una vez creadas todas las CPTs, se asignan sus variables a cada una de ellas
# mediante la función setCPT().
# Esta función termina devolviendo e imprimiendo por pantalla todas las CPTs 
# que aparecen en la red Bayesiana.

def constructHeartAttackBayesNet():
    """
    Optional question

    Define the constructHeartAttackBayesNet() method. Build the Bayesian network structure
    described in the image below, including its variables, values, connections and CPTs. This 
    function will return the constructed Bayesian network.
    """

    "*** MY CODE STARTS HERE ***"
    variableList = ['Exercise', 'Smokes', 'BP', 'Chol', 'Attack']
    edgeTuplesList = [('Exercise', 'BP'), ('Smokes', 'BP'), ('Smokes', 'Chol'), ('BP', 'Attack')]
    variableDomainsDict = {}
    variableDomainsDict['Exercise']  = ['yes', 'no']
    variableDomainsDict['Smokes'] = ['yes', 'no']
    variableDomainsDict['BP']  = ['yes', 'no']
    variableDomainsDict['Chol']  = ['yes', 'no']
    variableDomainsDict['Attack']  = ['yes', 'no']
    bayesNet = bn.constructEmptyBayesNet(variableList, edgeTuplesList, variableDomainsDict)
    exerciseCPT  = bn.Factor(['Exercise'], [], variableDomainsDict)
    exerciseAssignmentDict = {'Exercise' : 'yes'}
    exerciseCPT.setProbability(exerciseAssignmentDict, 0.4)
    exerciseAssignmentDict = {'Exercise' : 'no'}
    exerciseCPT.setProbability(exerciseAssignmentDict, 0.6)
    smokesCPT  = bn.Factor(['Smokes'], [], variableDomainsDict)
    smokesAssignmentDict = {'Smokes' : 'yes'}
    smokesCPT.setProbability(smokesAssignmentDict, 0.15)
    smokesAssignmentDict = {'Smokes' : 'no'}
    smokesCPT.setProbability(smokesAssignmentDict, 0.85)
    BPCPT  = bn.Factor(['BP'], ['Exercise', 'Smokes'], variableDomainsDict)
    BES = {'BP' : 'yes', 'Exercise' : 'yes', 'Smokes' : 'yes'}
    bES = {'BP' : 'no',  'Exercise' : 'yes', 'Smokes' : 'yes'}
    BeS = {'BP' : 'yes', 'Exercise' : 'no',  'Smokes' : 'yes'}
    beS = {'BP' : 'no',  'Exercise' : 'no',  'Smokes' : 'yes'}
    BEs = {'BP' : 'yes', 'Exercise' : 'yes', 'Smokes' : 'no' }
    bEs = {'BP' : 'no',  'Exercise' : 'yes', 'Smokes' : 'no' }
    Bes = {'BP' : 'yes', 'Exercise' : 'no',  'Smokes' : 'no' }
    bes = {'BP' : 'no',  'Exercise' : 'no',  'Smokes' : 'no' }
    BPCPT.setProbability(BES, 0.45)
    BPCPT.setProbability(bES, 0.55)
    BPCPT.setProbability(BeS, 0.95)
    BPCPT.setProbability(beS, 0.05)
    BPCPT.setProbability(BEs, 0.05)
    BPCPT.setProbability(bEs, 0.95)
    BPCPT.setProbability(Bes, 0.55)
    BPCPT.setProbability(bes, 0.45)
    cholCPT = bn.Factor(['Chol'], ['Smokes'], variableDomainsDict)
    CS = {'Chol' : 'yes', 'Smokes' : 'yes'}
    cS = {'Chol' : 'no',  'Smokes' : 'yes'}
    Cs = {'Chol' : 'yes', 'Smokes' : 'no'} 
    cs = {'Chol' : 'no',  'Smokes' : 'no'} 
    cholCPT.setProbability(CS, 0.80)
    cholCPT.setProbability(cS, 0.20)
    cholCPT.setProbability(Cs, 0.40)
    cholCPT.setProbability(cs, 0.60)
    attackCPT  = bn.Factor(['Attack'], ['BP'], variableDomainsDict)
    AB = {'Attack' : 'yes', 'BP' : 'yes'}
    aB = {'Attack' : 'no',  'BP' : 'yes'}
    Ab = {'Attack' : 'yes', 'BP' : 'no'} 
    ab = {'Attack' : 'no',  'BP' : 'no'} 
    attackCPT.setProbability(AB, 0.75)
    attackCPT.setProbability(aB, 0.25)
    attackCPT.setProbability(Ab, 0.05)
    attackCPT.setProbability(ab, 0.95)
    bayesNet.setCPT('Exercise', exerciseCPT)
    bayesNet.setCPT('Smokes', smokesCPT)
    bayesNet.setCPT('BP', BPCPT)
    bayesNet.setCPT('Chol', cholCPT)
    bayesNet.setCPT('Attack', attackCPT)
    print(bayesNet)
    return bayesNet
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 1

# PREGUNTA 1: ESTRUCTURA DE LA RED BAYESIANA

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función constructBayesNet():
# En primer lugar, se itera sobre todas las 'casas', donde se va creando cada una de ellas
# mediante la creación de una variable de observación obsVar y la adición de todos los bordes 
# existentes entre las variables de observación y las variables de posición.
# Una vez fuera del bucle for, se crean los bordes para las variables de posición y los 
# dominios para todas y cada una de las variables.

def constructBayesNet(gameState):
    """
    Question 1: Bayes net structure

    Construct an empty Bayes net according to the structure given in the project
    description.

    There are 5 kinds of variables in this Bayes net:
    - a single "x position" variable (controlling the x pos of the houses)
    - a single "y position" variable (controlling the y pos of the houses)
    - a single "food house" variable (containing the house centers)
    - a single "ghost house" variable (containing the house centers)
    - a large number of "observation" variables for each cell Pacman can measure

    You *must* name all position and house variables using the constants
    (X_POS_VAR, FOOD_HOUSE_VAR, etc.) at the top of this file. 

    The full set of observation variables can be obtained as follows:

        for housePos in gameState.getPossibleHouses():
            for obsPos in gameState.getHouseWalls(housePos)
                obsVar = OBS_VAR_TEMPLATE % obsPos

    In this method, you should:
    - populate `obsVars` using the procedure above
    - populate `edges` with every edge in the Bayes Net (a tuple `(from, to)`)
    - set each `variableDomainsDict[var] = values`, where `values` is the set
      of possible assignments to `var`. These should again be set using the
      constants defined at the top of this file.
    """

    obsVars = []
    edges = []
    variableDomainsDict = {}

    "*** MY CODE STARTS HERE ***"
    for housePos in gameState.getPossibleHouses():
            for obsPos in gameState.getHouseWalls(housePos):
                obsVar = OBS_VAR_TEMPLATE % obsPos
                obsVars.append(obsVar)
                edges.append((FOOD_HOUSE_VAR, obsVar)) 
                edges.append((GHOST_HOUSE_VAR, obsVar))
    edges.append((X_POS_VAR, FOOD_HOUSE_VAR))
    edges.append((X_POS_VAR, GHOST_HOUSE_VAR))
    edges.append((Y_POS_VAR, FOOD_HOUSE_VAR))
    edges.append((Y_POS_VAR, GHOST_HOUSE_VAR))
    variableDomainsDict[X_POS_VAR]  = X_POS_VALS
    variableDomainsDict[Y_POS_VAR]  = Y_POS_VALS
    variableDomainsDict[FOOD_HOUSE_VAR]  = HOUSE_VALS
    variableDomainsDict[GHOST_HOUSE_VAR]  = HOUSE_VALS
    for obsVar in obsVars:
        variableDomainsDict[obsVar]  = OBS_VALS
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

    variables = [X_POS_VAR, Y_POS_VAR] + HOUSE_VARS + obsVars
    net = bn.constructEmptyBayesNet(variables, edges, variableDomainsDict)
    return net, obsVars

def fillCPTs(bayesNet, gameState):
    fillXCPT(bayesNet, gameState)
    fillYCPT(bayesNet, gameState)
    fillHouseCPT(bayesNet, gameState)
    fillObsCPT(bayesNet, gameState)

def fillXCPT(bayesNet, gameState):
    from layout import PROB_FOOD_LEFT 
    xFactor = bn.Factor([X_POS_VAR], [], bayesNet.variableDomainsDict())
    xFactor.setProbability({X_POS_VAR: FOOD_LEFT_VAL}, PROB_FOOD_LEFT)
    xFactor.setProbability({X_POS_VAR: GHOST_LEFT_VAL}, 1 - PROB_FOOD_LEFT)
    bayesNet.setCPT(X_POS_VAR, xFactor)

#______________________________________________________________________________
# QUESTION 2

# PREGUNTA 2: PROBABILIDADES DE LA RED BAYESIANA

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función fillYCPT():
# Se asigna una probabilidad a cada una de las variables en la posición y.

def fillYCPT(bayesNet, gameState):
    """
    Question 2: Bayes net probabilities

    Fill the CPT that gives the prior probability over the y position variable.
    See the definition of `fillXCPT` above for an example of how to do this.
    You can use the PROB_* constants imported from layout rather than writing
    probabilities down by hand.
    """

    yFactor = bn.Factor([Y_POS_VAR], [], bayesNet.variableDomainsDict())
    "*** MY CODE STARTS HERE ***"
    yFactor.setProbability({Y_POS_VAR: BOTH_TOP_VAL}, PROB_BOTH_TOP)
    yFactor.setProbability({Y_POS_VAR: BOTH_BOTTOM_VAL}, PROB_BOTH_BOTTOM)
    yFactor.setProbability({Y_POS_VAR: LEFT_TOP_VAL}, PROB_ONLY_LEFT_TOP)
    yFactor.setProbability({Y_POS_VAR: LEFT_BOTTOM_VAL}, PROB_ONLY_LEFT_BOTTOM)    
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()
    bayesNet.setCPT(Y_POS_VAR, yFactor)

def fillHouseCPT(bayesNet, gameState):
    foodHouseFactor = bn.Factor([FOOD_HOUSE_VAR], [X_POS_VAR, Y_POS_VAR], bayesNet.variableDomainsDict())
    for assignment in foodHouseFactor.getAllPossibleAssignmentDicts():
        left = assignment[X_POS_VAR] == FOOD_LEFT_VAL
        top = assignment[Y_POS_VAR] == BOTH_TOP_VAL or \
                (left and assignment[Y_POS_VAR] == LEFT_TOP_VAL)

        if top and left and assignment[FOOD_HOUSE_VAR] == TOP_LEFT_VAL or \
                top and not left and assignment[FOOD_HOUSE_VAR] == TOP_RIGHT_VAL or \
                not top and left and assignment[FOOD_HOUSE_VAR] == BOTTOM_LEFT_VAL or \
                not top and not left and assignment[FOOD_HOUSE_VAR] == BOTTOM_RIGHT_VAL:
            prob = 1
        else:
            prob = 0

        foodHouseFactor.setProbability(assignment, prob)
    bayesNet.setCPT(FOOD_HOUSE_VAR, foodHouseFactor)

    ghostHouseFactor = bn.Factor([GHOST_HOUSE_VAR], [X_POS_VAR, Y_POS_VAR], bayesNet.variableDomainsDict())
    for assignment in ghostHouseFactor.getAllPossibleAssignmentDicts():
        left = assignment[X_POS_VAR] == GHOST_LEFT_VAL
        top = assignment[Y_POS_VAR] == BOTH_TOP_VAL or \
                (left and assignment[Y_POS_VAR] == LEFT_TOP_VAL)

        if top and left and assignment[GHOST_HOUSE_VAR] == TOP_LEFT_VAL or \
                top and not left and assignment[GHOST_HOUSE_VAR] == TOP_RIGHT_VAL or \
                not top and left and assignment[GHOST_HOUSE_VAR] == BOTTOM_LEFT_VAL or \
                not top and not left and assignment[GHOST_HOUSE_VAR] == BOTTOM_RIGHT_VAL:
            prob = 1
        else:
            prob = 0

        ghostHouseFactor.setProbability(assignment, prob)
    bayesNet.setCPT(GHOST_HOUSE_VAR, ghostHouseFactor)

def fillObsCPT(bayesNet, gameState):
    """
    This funcion fills the CPT that gives the probability of an observation in each square,
    given the locations of the food and ghost houses.

    This function creates a new factor for *each* of 4*7 = 28 observation
    variables. Don't forget to call bayesNet.setCPT for each factor you create.

    The XXXPos variables at the beginning of this method contain the (x, y)
    coordinates of each possible house location.

    IMPORTANT:
    Because of the particular choice of probabilities higher up in the Bayes
    net, it will never be the case that the ghost house and the food house are
    in the same place. However, the CPT for observations must still include a
    vaild probability distribution for this case. To conform with the
    autograder, this function uses the *food house distribution* over colors when both the food
    house and ghost house are assigned to the same cell.
    """

    bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = gameState.getPossibleHouses()

    #convert coordinates to values (strings)
    coordToString = {
        bottomLeftPos: BOTTOM_LEFT_VAL,
        topLeftPos: TOP_LEFT_VAL,
        bottomRightPos: BOTTOM_RIGHT_VAL,
        topRightPos: TOP_RIGHT_VAL
    }

    for housePos in gameState.getPossibleHouses():
        for obsPos in gameState.getHouseWalls(housePos):

            obsVar = OBS_VAR_TEMPLATE % obsPos
            newObsFactor = bn.Factor([obsVar], [GHOST_HOUSE_VAR, FOOD_HOUSE_VAR], bayesNet.variableDomainsDict())
            assignments = newObsFactor.getAllPossibleAssignmentDicts()

            for assignment in assignments:
                houseVal = coordToString[housePos]
                ghostHouseVal = assignment[GHOST_HOUSE_VAR]
                foodHouseVal = assignment[FOOD_HOUSE_VAR]

                if houseVal != ghostHouseVal and houseVal != foodHouseVal:
                    newObsFactor.setProbability({
                        obsVar: RED_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 0)
                    newObsFactor.setProbability({
                        obsVar: BLUE_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 0)
                    newObsFactor.setProbability({
                        obsVar: NO_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 1)
                else:
                    if houseVal == ghostHouseVal and houseVal == foodHouseVal:
                        prob_red = PROB_FOOD_RED
                    elif houseVal == ghostHouseVal:
                        prob_red = PROB_GHOST_RED
                    elif houseVal == foodHouseVal:
                        prob_red = PROB_FOOD_RED

                    prob_blue = 1 - prob_red

                    newObsFactor.setProbability({
                        obsVar: RED_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, prob_red)
                    newObsFactor.setProbability({
                        obsVar: BLUE_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, prob_blue)
                    newObsFactor.setProbability({
                        obsVar: NO_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 0)

            bayesNet.setCPT(obsVar, newObsFactor)

#______________________________________________________________________________
# QUESTION 7

# PREGUNTA 7: INFERENCIA MARGINAL

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función getMostLikelyFoodHousePosition():
# En primer lugar, se llama a la función de inferencia por eliminación de variables
# completada en la pregunta 6 (inferenceByVariableElimination()).
# Después, se busca la posición que tenga una probabilidad más alta de tener o no comida.
# Para ello, se itera sobre todas las posiciones posibles, donde se comprueba si la probabilidad
# que se tiene en la posición actual es mayor que la probabilidad máxima, y, de ser así, 
# se actualiza este valor como la posición con la probabilidad más alta de tener comida encontrada 
# hasta ese momento, además de actualizarse el valor de la probabilidad máxima.
# Por último, la función termina devolviendo el valor de la posición en la que es más probable
# que se encuentre la comida.

def getMostLikelyFoodHousePosition(evidence, bayesNet, eliminationOrder):
    """
    Question 7: Marginal inference for pacman

    Find the most probable position for the food house.
    First, call the variable elimination method you just implemented to obtain
    p(FoodHouse | everything else). Then, inspect the resulting probability
    distribution to find the most probable location of the food house. Return
    this.

    (This should be a very short method.)
    """
    "*** MY CODE STARTS HERE ***"
    probDist = inference.inferenceByVariableElimination(bayesNet, FOOD_HOUSE_VAR, evidence, eliminationOrder)
    mostProbablePosition = None
    maxProb = 0
    for position in probDist.getAllPossibleAssignmentDicts():
        if probDist.getProbability(position) > maxProb:
            mostProbablePosition = position
            maxProb = probDist.getProbability(position)
    return mostProbablePosition
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()


class BayesAgent(game.Agent):

    def registerInitialState(self, gameState):
        self.bayesNet, self.obsVars = constructBayesNet(gameState)
        fillCPTs(self.bayesNet, gameState)

        self.distances = cacheDistances(gameState)
        self.visited = set()
        self.steps = 0

    def getAction(self, gameState):
        self.visited.add(gameState.getPacmanPosition())
        self.steps += 1

        if self.steps < 40:
            return self.getRandomAction(gameState)
        else:
            return self.goToBest(gameState)

    def getRandomAction(self, gameState):
        legal = list(gameState.getLegalActions())
        legal.remove(Directions.STOP)
        random.shuffle(legal)
        successors = [gameState.generatePacmanSuccessor(a).getPacmanPosition() for a in legal]
        ls = [(a, s) for a, s in zip(legal, successors) if s not in gameState.getPossibleHouses()]
        ls.sort(key=lambda p: p[1] in self.visited)
        return ls[0][0]

    def getEvidence(self, gameState):
        evidence = {}
        for ePos, eColor in gameState.getEvidence().items():
            obsVar = OBS_VAR_TEMPLATE % ePos
            obsVal = {
                "B": BLUE_OBS_VAL,
                "R": RED_OBS_VAL,
                " ": NO_OBS_VAL
            }[eColor]
            evidence[obsVar] = obsVal
        return evidence

    def goToBest(self, gameState):
        evidence = self.getEvidence(gameState)
        unknownVars = [o for o in self.obsVars if o not in evidence]
        eliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR, GHOST_HOUSE_VAR]
        bestFoodAssignment = getMostLikelyFoodHousePosition(evidence, 
                self.bayesNet, eliminationOrder)

        tx, ty = dict(
            zip([BOTTOM_LEFT_VAL, TOP_LEFT_VAL, BOTTOM_RIGHT_VAL, TOP_RIGHT_VAL],
                gameState.getPossibleHouses()))[bestFoodAssignment[FOOD_HOUSE_VAR]]
        bestAction = None
        bestDist = float("inf")
        for action in gameState.getLegalActions():
            succ = gameState.generatePacmanSuccessor(action)
            nextPos = succ.getPacmanPosition()
            dist = self.distances[nextPos, (tx, ty)]
            if dist < bestDist:
                bestDist = dist
                bestAction = action
        return bestAction

#______________________________________________________________________________
# QUESTION 8

# PREGUNTA 8: VALOR DE INFORMACIÓN PERFECTA

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función computeEnterValues():
# En primer lugar, dada la evidencia, se calcula el valor de ir a la izquierda y después
# a la derecha de forma inmediata.
# Para ello, se obtiene la distribución conjunta sobre las posiciones de la comida y la casa 
# de los fantasmas usando su procedimiento de inferencia, donde la recompensa asociada por 
# entrar a cada casa se da en las variables del tipo *_REWARD situadas en la parte superior
# de este fichero.
# - Función computeExploreValue():
# En primer lugar, se calcula el valor esperado de explorar primero lo que queda 
# oculto, para después entrar con el valor esperado más alto.
# Para ello, se utiliza la función getExplorationProbsAndOutcomes, la cual devuelve 
# pares de la forma (problema, exploraciónEvidencia), donde "evidencia" es una nueva 
# evidencia que incluye todas las observaciones faltantes completadas, y "prob" es
# la probabilidad de que ocurra ese conjunto de observaciones.

class VPIAgent(BayesAgent):

    def __init__(self):
        BayesAgent.__init__(self)
        self.behavior = None
        NORTH = Directions.NORTH
        SOUTH = Directions.SOUTH
        EAST = Directions.EAST
        WEST = Directions.WEST
        self.exploreActionsRemaining = \
                list(reversed([NORTH, NORTH, NORTH, NORTH, EAST, EAST, EAST,
                    EAST, SOUTH, SOUTH, SOUTH, SOUTH, WEST, WEST, WEST, WEST]))

    def reveal(self, gameState):
        bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = \
                gameState.getPossibleHouses()
        for housePos in [bottomLeftPos, topLeftPos, bottomRightPos]:
            for ox, oy in gameState.getHouseWalls(housePos):
                gameState.data.observedPositions[ox][oy] = True

    def computeEnterValues(self, evidence, eliminationOrder):
        """
        Question 8a: Value of perfect information

        Given the evidence, compute the value of entering the left and right
        houses immediately. You can do this by obtaining the joint distribution
        over the food and ghost house positions using your inference procedure.
        The reward associated with entering each house is given in the *_REWARD
        variables at the top of the file.

        *Do not* take into account the "time elapsed" cost of traveling to each
        of the houses---this is calculated elsewhere in the code.
        """

        leftExpectedValue = 0
        rightExpectedValue = 0

        "*** MY CODE STARTS HERE ***"
        inference_factor = inference.inferenceByVariableElimination(self.bayesNet, [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR], evidenceDict=evidence, eliminationOrder=eliminationOrder)
        for assignmentDict in inference_factor.getAllPossibleAssignmentDicts():
            prob = inference_factor.getProbability(assignmentDict)
            if prob != 0:
                if assignmentDict["foodHouse"] == "topRight":
                    prob_right = prob
                else:
                    prob_left = prob
        leftExpectedValue = prob_left * WON_GAME_REWARD + GHOST_COLLISION_REWARD * prob_right
        rightExpectedValue = prob_right * WON_GAME_REWARD + GHOST_COLLISION_REWARD * prob_left
        return leftExpectedValue,rightExpectedValue
        "*** MY CODE FINISH HERE ***"
        # util.raiseNotDefined()

        return leftExpectedValue, rightExpectedValue

    def getExplorationProbsAndOutcomes(self, evidence):
        unknownVars = [o for o in self.obsVars if o not in evidence]
        assert len(unknownVars) == 7
        assert len(set(evidence.keys()) & set(unknownVars)) == 0
        firstUnk = unknownVars[0]
        restUnk = unknownVars[1:]

        unknownVars = [o for o in self.obsVars if o not in evidence]
        eliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR]
        houseMarginals = inference.inferenceByVariableElimination(self.bayesNet,
                [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR], evidence, eliminationOrder)

        probs = [0 for i in range(8)]
        outcomes = []
        for nRed in range(8):
            outcomeVals = [RED_OBS_VAL] * nRed + [BLUE_OBS_VAL] * (7 - nRed)
            outcomeEvidence = dict(zip(unknownVars, outcomeVals))
            outcomeEvidence.update(evidence)
            outcomes.append(outcomeEvidence)

        for foodHouseVal, ghostHouseVal in [(TOP_LEFT_VAL, TOP_RIGHT_VAL),
                (TOP_RIGHT_VAL, TOP_LEFT_VAL)]:

            condEvidence = dict(evidence)
            condEvidence.update({FOOD_HOUSE_VAR: foodHouseVal, 
                GHOST_HOUSE_VAR: ghostHouseVal})
            assignmentProb = houseMarginals.getProbability(condEvidence)

            oneObsMarginal = inference.inferenceByVariableElimination(self.bayesNet,
                    [firstUnk], condEvidence, restUnk + [X_POS_VAR, Y_POS_VAR])

            assignment = oneObsMarginal.getAllPossibleAssignmentDicts()[0]
            assignment[firstUnk] = RED_OBS_VAL
            redProb = oneObsMarginal.getProbability(assignment)

            for nRed in range(8):
                outcomeProb = combinations(7, nRed) * \
                        redProb ** nRed * (1 - redProb) ** (7 - nRed)
                outcomeProb *= assignmentProb
                probs[nRed] += outcomeProb

        return list(zip(probs, outcomes))

    def computeExploreValue(self, evidence, enterEliminationOrder):
        """
        Question 8b: Value of perfect information

        Compute the expected value of first exploring the remaining unseen
        house, and then entering the house with highest expected value.

        The method `getExplorationProbsAndOutcomes` returns pairs of the form
        (prob, explorationEvidence), where `evidence` is a new evidence
        dictionary with all of the missing observations filled in, and `prob` is
        the probability of that set of observations occurring.

        You can use getExplorationProbsAndOutcomes to
        determine the expected value of acting with this extra evidence.
        """

        expectedValue = 0

        "*** MY CODE STARTS HERE ***"
        exploration = self.getExplorationProbsAndOutcomes(evidence)
        for prob_explore,evidence_explore in exploration:
            expectedValue = max(self.computeEnterValues(evidence_explore, enterEliminationOrder)) * prob_explore + expectedValue
        return expectedValue
        "*** MY CODE FINISH HERE ***"
        # util.raiseNotDefined()

        return expectedValue

    def getAction(self, gameState):

        if self.behavior == None:
            self.reveal(gameState)
            evidence = self.getEvidence(gameState)
            unknownVars = [o for o in self.obsVars if o not in evidence]
            enterEliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR]
            exploreEliminationOrder = [X_POS_VAR, Y_POS_VAR]

            print(evidence)
            print(enterEliminationOrder)
            print(exploreEliminationOrder)
            enterLeftValue, enterRightValue = \
                    self.computeEnterValues(evidence, enterEliminationOrder)
            exploreValue = self.computeExploreValue(evidence,
                    exploreEliminationOrder)

            # TODO double-check
            enterLeftValue -= 4
            enterRightValue -= 4
            exploreValue -= 20

            bestValue = max(enterLeftValue, enterRightValue, exploreValue)
            if bestValue == enterLeftValue:
                self.behavior = ENTER_LEFT
            elif bestValue == enterRightValue:
                self.behavior = ENTER_RIGHT
            else:
                self.behavior = EXPLORE

            # pause 1 turn to reveal the visible parts of the map
            return Directions.STOP

        if self.behavior == ENTER_LEFT:
            return self.enterAction(gameState, left=True)
        elif self.behavior == ENTER_RIGHT:
            return self.enterAction(gameState, left=False)
        else:
            return self.exploreAction(gameState)

    def enterAction(self, gameState, left=True):
        bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = \
                gameState.getPossibleHouses()

        dest = topLeftPos if left else topRightPos

        actions = gameState.getLegalActions()
        neighbors = [gameState.generatePacmanSuccessor(a) for a in actions]
        neighborStates = [s.getPacmanPosition() for s in neighbors]
        best = min(zip(actions, neighborStates), 
                key=lambda x: self.distances[x[1], dest])
        return best[0]

    def exploreAction(self, gameState):
        if self.exploreActionsRemaining:
            return self.exploreActionsRemaining.pop()

        evidence = self.getEvidence(gameState)
        enterLeftValue, enterRightValue = self.computeEnterValues(evidence,
                [X_POS_VAR, Y_POS_VAR])

        if enterLeftValue > enterRightValue:
            self.behavior = ENTER_LEFT
            return self.enterAction(gameState, left=True)
        else:
            self.behavior = ENTER_RIGHT
            return self.enterAction(gameState, left=False)

def cacheDistances(state):
    width, height = state.data.layout.width, state.data.layout.height
    states = [(x, y) for x in range(width) for y in range(height)]
    walls = state.getWalls().asList() + state.data.layout.redWalls.asList() + state.data.layout.blueWalls.asList()
    states = [s for s in states if s not in walls]
    distances = {}
    for i in states:
        for j in states:
            if i == j:
                distances[i, j] = 0
            elif util.manhattanDistance(i, j) == 1:
                distances[i, j] = 1
            else:
                distances[i, j] = 999999
    for k in states:
        for i in states:
            for j in states:
                if distances[i,j] > distances[i,k] + distances[k,j]:
                    distances[i,j] = distances[i,k] + distances[k,j]

    return distances

# http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def combinations(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    return numer / denom


# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

# PREGUNTA 1: CALENTAMIENTO

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función sentence1():
# Representa la veracidad de estas tres sentencias encadenándolas en este orden:
# A ∨ B ; ¬A ⇔ (¬B ∨ C) ; ¬A ∨ ¬B ∨ C.
# - Función sentence2():
# Representa la veracidad de estas cuatro sentencias encadenándolas en este orden:
# C ⇔ (B ∨ D) ; A ⇒ (¬B ∧ ¬D) ; ¬(B ∧ ¬C) ⇒ A ; ¬D ⇒ C
# - Función sentence3():
# Crea cuatro símbolos, cada uno para un estado concreto:
# PacmanAlive_0: Pacman vivo en el tiempo 0.
# PacmanAlive_1: Pacman vivo en el tiempo 1.
# PacmanBorn_0: Pacman nace en el tiempo 0.
# PacmanKilled_0: Pacman asesinado en el tiempo 0.
# Y que después, se aplican en las siguientes tres sentencias:
# 1. Pacman está vivo en el tiempo 1 si y solo si estaba vivo en el tiempo 0 y no 
# fue asesinado en el tiempo 0 o no estaba vivo en el tiempo 0 y nació en el tiempo 0.
# 2. En el tiempo 0, Pacman no puede estar a la vez vivo y nacer.
# 3. Pacman nace en el tiempo 0.
# - Función entails():
# Se llama a la función findModel(), que devuelve un diccionario, y logic.disjoin()
# devuelve una expresión.
# Si algún valor del diccionario es True, la conclusión es verdadera, es decir,
# devuelve True si la premise implica la conclusión.
# - Función plTrueInverse():
# Devuelve true si la condición en pl_true es verdadera.

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** MY CODE STARTS HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    s1 = A | B
    s2 = ~A % (~B | C)
    s3 = logic.disjoin(~A, ~B, C)
    return logic.conjoin(s1, s2, s3)
    "*** MY CODE FINISH HERE ***"


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """

    "*** MY CODE STARTS HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    s1 = C % (B | D)
    s2 = A >> (~B & ~D)
    s3 = ~(B & ~C) >> A
    s4 = ~D >> C
    return logic.conjoin(s1, s2, s3, s4)
    "*** MY CODE FINISH HERE ***"


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    (Project update: for this question only, [0] and _t are both acceptable.)
    """
    "*** MY CODE STARTS HERE ***"
    A = logic.PropSymbolExpr("PacmanAlive", time = 0)
    B = logic.PropSymbolExpr("PacmanAlive", time = 1)
    C = logic.PropSymbolExpr("PacmanBorn", time = 0)
    D = logic.PropSymbolExpr("PacmanKilled", time = 0)
    s1 = B % ((A & ~D) | (~A & C))
    s2 = ~(A & C)
    s3 = C
    return logic.conjoin(s1, s2, s3)
    "*** MY CODE FINISH HERE ***"

def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf_sentence = to_cnf(sentence)
    return pycoSAT(cnf_sentence)

def findModelCheck() -> Dict[Any, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    This can be solved with a one-line return statement.
    """
    class dummyClass:
        """dummy('A') has representation A, unlike a string 'A' that has repr 'A'.
        Of note: Expr('Name') has representation Name, not 'Name'.
        """
        def __init__(self, variable_name: str = 'A'):
            self.variable_name = variable_name
        
        def __repr__(self):
            return self.variable_name

    return {dummyClass('a'): True}


def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** MY CODE STARTS HERE ***"
    s = premise & ~conclusion
    return not findModel(s)
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()
    

def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** MY CODE STARTS HERE ***"
    return not logic.pl_true(inverse_statement, assignments)
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 2

# PREGUNTA 2: EJERCICIOS LÓGICOS

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función atLeastOne():
# Devuelve una sola expresión en CNF verdadera si al menos una de las expresiones
# de la lista es verdadera (con literals se desempaqueta la lista).
# - Función atMostOne():
# Crea clausula, una lista vacía para ir almacenando expresiones.
# itertools.combinations va devolviendo una lista de tuplas iterando por 
# todas las combinaciones posibles de dos elementos de la lista.
# Dentro del for, se añade a clausula la expresión ~elements[0] | ~elements[1], donde 
# elements[0] y elements[1] son las tuplas de la lista, siendo i[0] el primer elemento de la tupla.
# Por último, devuelve la lista clausula con todas las expresiones de dicha lista.
# - Función exactlyOne():
# Devuelve una sola expresión en CNF verdadera si solo una de las expresiones
# de la lista es verdadera, ya que atLeast devuelve al menos 1, y atMostOne devuelve
# como mucho 1, por lo que, al juntarlas, su intersección es 1.

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** MY CODE STARTS HERE ***"
    return logic.disjoin(literals)
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** MY CODE STARTS HERE ***"
    combinations = itertools.combinations(literals, 2)
    clausula = []
    for elements in combinations:
        clausula.append(~elements[0] | ~elements[1])
    return logic.conjoin(clausula)
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()


def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** MY CODE STARTS HERE ***"
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 3

# PREGUNTA 3: PACPHYSICS Y SATISFACIBILIDAD

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función pacmanSuccessorAxiomSingle():
# Devuelve una expresión que contiene todas las condiciones para que Pacman
# esté en la posición (x, y) en el instante t, además de ir leyendo la
# construcción de possible_causes, para la que he utilizado disjoin.
# - Función pacphysicsAxioms():
# Para todas y cada una de las posiciones (x, y) del mapa, comprueba si hay una pared
# en dicha posición, y si esto es así, concluye que Pacman no puede estar en esa posición.
# El primer append establece que solo puede haber un Pacman por posición.
# El segundo append establece que solo puede haber una acción por instante de tiempo.
# Si se tiene un sensorModel no nulo, éste se agrega a las sentencias,
# al igual que todos y cada uno de sus axiomas.
# Si se provee una función para calcular las axiomas del sucesor, ésta se agrega a las sentencias,
# al igual que todos y cada uno de sus axiomas.
# Por último, devuelve todas las sentencias unidas en una sola gracias a conjoin.
# - Función checkLocationSatisfiability():
# En primer lugar, se agregan a la lista map_sent todas las sentencias
# referentes a las paredes, que se unen en una sola gracias a conjoin.
# El primer append agrega los axiomas de las físicas del Pacman en el instante 0.
# El segundo append agrega la posición inicial del Pacman.
# El tercer append agrega la acción del Pacman en el instante 0.
# El cuarto append agrega los axiomas de las físicas del Pacman en el instante 1.
# El quinto append agrega la acción del Pacman en el instante 1.
# Y por último, se agrega la posición final en la que Pacman sí se encuentra
# y en las que no, para después retornar dichos modelos.

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    # the if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None
    
    "*** MY CODE STARTS HERE ***"
    return logic.PropSymbolExpr(pacman_str, x, y, time=now) % logic.disjoin(possible_causes)
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    pacphysics_sentences = []

    "*** MY CODE STARTS HERE ***"
    for (x, y) in all_coords:
        pacphysics_sentences.append(PropSymbolExpr(wall_str,x ,y) >> ~PropSymbolExpr(pacman_str, x, y, time=t))
    pacphysics_sentences.append(exactlyOne([PropSymbolExpr(pacman_str, x, y, time=t) for (x, y) in non_outer_wall_coords]))
    pacphysics_sentences.append(exactlyOne([PropSymbolExpr(action, time=t) for action in DIRECTIONS]))
    if sensorModel != None:
        pacphysics_sentences.append(sensorModel(t, non_outer_wall_coords))
    if successorAxioms != None:
        if t != 0:
            pacphysics_sentences.append(successorAxioms(t, walls_grid, non_outer_wall_coords))    
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

    return conjoin(pacphysics_sentences)


def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** MY CODE STARTS HERE ***"
    KB.append(pacphysicsAxioms(0, all_coords, non_outer_wall_coords, walls_grid, None, allLegalSuccessorAxioms))
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    KB.append(PropSymbolExpr(action0, time=0))
    KB.append(pacphysicsAxioms(1, all_coords, non_outer_wall_coords, walls_grid, None, allLegalSuccessorAxioms)) 
    KB.append(PropSymbolExpr(action1, time=1)) 
    model1 = findModel(conjoin(KB) & PropSymbolExpr(pacman_str, x1, y1,time=1))
    model2 = findModel(conjoin(KB) & ~PropSymbolExpr(pacman_str, x1, y1, time=1))
    return (model1, model2)    
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 4

# PREGUNTA 4: PATH PLANNING CON LÓGICA

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función positionLogicPlan():
# En primer lugar, se agrega la posición inicial del Pacman y
# se itera en los primeros 50 instantes de tiempo.
# Se crea una lista que almacena todas las posiciones que se van visitando.
# Después, se itera por todas las posiciones del mapa que no sean muros, y si
# se tiene que no se está en el primer instante de tiempo, éste se agrega a la
# posición del Pacman del instante de tiempo anterior, además de los axiomas 
# de su sucesor y las demás posiciones del Pacman.
# Ya fuera del for, se establece que sólo puede haber una posición por instante de tiempo.
# Después, se crea una lista para almacenar todas las acciones realizadas, las
# cuales se agregan en dicha lista, estableciendo nuevamente que solo puede haber
# una acción por instante de tiempo.
# Por último, se agrega la posición final del Pacman y se extrae la secuencia de acciones,
# retornando dicha secuencia.

def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** MY CODE STARTS HERE ***"
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    for t in range(50):
        coord_list = []
        for x, y in non_wall_coords:
            if t != 0:
                KB.append(pacmanSuccessorAxiomSingle(x, y, t, walls_grid))
            coord_list.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(coord_list))
        action_list = []
        for direction in actions: 
            action_list.append(PropSymbolExpr(direction, time=t))
        KB.append(exactlyOne(action_list))
        knowledge_base = conjoin(KB) 
        model = findModel(knowledge_base & PropSymbolExpr(pacman_str, xg, yg, time=t))
        if model:
            action_sequence = extractActionSequence(model, actions)
            return action_sequence    
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 5

# PREGUNTA 5: COMIENDO TODA LA COMIDA

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función foodLogicPlan():
# En primer lugar, se agrega la posición inicial del Pacman y
# se itera en los primeros 50 instantes de tiempo.
# Después, se itera por todas y cada una de las posiciones en las que
# haya comida, y éstas se agregan a la posición inicial de la misma.
# Se crea una lista que almacena toda la comida que ya ha sido recogida.
# Después, se crea una expresión para la posición actual de la comida, 
# otra para la posición futura de la comida, y otra para la posición del Pacman,
# y todas ellas se agregan en una única sentencia, estableciendo que si la comida
# está en la posición del Pacman, ésta no tendrá una posición futura, agregando
# la posición en la que se ha encontrado la comida a la lista.
# Una vez hecho esto, se crea una lista que almacena todas las posiciones 
# que se van visitando.
# Después, se itera por todas las posiciones del mapa que no sean muros, y si
# se tiene que no se está en el primer instante de tiempo, éste se agrega a la
# posición del Pacman del instante de tiempo anterior, además de los axiomas 
# de su sucesor y las demás posiciones del Pacman.
# Ya fuera del for, se establece que sólo puede haber una posición por instante de tiempo.
# Después, se crea una lista para almacenar todas las acciones realizadas, las
# cuales se agregan en dicha lista, estableciendo nuevamente que solo puede haber
# una acción por instante de tiempo.
# Por último, se agrega la posición final del Pacman y se extrae la secuencia de acciones,
# retornando dicha secuencia.

def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** MY CODE STARTS HERE ***"
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    for x0,y0 in food:
        KB.append(PropSymbolExpr(food_str, x0, y0, time=0))
    for t in range(50):
        food_list = []
        for x, y in food: 
            food__actual_position = PropSymbolExpr(food_str,x,y,time=t)
            food_future_position = PropSymbolExpr(food_str,x,y,time=t+1)
            pacman_position = PropSymbolExpr(pacman_str,x,y,time=t)
            KB.append(((food__actual_position & ~pacman_position) % food_future_position))
            food_list.append(food__actual_position)
        coord_list = []
        for x, y in non_wall_coords:
            if t != 0:
                KB.append(pacmanSuccessorAxiomSingle(x, y, t, walls))
            coord_list.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(coord_list))
        action_list = []
        for direction in actions: 
            action_list.append(PropSymbolExpr(direction, time=t))
        KB.append(exactlyOne(action_list))
        knowledge_base = conjoin(KB)
        model = findModel(knowledge_base & ~disjoin(food_list))
        if model:
            action_sequence = extractActionSequence(model, actions)
            return action_sequence
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 6
    
# PREGUNTA 6: LOCALIZACIÓN
    
# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función localization():
# En primer lugar, se agregan a la KB tanto las ubicaciones que corresponden a
# una pared como las que no.
# Una vez hecho esto, ya dentro del for, se ejecuta el bloque de código correspondiente
# a la función auxiliar Aux1, donde se añaden a la KB todos los pacphysics_axioms()
# de la pregunta 3, así como todas las acciones preescritas por agent.actions[t]
# y las reglas de percepción resultantes de llamar a agent.getPercepts().
# Después, se ejecuta el bloque de código correspondiente a la función auxiliar Aux2,
# donde se crea una lista vacía llamada possible_locations y se va iterando por todas 
# las ubicaciones que no sean paredes. Sabiendo esto, se utiliza la KB y entails para
# demostrar si Pacman se encuentra en una ubicación determinada (x,y) o no en un momento 
# específico. Por tanto, si se demuestra que Pacman se encuentra en la ubicación (x,y) en
# el momento t, se asigna dicha ubicación a la lista possible_locations, además de añadir a
# la KB todas las ubicación (x,y) en las que se encuentre tanto probadamente como no
# en el tiempo t.
# Por último, llama a agent.moveToNextState() en la acción actual del agente en el instante
# t y termina devolviendo las posibles ubicaciones mediante yield.
# Es importante mencionar que se ha añadido una condición en la que se establece que si 
# los resultados de entails se contradicen entre sí, se imprime una traza alertando 
# sobre ello.

def localization(problem, agent) -> Generator:
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []

    # util.raiseNotDefined()
    # for t in range(agent.num_timesteps):

    "*** MY CODE STARTS HERE ***"
    for x, y in all_coords:
        if (x, y) in walls_list:
            KB.append(PropSymbolExpr(wall_str, x, y))
        else:
            KB.append(~PropSymbolExpr(wall_str, x, y))
    for t in range(agent.num_timesteps):
        KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords, walls_grid, sensorAxioms, allLegalSuccessorAxioms))
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        KB.append(fourBitPerceptRules(t, agent.getPercepts()))
        possible_locations = []
        for x, y in non_outer_wall_coords:
            pacman_at = PropSymbolExpr(pacman_str, x, y, time=t)
            if entails(conjoin(KB), pacman_at):
                if entails(conjoin(KB), ~pacman_at):
                    print("Contadiction at ", x, y)
                KB.append(pacman_at)
            elif entails(conjoin(KB), ~pacman_at):
                KB.append(~pacman_at)
            if findModel(conjoin(KB) & pacman_at):
                possible_locations.append((x,y))
        agent.moveToNextState(agent.actions[t])
        "*** MY CODE FINISH HERE ***"

        yield possible_locations

#______________________________________________________________________________
# QUESTION 7
        
# PREGUNTA 7: MAPEO
    
# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función mapping():
# En primer lugar, se añade a la KB la ubicación inicial de Pacman (pac_x0,pac_y0), indicando
# también si hay una pared en la ubicación en la que se encuentra en ese momento.
# Una vez hecho esto, ya dentro del for, se ejecuta el bloque de código correspondiente
# a la función auxiliar Aux1, donde se añaden a la KB todos los pacphysics_axioms()
# de la pregunta 3, así como todas las acciones preescritas por agent.actions[t]
# y las reglas de percepción resultantes de llamar a agent.getPercepts().
# Después, se ejecuta el bloque de código correspondiente a la función auxiliar Aux3,
# donde se va iterando por todas las ubicaciones que no sean paredes. Sabiendo esto, 
# se utiliza la KB y entails para demostrar si Pacman se encuentra en una ubicación en la que 
# haya o no una pared. Por tanto, se añaden a la KB todas las ubicaciones (x,y) en las 
# que exista o no probabilidad de que haya una pared en la ubicación que se encuentre Pacman
# en ese momento.
# Por último, llama a agent.moveToNextState() en la acción actual del agente en el instante
# t y termina devolviendo el mapa conocido mediante yield known_map.
# Es importante mencionar que se ha añadido una condición en la que se establece que si 
# los resultados de entails se contradicen entre sí, se imprime una traza alertando 
# sobre ello. 

def mapping(problem, agent) -> Generator:
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    # util.raiseNotDefined()
    # for t in range(agent.num_timesteps):

    "*** MY CODE STARTS HERE ***"
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    known_map[pac_x_0][pac_y_0] = 0
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    for t in range(agent.num_timesteps):
        KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords, known_map, sensorAxioms, allLegalSuccessorAxioms))
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        KB.append(fourBitPerceptRules(t, agent.getPercepts()))
        for x, y in non_outer_wall_coords:
            wall_at = PropSymbolExpr(wall_str, x, y)
            if entails(conjoin(KB), wall_at):
                if entails(conjoin(KB), ~wall_at):
                    print("Contadiction at ", x, y)
                KB.append(wall_at)
                if known_map[x][y] == -1: 
                    known_map[x][y] = 1
            elif entails(conjoin(KB), ~wall_at):
                KB.append(~wall_at)
                if known_map[x][y] == -1: 
                    known_map[x][y] = 0
        agent.moveToNextState(agent.actions[t])
        "*** MY CODE FINISH HERE ***"

        yield known_map

#______________________________________________________________________________
# QUESTION 8
        
# PREGUNTA 8: SLAM
    
# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función slam():
# En primer lugar, se añade a la KB la ubicación inicial de Pacman (pac_x0,pac_y0), indicando
# también si hay una pared en la ubicación en la que se encuentra.
# Una vez hecho esto, ya dentro del for, se ejecuta el bloque de código correspondiente
# a la función auxiliar Aux1, donde se añaden a la KB todos los pacphysics_axioms()
# de la pregunta 3, así como todas las acciones preescritas por agent.actions[t]
# y las reglas de percepción resultantes de llamar a agent.getPercepts().
# Después, se ejecuta el bloque de código correspondiente a la función auxiliar Aux3,
# donde se va iterando por todas las ubicaciones que no sean paredes. Sabiendo esto, 
# se utiliza la KB y entails para demostrar si Pacman se encuentra en una ubicación en la que 
# haya o no una pared. Por tanto, se añaden a la KB todas las ubicaciones (x,y) en las 
# que exista o no probabilidad de que haya una pared en la ubicación que se encuentre Pacman
# en ese momento.
# A continuación, se ejecuta el bloque de código correspondiente a la función auxiliar Aux2,
# donde se crea una lista vacía llamada possible_locations y se va iterando por todas 
# las ubicaciones que no sean paredes. Sabiendo esto, se utiliza la KB y entails para
# demostrar si Pacman se encuentra en una ubicación determinada (x,y) o no en un momento 
# específico. Por tanto, si se demuestra que Pacman se encuentra en la ubicación (x,y) en
# el tiempo t, se asigna dicha ubicación a la lista possible_locations, además de añadir a
# la KB todas las ubicaciones (x,y) en las que se encuentre tanto probadamente como no
# en el tiempo t.
# Por último, llama a agent.moveToNextState() en la acción actual del agente en el instante
# t y termina devolviendo las posibles ubicaciones, tanto de Pacman como del mapa conocido 
# mediante yield known_map y possible_locations.
# Es importante mencionar que se ha añadido una condición en la que se establece que si 
# los resultados de entails se contradicen entre sí, se imprime una traza alertando 
# sobre ello. 

def slam(problem, agent) -> Generator:
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    # util.raiseNotDefined()
    # for t in range(agent.num_timesteps):

    "*** MY CODE STARTS HERE ***"
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    known_map[pac_x_0][pac_y_0] = 0
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    for t in range(agent.num_timesteps):
        KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords, known_map, SLAMSensorAxioms, SLAMSuccessorAxioms))
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        KB.append(numAdjWallsPerceptRules(t, agent.getPercepts()))
        for x, y in non_outer_wall_coords:
            wall_at = PropSymbolExpr(wall_str, x, y)
            if entails(conjoin(KB), wall_at):
                if entails(conjoin(KB), ~wall_at):
                    print("Contadiction at ", x, y)
                KB.append(wall_at)
                if known_map[x][y] == -1: 
                    known_map[x][y] = 1
            elif entails(conjoin(KB), ~wall_at):
                KB.append(~wall_at)
                if known_map[x][y] == -1: 
                    known_map[x][y] = 0
        possible_locations = []
        for x, y in non_outer_wall_coords:
            pacman_at = PropSymbolExpr(pacman_str, x, y, time=t)
            if entails(conjoin(KB), pacman_at):
                if entails(conjoin(KB), ~pacman_at):
                    print("Contadiction at ", x, y)
                KB.append(pacman_at)
            elif entails(conjoin(KB), ~pacman_at):
                KB.append(~pacman_at)
            if findModel(conjoin(KB) & pacman_at):
                possible_locations.append((x,y))
        agent.moveToNextState(agent.actions[t])
        "*** MY CODE FINISH HERE ***"
        
        yield (known_map, possible_locations)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

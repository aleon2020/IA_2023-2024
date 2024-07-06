# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util
import functools

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

#______________________________________________________________________________
# QUESTION 3

# PREGUNTA 3: UNIR FACTORES

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función joinFactors():
# En primer lugar se crean dos listas, una para guardar las variables de los factores que 
# estén condicionados, y otra para aquellas que no estén condicionadas.
# Después, se recorren todos los factores, donde cada uno de ellos se guarda sus variables y sus 
# dominios.
# Ya fuera del for, se crean dos conjuntos: Ambos almacenan las variables de los factores, pero
# uno lo hace para las variables condicionadas y el otro para las variables no condicionadas,
# y después, estos dos conjuntos se restan para eliminar las variables no condicionadas.
# Por último, se crea un nuevo factor, del cual voy recorriendo todas sus posibles 
# asignaciones, donde se multiplican las probabilidades de todos sus factores y se asigna dicha 
# probabilidad al nuevo factor.

def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** MY CODE STARTS HERE ***"
    listConditioned = []
    listUnconditioned = []
    for factor in factors:
        listConditioned += factor.conditionedVariables()
        listUnconditioned += factor.unconditionedVariables() 
        domainFactor = factor.variableDomainsDict()
    conditioned = set(listConditioned)
    unconditioned = set(listUnconditioned)
    conditioned = conditioned - unconditioned
    newFactor = Factor(unconditioned, conditioned, domainFactor)
    for assignment in newFactor.getAllPossibleAssignmentDicts():
        prob = 1
        for factor in factors:
            prob *= factor.getProbability(assignment)
        newFactor.setProbability(assignment, prob)
    return newFactor
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()

#______________________________________________________________________________
# QUESTION 4

# PREGUNTA 4: ELIMINACIÓN

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función eliminate():
# En primer lugar se crean dos listas, una para las variables condicionadas y 
# otra para las variables no condicionadas.
# Además, se crea otra variable correspondiente al dominio de las variables.
# Después, se elimina el objetivo fijado en la lista de variables no condicionadas.
# Una vez hecho esto, se crea el nuevo factor y su estructura, a la que se le van
# asignando probabilidades.
# Por último, se crea una lista de posibles valores de la variable a eliminar, la cual
# almacena la suma de todas las probabilidades calculadas, y se asigna dicho valor al
# nuevo factor.

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** MY CODE STARTS HERE ***"
        unconditionedVar = factor.unconditionedVariables()
        conditionedVar = factor.conditionedVariables()
        domain = factor.variableDomainsDict()
        unconditionedVar.remove(eliminationVariable)
        newFactor = Factor(unconditionedVar, conditionedVar, domain)
        assignments = newFactor.getAllPossibleAssignmentDicts()
        for assignment in assignments:
          eliminateVarValues = factor.variableDomainsDict()[eliminationVariable]
          prob = 0.0
          for value in eliminateVarValues:
              assignment[eliminationVariable] = value
              prob_value = factor.getProbability(assignment)
              prob += prob_value
          newFactor.setProbability(assignment, prob)
        return newFactor
        "*** MY CODE FINISH HERE ***"
        # util.raiseNotDefined()

    return eliminate

eliminate = eliminateWithCallTracking()

#______________________________________________________________________________
# QUESTION 5

# PREGUNTA 5: NORMALIZACIÓN

# EXPLICACIÓN DEL ALGORITMO IMPLEMENTADO:
# - Función normalize():
# En primer lugar se crea la lista 'assignments', la cual va a almacenar todas las posibles
# combinaciones de variables que pueda haber en el factor.
# Después, ya dentro del for, se suman todas las probabilidades del factor original y se crean
# dos listas más, una para las variables condicionadas y otra para las no condicionadas, además
# de sus respectivas copias auxiliares en el caso de que éstas se modifiquen.
# Una vez hecho esto, se itera por todas las variables no condicionadas, y en caso de que éstas
# sólo tengan un valor, pasan de ser no condicionadas a condicionadas.
# Por último, se crea un nuevo factor en base a las listas de variables tanto condicionadas como
# no condicionadas, donde se calcula, para todas las combinaciones posibles, la probabilidad de
# que ocurra dicha combinación, la cual se añade al nuevo factor.

def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print("Factor failed normalize typecheck: ", factor)
            raise ValueError("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    "*** MY CODE STARTS HERE ***"
    totalProb = 0.0
    assignments = factor.getAllPossibleAssignmentDicts()
    for assignment in assignments:
        probF = factor.getProbability(assignment)
        totalProb += probF
        unconditionedVar = factor.unconditionedVariables()
        conditionedVar = factor.conditionedVariables()
        unconditionedVar_aux = unconditionedVar.copy()
        conditionedVar_aux = conditionedVar.copy()
        for var in unconditionedVar:
            values = factor.variableDomainsDict()[var]
            n_values = len(values)
            if n_values == 1:
                unconditionedVar_aux.remove(var)
                conditionedVar_aux.add(var)
    newFactor = Factor(unconditionedVar_aux, conditionedVar_aux, variableDomainsDict)
    assignments = factor.getAllPossibleAssignmentDicts()
    for assignment in assignments:
        probability_norm = factor.getProbability(assignment) / totalProb
        newFactor.setProbability(assignment, probability_norm)
    return newFactor    
    "*** MY CODE FINISH HERE ***"
    # util.raiseNotDefined()
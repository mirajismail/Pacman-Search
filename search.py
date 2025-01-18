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
from game import Directions
from typing import List

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

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
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
    # stack that stores a tuple of (state, path_to_state)
    return graphSearch(problem, util.Stack)

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    # queue that stores a tuple of (state, path_to_state)
    return graphSearch(problem, util.Queue)

def priorityFunc(graph_node):
    # we store the cost in the last position
    return graph_node[-1]

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    return graphSearch(problem, lambda: util.PriorityQueueWithFunction(priorityFunc))

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearchRepeatExpansion(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    priorityQueue = util.PriorityQueue()
    start_state = problem.getStartState()
    priorityQueue.push((start_state, 0, []), 1)
    visited = set()
    best_path = []
    lowest_cost = {}
    lowest_cost[start_state] = 0
    while not priorityQueue.isEmpty():
        curr_state, curr_cost, curr_path = priorityQueue.pop()
        if curr_state not in lowest_cost or curr_cost < lowest_cost[curr_state]:
            lowest_cost[curr_state] = curr_cost
        if problem.isGoalState(curr_state):
            if (curr_cost == lowest_cost[curr_state]):
                print("new best path", curr_path, curr_cost)
                best_path = curr_path
            continue
        succs = problem.getSuccessors(curr_state)
        for succ_state, succ_dir, succ_cost in succs:
            cost = succ_cost + curr_cost
            if succ_state not in lowest_cost or cost < lowest_cost[succ_state]:
                priority = cost + heuristic(succ_state, problem)
                succ_path = list(curr_path)
                succ_path.append(succ_dir)
                print("adding state", succ_state, "with cost", cost, "priority", priority)
                priorityQueue.update((succ_state, cost, succ_path), priority)
    return best_path

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    return graphSearch(problem, lambda: util.PriorityQueueWithFunction(priorityFunc), heuristicFunc=heuristic)


def graphSearch(problem: SearchProblem, DataStructure: any, heuristicFunc=nullHeuristic) -> List[Directions]:
    container = DataStructure()
    container.push((problem.getStartState(), 0, [], 1))
    visited = set()
    while not container.isEmpty():
        curr_state, curr_cost, curr_path, _ = container.pop()
        if problem.isGoalState(curr_state):
            return curr_path
        if curr_state in visited:
            continue
        visited.add(curr_state)
        succs = problem.getSuccessors(curr_state)
        for succ_state, succ_dir, succ_cost in succs:
            if True or succ_state not in visited:
                succ_path = list(curr_path)
                succ_path.append(succ_dir)
                cost = succ_cost + curr_cost
                priority = cost + heuristicFunc(succ_state, problem)
                container.push((succ_state, cost, succ_path, priority))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

import queue

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
Pacman agents (in search_agents.py).
"""

from builtins import object
import util
import os

def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tiny_maze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tiny_maze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depth_first_search(problem):
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

def breadth_first_search(problem):
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

def uniform_cost_search(problem, heuristic=None):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()


# 
# heuristics
# 
def a_really_really_bad_heuristic(position, problem):
    from random import random, sample, choices
    return int(random()*1000)

def null_heuristic(state, problem=None):
    return 0

def manhattan_distance(state, problem=None):
    current_x, current_y = state
    goal_x, goal_y = problem.goal
    return abs(current_x - goal_x) + abs(current_y - goal_y)

def euclidean_heuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

def remaining_food(state, problem=None):
    from search_agents import FoodSearchProblem
    if not isinstance(problem, FoodSearchProblem):
        return 0
        
    if isinstance(problem, FoodSearchProblem):
        flatten = lambda *m: (i for n in m for i in (flatten(*n) if isinstance(n, (tuple, list)) else (n,)))
        
        position, food_grid = state
        pacman_x, pacman_y = position
        
        # 
        # remaining food heur
        # 
        return sum(flatten(list(food_grid)))

def remaining_food_times_5(state, problem=None):
    from search_agents import FoodSearchProblem
    if not isinstance(problem, FoodSearchProblem):
        return 0
        
    if isinstance(problem, FoodSearchProblem):
        flatten = lambda *m: (i for n in m for i in (flatten(*n) if isinstance(n, (tuple, list)) else (n,)))
        
        position, food_grid = state
        pacman_x, pacman_y = position
        
        # 
        # remaining food heur
        # 
        return sum(flatten(list(food_grid))) * 5
        
def manhattans_all_food(state, problem=None):
    from search_agents import FoodSearchProblem
    if not isinstance(problem, FoodSearchProblem):
        return 0
        
    if isinstance(problem, FoodSearchProblem):
        flatten = lambda *m: (i for n in m for i in (flatten(*n) if isinstance(n, (tuple, list)) else (n,)))
        
        position, food_grid = state
        pacman_x, pacman_y = position
        
        
        food_positions = []
        for column_index, each_column in enumerate(food_grid):
            for row_index, each_cell in enumerate(each_column):
                if each_cell:
                    food_positions.append((column_index, row_index))
        
        distances = [ abs(pacman_x - food_x) + abs(pacman_y - food_y) for food_x,food_y in food_positions ] + [ 0 ]
        # 
        # manhattan_distance to all food heur
        # 
        return sum(distances) # very very very not-admissible, but solves the problem
        
def manhattans_closest_food(state, problem=None):
    from search_agents import FoodSearchProblem
    if not isinstance(problem, FoodSearchProblem):
        return 0
        
    if isinstance(problem, FoodSearchProblem):
        flatten = lambda *m: (i for n in m for i in (flatten(*n) if isinstance(n, (tuple, list)) else (n,)))
        
        position, food_grid = state
        pacman_x, pacman_y = position
        food_grid = list(food_grid)
        
        food_positions = []
        for column_index, each_column in enumerate(food_grid):
            for row_index, each_cell in enumerate(each_column):
                if each_cell:
                    food_positions.append((column_index, row_index))
        
        distances = [ abs(pacman_x - food_x) + abs(pacman_y - food_y) for food_x,food_y in food_positions ] + [ 0 ]
        
        # 
        # nearest point
        # 
        return min(distances) +  sum(flatten(food_grid)) # does worse than manhattans_all_food

def greedy_chain(state, problem=None):
    from search_agents import FoodSearchProblem
    if not isinstance(problem, FoodSearchProblem):
        return 0
        
    if isinstance(problem, FoodSearchProblem):
        flatten = lambda *m: (i for n in m for i in (flatten(*n) if isinstance(n, (tuple, list)) else (n,)))
        
        position, food_grid = state
        pacman_x, pacman_y = position
        food_grid = list(food_grid)
        
        
        food_positions = []
        for column_index, each_column in enumerate(food_grid):
            for row_index, each_cell in enumerate(each_column):
                if each_cell:
                    food_positions.append((column_index, row_index))
        
        distances = [ abs(pacman_x - food_x) + abs(pacman_y - food_y) for food_x,food_y in food_positions ] + [ 0 ]
        
        # 
        # nearest chain (greedy)
        # 
        remaining_nodes = list(food_positions)
        distances = [ abs(pacman_x - food_x) + abs(pacman_y - food_y) for food_x,food_y in remaining_nodes ]
        cost = 0
        def max_index(iterable):
            iterable = tuple(iterable)
            if len(iterable) == 0:
                return None
            max_value = max(iterable)
            from random import sample
            options = tuple( each_index for each_index, each in enumerate(iterable) if each == max_value )
            return sample(options, 1)[0]
        
        while len(remaining_nodes):
            index_of_closest = max_index([ -each for each in distances])
            remaining_nodes.pop(index_of_closest)
            cost += distances.pop(index_of_closest)
            distances = [ abs(pacman_x - food_x) + abs(pacman_y - food_y) for food_x,food_y in remaining_nodes ]
        return cost

def discounted_cluster_gravity(state, problem=None):
    from search_agents import FoodSearchProblem
    if not isinstance(problem, FoodSearchProblem):
        return 0
        
    if isinstance(problem, FoodSearchProblem):
        flatten = lambda *m: (i for n in m for i in (flatten(*n) if isinstance(n, (tuple, list)) else (n,)))
        
        position, food_grid = state
        pacman_x, pacman_y = position
        food_grid = list(food_grid)
        
        
        food_positions = []
        for column_index, each_column in enumerate(food_grid):
            for row_index, each_cell in enumerate(each_column):
                if each_cell:
                    food_positions.append((column_index, row_index))
        
        distances = [ abs(pacman_x - food_x) + abs(pacman_y - food_y) for food_x,food_y in food_positions ] + [ 0 ]
        
        zipped = [ dict(position=a, distance=b) for a,b in zip(food_positions, distances)]
        zipped.sort(reverse=False, key=lambda each: each["distance"])
        sorted_food_positions = [ each["position"] for each in zipped ]
        
        # 
        # graviation to clusters
        # 
        def cost_of_nearest(x,y):
            return min([ abs(x - food_x) + abs(y - food_y) for food_x,food_y in sorted_food_positions ] + [ 0 ])
        
        return sum(cost_of_nearest(x,y)*(1/(index+1)) for index, (x,y) in enumerate([position]+sorted_food_positions))

def heuristic1(state, problem=None):
    from search_agents import FoodSearchProblem
    
    # 
    # heuristic for the find-the-goal problem
    # 
    if isinstance(problem, SearchProblem):
        # data
        pacman_x, pacman_y = state
        goal_x, goal_y     = problem.goal
        
        # YOUR CODE HERE (set value of optimisitic_number_of_steps_to_goal)
        
        optimisitic_number_of_steps_to_goal = 0
        return optimisitic_number_of_steps_to_goal
    # 
    # traveling-salesman problem (collect multiple food pellets)
    # 
    elif isinstance(problem, FoodSearchProblem):
        # the state includes a grid of where the food is (problem isn't ter)
        position, food_grid = state
        pacman_x, pacman_y = position
        food_grid = list(food_grid)
        
        # YOUR CODE HERE (set value of optimisitic_number_of_steps_to_goal)
        
        optimisitic_number_of_steps_to_goal = 0
        return optimisitic_number_of_steps_to_goal

def a_star_search(problem, heuristic=heuristic1):
    opened_states = util.PriorityQueue()  # Stores states that need to be expanded for Uniform Cost Search.
    current_path = util.PriorityQueue()  # Stores path of expanded states.
    closed_states = []  # Stores states that have been expanded.
    final_path = []  # Store final path of states.

    opened_states.push(problem.get_start_state(), 0)
    current_state = opened_states.pop()  # Current State.
    while not problem.is_goal_state(current_state):  # Search until goal state.
        if current_state not in closed_states:  # New state found.
            closed_states.append(current_state)  # Add state to closed_states.

            for state, action, cost in problem.get_successors(current_state):  # To calculate costs of successors of current state.
                path_cost = cost + heuristic(state, problem)
                if state not in closed_states:  # If successor is a new state add to opened_states queue and store path.
                    opened_states.push(state, path_cost)
                    current_path.push(final_path + [action], path_cost)

        current_state = opened_states.pop()  # Update current state.
        final_path = current_path.pop()  # Add to final path.

    return final_path


# (you can ignore this, although it might be helpful to know about)
# This is effectively an abstract class
# it should give you an idea of what methods will be available on problem-objects
class SearchProblem(object):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, step_cost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'step_cost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()

# if os.path.exists("./hidden/search.py"): from hidden.search import *
# fallback on a_star_search
for function in [breadth_first_search, depth_first_search, uniform_cost_search, ]:
    try: function(None)
    except util.NotDefined as error: exec(f"{function.__name__} = a_star_search", globals(), globals())
    except: pass

# Abbreviations
bfs   = breadth_first_search
dfs   = depth_first_search
astar = a_star_search
ucs   = uniform_cost_search

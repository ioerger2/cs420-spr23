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
        
        # YOUR CODE HERE (set value of optimisitic_number_of_steps_to_goal)
        
        optimisitic_number_of_steps_to_goal = 0
        return optimisitic_number_of_steps_to_goal

######################################
#TRI
# > python3 pacman.py -l food_search_1 -p SearchAgent -a prob=FoodSearchProblem 

import itertools
unique = itertools.count()

# can initialize with just a state alone, or with state+action+parent

class Node:
  def __init__(self,state=None,action=None,parent=None): 
    self.state = state
    self.parent = parent
    self.action = action

    self.depth = 0 if parent==None else parent.depth+1
    position, food = self.state
    self.loc = position
    self.nfood = sum([1 if food[x][y] else 0 for x in range(food.width) for y in range(food.height)])

    #self.priority = (self.nfood,next(unique)) # unique is a counter for breaking ties in PriorityQueue; this is just h(n)
    #self.priority = (self.depth+self.sum_food_dists(),next(unique)) # add dists from pacman to each food pellet (doesn't work well for fs2 and fs3, which are sparse)
    #self.priority = (self.depth+self.nfood+self.closest_food(),next(unique)) # add dists from pacman to each food pellet (doesn't work well for fs2 and fs3, which are sparse)

    self.priority = (self.depth+5*self.nfood,next(unique)) # f(n)=g(n)+h(n); this works well!

    #size=food.width*food.height
    #frac = self.nfood/float(size)
    #self.priority = (frac*self.nfood+(1-frac)*self.sum_food_dists(),next(unique)) # mixture

  def closest_food(self):
    coords,food = self.state
    x,y = coords
    food_list = self.get_food_list()
    dist = 99999
    for (i,j) in food_list:
      d = abs(x-i)+abs(y-j)
      if d<dist: dist = d
    return dist

  def sum_food_dists(self):
    coords,food = self.state
    x,y = coords
    food_list = self.get_food_list()
    dist = 0
    for (i,j) in food_list:
      dist += abs(x-i)+abs(y-j)
    return dist

  def is_goal(self):
    # count food pellets
    return self.nfood==0

  def get_path(self):
    if self.parent==None: return []
    return self.parent.get_path()+[self.action]

  # key includes coords of pacman, plus list of food locations

  def get_key(self):
    return "x=%s,y=%s,f=%s" % (self.loc[0],self.loc[1],str(self.get_food_list()))
    # could also do it this way, since food has Grid.__hash__()
    #key = "x=%s,y=%s,f=%s" % (self.loc[0],self.loc[1],str(self.get_food_list()))
    #print("get_key(): %s | %s" % (self.state,key ))
    #return self.state 

  def get_food_list(self):
    position,food = self.state
    food_positions = []
    for x in range(food.width):
      for y in range(food.height):
        if food[x][y]: food_positions.append((x,y))
    return food_positions

  def __lt__(self,other): return self.priority < other.priority

def breadth_first_search(problem):

  start_state = problem.get_start_state()
  pos,food = start_state
  nfood = sum([1 if food[x][y] else 0 for x in range(food.width) for y in range(food.height)])
  print("initial food pellets: %s" % nfood)

  visited = {}
  frontier = queue.Queue()
  frontier.put(Node(state=start_state))

  while not frontier.empty():
    node = frontier.get()
    print("depth=%s, key=%s" % (node.depth,node.get_key()))
    if node.is_goal(): path = node.get_path(); print("solution: %s" % (str(path))); return path
    transitions = problem.get_successors(node.state) # transition objects have .state and .action
    children = [Node(state=x.state,action=x.action,parent=node) for x in transitions]
    for child in children: 
      key = child.get_key()
      if key not in visited:
        visited[key] = 1
        frontier.put(child)

  print("no solution found :-(")
  return []

  #example_path = [  transitions[0].action  ]
  #path_cost = problem.get_cost_of_actions(example_path)
  #return example_path

def a_star_search(problem,heuristic=heuristic1):
  #return breadth_first_search(problem)

  start_state = problem.get_start_state()
  pos,food = start_state
  nfood = sum([1 if food[x][y] else 0 for x in range(food.width) for y in range(food.height)])
  print("initial food pellets: %s" % nfood)

  visited = {}
  frontier = queue.PriorityQueue()
  frontier.put(Node(state=start_state))

  while not frontier.empty():
    node = frontier.get()
    print("priority=%s, depth=%s, pos=%s, nfood=%s" % (node.priority,node.depth,node.state[0],node.nfood))
    #print("priority=%s, depth=%s, key=%s" % (node.priority,node.depth,node.get_key()))
    if node.is_goal(): path = node.get_path(); print("solution: %s" % (str(path))); return path
    transitions = problem.get_successors(node.state) # transition objects have .state and .action
    children = [Node(state=x.state,action=x.action,parent=node) for x in transitions]
    for child in children: 
      key = child.get_key()
      if key not in visited:
        visited[key] = 1
        frontier.put(child)

  print("no solution found :-(")
  return []




######################################

    #def a_star_search(problem, heuristic=heuristic1):
    #"""Search the node that has the lowest combined cost and heuristic first."""
    #"*** YOUR CODE HERE ***"
    
    # What does this function need to return?
    #     list of actions that reaches the goal
    # 
    # What data is available?
    #     start_state = problem.get_start_state() # returns a string
    # 
    #     problem.is_goal_state(start_state) # returns boolean
    # 
    #     transitions = problem.get_successors(start_state)
    #     transitions[0].state
    #     transitions[0].action
    #     transitions[0].cost
    # 
    #     print(transitions) # would look like the list-of-lists on the next line
    #     [
    #         [ "B", "0:A->B", 1.0, ],
    #         [ "C", "1:A->C", 2.0, ],
    #         [ "D", "2:A->D", 4.0, ],
    #     ]
    # 
    # Example:
    #     start_state = problem.get_start_state()
    #     transitions = problem.get_successors(start_state)
    #     example_path = [  transitions[0].action  ]
    #     path_cost = problem.get_cost_of_actions(example_path)
    #     return example_path
    
    #util.raise_not_defined()

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

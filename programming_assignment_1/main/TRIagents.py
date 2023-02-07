# pacman_agents.py
# ---------------
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

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy
import heapq

##############################
# reads map, plans minimal route to collect all food

class Node:
  def __init__(self,action,coords,food_list,parent): 
    self.action = action
    self.coords = coords 
    self.food_list = food_list
    self.parent = parent # None if root
    self._score = self.calc_score() # cache
    if parent==None: self.depth = 0
    else: self.depth = 1+parent.depth
  def extract_path(self):
    if self.parent==None: return []
    prev = self.parent.extract_path()
    return prev+[self.action]
  def calc_score(self): return len(self.food_list) # heuristic (511 final score on medium_classic), 2455 on orig/food3
  def score(self): return self._score # heuristic (511 final score on medium_classic), 2455 on orig/food3
  def __lt__(self,other): return self.score()<other.score()
  #def score(self): return 0 # doesn't work, interferes with visited
  #def score(self): return -self.depth # meanders, like DFS (406)
  #def score(self): return self.depth # nice, like BFS (502)
  
class priority_queue:
  def __init__(self): self.heap = []
  def pop(self): x = self.heap[0]; heapq.heappop(self.heap); return x[1]
  def push(self,item): heapq.heappush(self.heap,(item.score(),item))
  def empty(self): return len(self.heap)==0

class AstarAgent(game.Agent):

    def __init__(self):
      self.plan = None

    def get_action(self, state):
      if self.plan==None: plan = self.make_plan(state)
      if self.step==len(self.plan): return Directions.STOP
      else: dir = self.plan[self.step]; self.step += 1; print("%s %s" % (self.step,dir)); return dir

    def make_plan(self,state):
      self.food_map = state.get_food() # returns a Grid which is a 2x2 array of bool
      self.wall_map = state.get_walls()
      ncols,nrows = self.food_map.width,self.food_map.height
      curr = state.get_pacman_position() # (i,j) for pacman's current location

      food_list = self.get_food_coords(self.food_map)
      plan = self.Astar(curr,food_list) # returns a list of actions
      print(plan)
      self.plan,self.step = plan,0
  
    def Astar(self,coords,food_list):
      visited = {}
      PQ = priority_queue()
      init = Node(None,coords,food_list,None)
      PQ.push(init)
      iter = 0
      while not PQ.empty():
        iter += 1
        if iter%100==0: print(iter)
        node = PQ.pop() # lowest score first
        food_list = node.food_list
        if node.coords in food_list: i = food_list.index(node.coords); food_list = food_list[:i]+food_list[i+1:]
        nfood = len(food_list)
        if nfood==0: # goal state
          plan = node.extract_path()
          print("found goal! iterations=%s, visited states=%s, plan length=%s" % (iter,len(visited.keys()),len(plan)))
          return plan
        for (action,coords) in self.possible_actions(node.coords,self.wall_map):
          if (coords,nfood) in visited: continue
          visited[(coords,nfood)] = 1
          neigh = Node(action,coords,food_list,node) 
          PQ.push(neigh)
      print("fail")

    # return a list of (action,coords) that are not a wall or out-of-bounds
    def possible_actions(self,coords,wall_map):
      i,j = coords
      ncols,nrows = wall_map.width,wall_map.height
      moves = [(Directions.NORTH,0,1),(Directions.SOUTH,0,-1),(Directions.WEST,-1,0),(Directions.EAST,1,0)]
      neighbors = []
      for (Dir,Di,Dj) in moves:
        i2,j2 = i+Di,j+Dj
        if i2<0 or i2>=ncols or j2<0 or j2>=nrows: continue
        if wall_map[i2][j2]: continue
        neighbors.append((Dir,(i2,j2)))
      return neighbors

    def get_food_coords(self,food_map):
      ncols,nrows = food_map.width,food_map.height
      coords = []
      for i in range(ncols):
        for j in range(nrows):
          if food_map[i][j]: coords.append((i,j))
      return coords

    # return a list of coord pairs that are not a wall or out-of-bounds
    def get_neighbors(self,i,j,wall_map):
      ncols,nrows = wall_map.width,wall_map.height
      deltas = [(-1,0,),(1,0),(0,1),(0,-1)]
      neighbors = []
      for (Di,Dj) in deltas:
        i2,j2 = i+Di,j+Dj
        if i2<0 or i2>=ncols or j2<0 or j2>=nrows: continue
        if wall_map[i2][j2]==-1: continue
        neighbors.append((i2,j2))
      return neighbors

    def print_grid_f(self,grid):
      ncols,nrows = grid.width,grid.height
      for j in range(nrows): 
        s = ""
        for i in range(ncols):
          s += "%5.2f " % (grid[i][nrows-j-1]) # upside down; (0,0) is bottom-left corner
        print(s)

##############################

# without requiring planning, go toward where food is, but stay away from ghosts
# make an array: mark each square with food==True as 1, no good=0, wall=-1
# do value iteration:
#   each cell gets updated with with average of its original value + value of neighbors states...

class GradientAgent(game.Agent):

    # return a list of coord pairs that are not a wall or out-of-bounds
    def get_neighbors(self,i,j,wall_map):
      ncols,nrows = wall_map.width,wall_map.height
      deltas = [(-1,0,),(1,0),(0,1),(0,-1)]
      neighbors = []
      for (Di,Dj) in deltas:
        i2,j2 = i+Di,j+Dj
        if i2<0 or i2>=ncols or j2<0 or j2>=nrows: continue
        if wall_map[i2][j2]==-1: continue
        neighbors.append((i2,j2))
      return neighbors

    def print_grid_f(self,grid):
      ncols,nrows = grid.width,grid.height
      for j in range(nrows): 
        s = ""
        for i in range(ncols):
          s += "%5.2f " % (grid[i][nrows-j-1]) # upside down; (0,0) is bottom-left corner
        print(s)

    def get_action(self, state):
        food_map = state.get_food() # returns a Grid which is a 2x2 array of bool
        wall_map = state.get_walls()
        ncols,nrows = food_map.width,food_map.height
        curr = state.get_pacman_position() # (i,j) for pacman's current location

        ghosts = state.get_ghost_states()
        ghost_positions = [x.get_position() for x in ghosts]
        scared = [1 if x.scared_timer>0 else 0 for x in ghosts]
        ghost_positions = [(int(a),int(b)) for a,b in ghost_positions] # round the coordinates

        values = food_map.copy()
        for i in range(ncols):
          for j in range(nrows): values[i][j] = 1 if values[i][j]==True else 0

        gamma = 0.9
        GHOST = 10
        for iter in range(100):
          newvalues = game.Grid(ncols,nrows) # make a copy for new array, initialized with 0
          for i in range(ncols):
            for j in range(nrows):
              if wall_map[i][j]: continue
              neigh = self.get_neighbors(i,j,wall_map)
              if len(neigh)>0:
                vals = []
                for (i2,j2) in neigh: 
                  val = food_map[i][j] # 1 for food in neighbor else 0; in center location (gets added n times, but then divided out in avg)
                  for k in range(len(ghosts)):
                    if (i2,j2)==ghost_positions[k]: 
                      if scared[k]: val += GHOST # I could scale this down with the timer
                      else: val -= GHOST
                  val += gamma*values[i2][j2]
                  vals.append(val)
                #Bellman does not work so well, because moving into ghost is never max, so ghosts create no repulsion
                #the problem is that they move, so you don't want to even get close
                GRADIENT = True
                if GRADIENT: newvalues[i][j] = numpy.mean(vals)
                else: newvalues[i][j] = max(vals) # Bellman eqns, max over neighbors
          values = newvalues

        self.print_grid_f(values)
        print("food=%s, pos=%s, ghosts: %s" % (food_map.count(True),curr,ghost_positions))
        print("===========================================")
          
        # decision policy

        vals = [values[i2][j2] for (i2,j2) in neigh]
        k = vals.index(max(vals))
        (i3,j3) = neigh[k]

        moves = [(Directions.NORTH,0,1),(Directions.SOUTH,0,-1),(Directions.WEST,-1,0),(Directions.EAST,1,0)]
        bestDir,bestVal = Directions.STOP,-999
        for (dir,Di,Dj) in moves:
          i2,j2 = curr[0]+Di,curr[1]+Dj
          if i2<0 or i2>=ncols or j2<0 or j2>=nrows: continue
          if not wall_map[i2][j2]:
            if values[i2][j2]>bestVal: bestDir,bestVal = dir,values[i2][j2]
        print(bestDir)
        return bestDir

#########################################

# this is like RightTurnAgent but prints out how much food is remaining each timestep

class FoodCountAgent(game.Agent):

    def get_action(self, state):
        # returns a Grid which is a 2x2 array of bool
        food_map = state.get_food()
        temp = []
        for row in food_map: temp += row
        food_cnt = sum([1 if x else 0 for x in temp])
        print("food=%s" % food_cnt)

        legal = state.get_legal_pacman_actions()
        current = state.get_pacman_state().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        right = Directions.RIGHT[current]
        if right in legal:
            return right
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.RIGHT[right] in legal:
            return Directions.RIGHT[right]
        return Directions.STOP


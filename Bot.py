import random
from collections import deque

class Bot():

  def __init__(self):
    pass 
  
  def move(self, data):
        # Choose a random direction to move in
        possible_moves = ["up", "down", "left", "right"]
        mv = random.choice(possible_moves)

        return mv


class mybot(Bot):
  def __init__(self):
    self.i = 0
  
  def move(self, data):
        # Choose a random direction to move in
        possible_moves = ["up", "right", "down", "left"]
        self.i += 1
        print(self.i)
        return possible_moves[self.i%4]

UNREACHABLE = 1000000
dir_to_word = ['up','right','down','left'] # e.g. (dy[1],dx[1]) = (0,1) <-> dir_to_word[1] = right
dy = [1,0,-1,0]
dx = [0,1,0,-1]
def opp(d):
  return (d+2)%4

def move_square(square, d):
  res = (0,0)
  res[0] = square[0] + dy[d]
  res[1] = square[1] + dx[d]
  return res

class node():
  def __init__(self):
    self.char = '.' # . = free, o = food, # = snake
    self.dist = UNREACHABLE # distance from my head
    self.src = -1 # which direction was used to reach this square

class EatBot(Bot):

  def __init__(self):
    pass

  def move(self, data):

    #find nearest accessible food and move towards it

    board = data['board']
    myID = data['you']['id']

    h = data['height']
    w = data['width']

    # set graph

    graph = [ [ node() for _ in range(w)] for _ in range(h)]

    for f in board['food']:
      graph[ food['y'] ][ food['x'] ].char = 'o'

    for s in board['snakes']:
      for pt in s['body']:
        graph[ pt['y'] ][ pt['x'] ].char = '#'

    # BFS on the graph

    start = data['you']['head']
    sx = start['x']
    sy = start['y']

    q = deque()
    q.append(sy)
    q.append(sx)

    graph[sy][sx].dist = 0

    best_food_distance = UNREACHABLE
    best_food = (-1,-1)

    while len(q):
      y = q.popleft()
      x = q.popleft()

      # try all moves
      for d in range(4):
        nx = x + dx[d]
        ny = y + dy[d]

        # ignore bad squares
        if nx < 0 or ny < 0 or nx == w or ny == h or graph[ny][nx].char == '#' or graph[ny][nx].dist != UNREACHABLE:
          continue

        graph[ny][nx].dist = graph[y][x].dist + 1
        graph[ny][nx].src = d

        q.append(ny)
        q.append(nx)

        if graph[ny][nx].char == 'o' and best_food_distance == UNREACHABLE:
          best_food_distance = graph[ny][nx].dist
          best_food = (ny,nx)

    # we found food, go to it
    if best_food_distance != UNREACHABLE:
      while move_square(best_food, opp( graph[ best_food[0] ][ best_food[1] ].src ) ) != (sy,sx):
        best_food = move_square(best_food, opp( graph[ best_food[0] ][ best_food[1] ].src ) )
      return dir_to_word[ graph[ best_food[0] ][best_food[1]].src ]
    else: # we didn't find any food, survive

      # move to a neighboring empty square
      for d in range(4):
        nx = sx + dx[d]
        ny = sy + dy[d]
        if nx < 0 or ny < 0 or nx == w or ny == h or graph[ny][nx].char == '#':
          continue
        return dir_to_word[d]

      # didn't find empty square, guess I'll die
      return "up"
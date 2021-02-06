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
  res = (square[0]+dy[d],square[1]+dx[d])
  return res

class node():
  def __init__(self):
    self.char = '.' # . = free, o = food, # = snake
    self.dist = UNREACHABLE # distance from my head
    self.src = -1 # which direction was used to reach this square

def make_graph(h,w,board): # given board makes empty graph
  graph = [ [ node() for _ in range(w)] for _ in range(h)]

  for f in board['food']:
    graph[ f['y'] ][ f['x'] ].char = 'o'

  for s in board['snakes']:
    for pt in s['body']:
      graph[ pt['y'] ][ pt['x'] ].char = '#'
  return graph

# modifies graph with dist/src, returns closest found food
# if risk_heads = False, treat squares next to bigger enemy snake heads as blocked
def find_food(data, graph, risk_heads = True):
  h = data['board']['height']
  w = data['board']['width']

  start = data['you']['head']
  sx = start['x']
  sy = start['y']

  if risk_heads == False:
    # block off adjacent squares near bigger enemy snake's
    my_size = len(data['you']['body'])
    for snake in data['board']['snakes']:
      if snake['id'] == data['you']['id']:
        continue
      # ignore smaller snakes
      if len(snake['body']) < my_size:
        continue

      for d in range(4):
        nx = snake['head']['x'] + dx[d]
        ny = snake['head']['y'] + dy[d]
        if nx < 0 or ny < 0 or nx == w or ny == h:
          continue
        if abs(nx-sx) + abs(ny-sy) > 1:
          continue
        graph[ny][nx].char = '#'

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

  return best_food

class EatBot(Bot):

  def __init__(self):
    pass

  def move(self, data):

    #find nearest accessible food and move towards it

    board = data['board']
    myID = data['you']['id']

    h = board['height']
    w = board['width']

    # BFS on the graph being careful
    graph = make_graph(h,w,board)
    best_food = find_food(data, graph, risk_heads = False)
    # if careful didn't work, and we're desperate, try not being careful
    if best_food == (-1,-1) and data['you']['health'] <= (h+w)//1.47:
      graph = make_graph(h,w,board)
      best_food = find_food(data,graph, risk_heads = True)


    sy = data['you']['head']['y']
    sx = data['you']['head']['x']

    # we found food, go to it
    if best_food != (-1,-1):
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
      # didn't find empty square - try risking heads

      graph = make_graph(h,w,board)
      for d in range(4):
        nx = sx + dx[d]
        ny = sy + dy[d]
        if nx < 0 or ny < 0 or nx == w or ny == h or graph[ny][nx].char == '#':
          continue
        return dir_to_word[d]

      # didn't find empty square, guess I'll die
      return "up"

class GravityBot(Bot):

  def __init__(self):
    self.EMPTY = (1,-0.1)
    self.FOOD = (5,-0.05)
    self.TAIL = (0.1,-0.1)
    self.BIGGER_HEAD = (-30,28)
    self.BLOCKED = (-0.2,0.15)

  def evaluate(self, graph, data, y, x):
    val = None

    if y < 0 or x < 0 or y == data['board']['height'] or x == data['board']['width']:
      return list(self.BLOCKED)

    if graph[y][x].char == '.':
      val = self.EMPTY
    elif graph[y][x].char == 'o':
      val = list(self.FOOD)
      val[0] += (1.01**(3*(100-data['you']['health'])))
    else:
      # head or tail
      my_size = len(data['you']['body'])
      for snake in data['board']['snakes']:
        if (x,y) == (snake['body'][-1]['x'],snake['body'][-1]['y']):
          val = self.TAIL
          break

        if snake['id'] == data['you']['id'] or len(snake['body']) < my_size:
          continue

        if (x,y) == (snake['body'][0]['x'],snake['body'][1]['y']):
          val = self.BIGGER_HEAD
          break

    if val == None:
      val = self.BLOCKED

    return list(val)

  def propagate_gravity(self, graph, data, ratings, sy, sx):

    val = self.evaluate(graph, data, sy, sx)
    h = data['board']['height']
    w = data['board']['width']
    tmp = [ [ [0,0] for _ in range(data['board']['width']) ] for _ in range(data['board']['height']) ]

    if sy >= 0 and sy < h and sx >= 0 and sx < w:
      tmp[sy][sx] = val

    q = deque()

    for d in range(4):
      nx = sx + dx[d]
      ny = sy + dy[d]
      if nx < 0 or ny < 0 or nx == w or ny == h or graph[ny][nx].char == '#':
        continue

      tmp[ny][nx] = val
      q.append(ny)
      q.append(nx)

    while len(q):
      y = q.popleft()
      x = q.popleft()

      for d in range(4):
        nx = x + dx[d]
        ny = y + dy[d]

        # ignore bad squares
        if nx < 0 or ny < 0 or nx == w or ny == h or graph[ny][nx].char == '#' or tmp[ny][nx] != [0,0]:
          continue

        new_val = tmp[y][x]
        new_val[0] += new_val[1]

        if new_val[0] * tmp[y][x][0] <= 0:
          continue

        tmp[ny][nx] = new_val
        q.append(ny)
        q.append(nx)

    for y in range(0,h):
      for x in range(0,w):
        ratings[y][x] += tmp[y][x][0]

  def move(self, data):

    board = data['board']
    myID = data['you']['id']

    h = board['height']
    w = board['width']

    # BFS on the graph being careful
    graph = make_graph(h,w,board)
    ratings = [ [0 for _ in range(w)] for _ in range(h) ]

    for y in range(-1,h+1):
      for x in range(-1,w+1):
        self.propagate_gravity(graph, data, ratings, y, x)

    sy = data['you']['head']['y']
    sx = data['you']['head']['x']

    best = -10000
    best_dir = 0

    for d in range(4):
      nx = sx + dx[d]
      ny = sy + dy[d]
      if nx < 0 or ny < 0 or nx == w or ny == h:
        continue

      if ratings[ny][nx] > best:
        best = ratings[ny][nx]
        best_dir = d

    return dir_to_word[best_dir]
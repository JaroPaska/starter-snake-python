import operator
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
    for pt in s['body'][:-1]:
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

action_name = ['up', 'down', 'right', 'left']
action_dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]

class Snake:
    def __init__(self, health, body):
        self.health = health
        self.body = body

class GameState:
    def __init__(self, width=11, height=11, snake_max_hp=100, snake_start_size=3, food_chance=15, min_food=1):
        self.dims = (width, height)
        self.snake_max_hp = snake_max_hp
        self.snake_start_size = snake_start_size
        self.food_chance = food_chance
        self.min_food = min_food
        self.snakes = []
        self.foods = set()

    @staticmethod
    def from_data(data):
        board = data['board']
        gs = GameState(width=board['width'], height=board['height'])
        foods = board['food']
        for food in foods:
            gs.foods.add((food['x'], food['y']))
        snakes = board['snakes']
        for snake in snakes:
            health = snake['health']
            body = []
            for point in snake['body']:
                body.append((point['x'], point['y']))
            gs.snakes.append(Snake(health=health, body=body))
        return gs

    def out_of_bounds(self, pos):
        if pos[0] < 0 or pos[0] >= self.dims[0]:
            return True
        if pos[1] < 0 or pos[1] >= self.dims[1]:
            return True
        return False

    def get_unoccupied(self):
        points = set()
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                points.add((x, y))
        for snake in self.snakes:
            for point in snake.body:
                points.remove(point)
        for point in self.foods:
            points.remove(point)
        return points

    def spawn_food(self, n):
        for _ in range(n):
            unoccupied = list(self.get_unoccupied())
            if len(unoccupied) > 0:
                food = unoccupied[random.randint(0, len(unoccupied) - 1)]
                self.foods.add(food)

    def maybe_spawn_food(self):
        if len(self.foods) < self.min_food:
            self.spawn_food(self.min_food - len(self.foods))
        elif random.randint(0, 99) < self.food_chance:
            self.spawn_food(1)

    def step(self, actions):
        tails = []
        for snake, action in zip(self.snakes, actions):
            snake.body.insert(0, tuple(map(operator.add, snake.body[0], action_dir[action])))
            tails.append(snake.body.pop())
            snake.health -= 1
        
        eaten = set()
        for snake, tail in zip(self.snakes, tails):
            if snake.body[0] in self.foods:
                snake.health = self.snake_max_hp
                snake.body.append(tail)
                eaten.add(snake.body[0])
        self.foods = self.foods.difference(eaten)
        
        self.maybe_spawn_food()

        elims = []
        for snake in self.snakes:
            elim = False
            if snake.health <= 0:
                elim = True
            if self.out_of_bounds(snake.body[0]):
                elim = True
            for other in self.snakes:
                if snake.body[0] in other.body[1:]:
                    elim = True
                if snake != other and snake.body[0] == other.body[0] and len(snake.body) <= len(other.body):
                    elim = True
            elims.append(elim)
        self.snakes = [snake for snake, elim in zip(self.snakes, elims) if not elim]

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

# counts connected free space from (sy,sx)
def count_space(graph, sy, sx):
  seen = set()
  q = deque()
  q.append(sy)
  q.append(sx)
  seen.add((sy,sx))

  space = 1

  while len(q):
    y = q.popleft()
    x = q.popleft()

    for d in range(4):
      ny = y + dy[d]
      nx = x + dx[d]
      if ny < 0 or nx < 0 or ny == h or nx == w or graph[ny][nx].char == '#' or (ny,nx) in seen:
        continue
      space += 1
      q.append(ny)
      q.append(nx)
      seen.add((ny,nx))

  return space


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

      graph = make_graph(h,w,board)

      biggest_component = 0
      best_dir = 0

      for d in range(4):
        ny = sy + dy[d]
        nx = sx + dx[d]
        if ny < 0 or nx < 0 or nx == w or ny == w or graph[ny][nx].char == '#':
          continue

        comp_size = count_space(graph, ny, nx)
        if comp_size > biggest_component:
          biggest_component = comp_size
          best_dir = d

        return dir_to_word[best_dir]


action_name = ['up', 'down', 'right', 'left']
action_dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]

class Snake:
    def __init__(self, idn, health, body):
        self.idn = idn
        self.health = health
        self.body = body

    def __repr__(self):
        return 'Snake[idn=' + self.idn + ', health=' + str(self.health) + ', body=' + str(self.body) + ']'

class GameState:
    def __init__(self, width=11, height=11, snake_max_hp=100, snake_start_size=3, food_chance=15, min_food=1):
        self.dims = (width, height)
        self.snake_max_hp = snake_max_hp
        self.snake_start_size = snake_start_size
        self.food_chance = food_chance
        self.min_food = min_food
        self.snakes = []
        self.foods = set()

    def __repr__(self):
        return 'GameState[snakes=' + str(self.snakes) + ', foods=' + str(self.foods) + ']'

    @staticmethod
    def from_data(data):
        board = data['board']
        gs = GameState(width=board['width'], height=board['height'])
        foods = board['food']
        for food in foods:
            gs.foods.add((food['x'], food['y']))
        snakes = board['snakes']
        for snake in snakes:
            idn = snake['id']
            health = snake['health']
            body = []
            for point in snake['body']:
                body.append((point['x'], point['y']))
            gs.snakes.append(Snake(idn=idn, health=health, body=body))
        return gs

    def out_of_bounds(self, pos):
        if pos[0] < 0 or pos[0] >= self.dims[0]:
            return True
        if pos[1] < 0 or pos[1] >= self.dims[1]:
            return True
        return False

    def unoccupied(self):
        points = set()
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                points.add((x, y))
        for snake in self.snakes:
            for point in snake.body:
                if point in points:
                    points.remove(point)
        for point in self.foods:
            if point in points:
                points.remove(point)
        return points

    def spawn_food(self, n):
        for _ in range(n):
            unoccupied = list(self.unoccupied())
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

    def ok_moves(self):
        moves = []
        for snake in self.snakes:
            ok = [True] * 4
            for action in range(4):
                dest = tuple(map(operator.add, snake.body[0], action_dir[action]))
                if self.out_of_bounds(dest):
                    ok[action] = False
                for other in self.snakes:
                    if dest in other.body[0:-1]:
                        ok[action] = False
            print('None ok!')
            if not any(ok):
                ok[0] = True
            moves.append([action for action in range(4) if ok[action]])  
        return moves

class MonteCarloBot(Bot):
    def play_out(self, gs, depth, idn, action=None):
        if len(gs.snakes) == 0:
            return 0
        if idn not in [snake.idn for snake in gs.snakes]:
            return 0.5
        if len(gs.snakes) == 1:
            return 1
        if depth == 0:
            return 0.5
        
        moves = gs.ok_moves()
        for i in range(len(moves)):
            moves[i] = random.choice(moves[i])
        
        if action != None:
            for i, snake in enumerate(gs.snakes):
                if snake.idn == idn:
                    moves[i] = action

        gs.step(moves)
        return self.play_out(gs, depth - 1, idn)

    def move(self, data):
        idn = data['you']['id']
        wins = [0, 0, 0, 0]
        counts = [0, 0, 0, 0]
        for _ in range(50):
            first_move = random.randint(0, 3)
            result = self.play_out(GameState.from_data(data), 20, idn, first_move)
            wins[first_move] += result
            counts[first_move] += 1

        wr = [-1 if counts[i] == 0 else wins[i] / counts[i] for i in range(4)]
        print(wr)
        best = wr.index(max(wr))
        return action_name[best]


'''
data = {
    'board': {
        'height': 11,
        'width': 11,
        'food': [
            {'x': 0, 'y': 0},
        ],
        'snakes': [
            {
                'id': 'me',
                'health': 3,
                'body': [
                    {'x': 1, 'y': 0},
                    {'x': 2, 'y': 0},
                    {'x': 3, 'y': 0}
                ]
            }, {
                'id': 'him',
                'health': 100,
                'body': [
                    {'x': 1, 'y': 5},
                    {'x': 2, 'y': 5},
                    {'x': 3, 'y': 5}
                ]
            }
        ]
    },
    'you': {
        'id': 'me'
    }
}

gs = GameState.from_data(data)
bot = MonteCarloBot()
print(bot.move(data))
'''
        
class GravityBot(Bot):

  def __init__(self):
    self.EMPTY = (1,-0.05)
    self.FOOD = (5,-0.25)
    self.TAIL = (0.1,-0.1)
    self.BIGGER_HEAD = (-999,975)
    self.BLOCKED = (-999,998.9)

  def evaluate(self, graph, data, y, x):
    val = None

    if y < 0 or x < 0 or y == data['board']['height'] or x == data['board']['width']:
      return list(self.BLOCKED)

    if graph[y][x].char == '.':
      val = self.EMPTY
    elif graph[y][x].char == 'o':
      val = list(self.FOOD)
      val[0] += (1.01**(5*(100-data['you']['health'])))
      val[1] = (self.FOOD[1] / self.FOOD[0]) * val[0]
      #print('food is ',val)
    else:
      # head or tail
      my_size = len(data['you']['body'])
      for snake in data['board']['snakes']:
        if (x,y) == (snake['body'][-1]['x'],snake['body'][-1]['y']):
          val = self.TAIL
          break

        if snake['id'] == data['you']['id'] or len(snake['body']) < my_size:
          continue

        if (x,y) == (snake['body'][0]['x'],snake['body'][0]['y']):
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
      if nx < 0 or ny < 0 or nx >= w or ny >= h or graph[ny][nx].char == '#':
        continue

      new_val = val[::]
      new_val[0] += new_val[1]

      if new_val[0] * val[0] <= 0:
        continue

      tmp[ny][nx] = new_val
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

        new_val = tmp[y][x][::]
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

    # for y in range(h-1,-1,-1):
    #   for x in range(w):
    #     if y == sy and x == sx:
    #       print('H',end='')
    #     print("{:.1f}".format(ratings[y][x]),end='\t')
    #   print()

    for d in range(4):
      nx = sx + dx[d]
      ny = sy + dy[d]
      if nx < 0 or ny < 0 or nx == w or ny == h:
        continue

      if ratings[ny][nx] > best:
        best = ratings[ny][nx]
        best_dir = d
    
    #print(best_dir, dir_to_word[best_dir])

    return dir_to_word[best_dir]

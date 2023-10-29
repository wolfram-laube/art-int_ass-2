from . import enc
import numpy as np

# this method generates a random maze according to prim's randomized
# algorithm
# http://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_Prim.27s_algorithm


class Level(object):
    def __init__(self, rng, size):
        self.rng = rng
        self.size = size
        self.field = np.full((self.size, self.size), enc.WALL, dtype=np.int8)
        self.costs = self.rng.randint(1, 5, self.field.shape, dtype=np.int8)

        self.start = (0, 0)

        self.deltas = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ]
        self.random_walk()
        end = np.where(self.field == enc.SPACE)
        self.end = (int(end[0][-1]), int(end[1][-1]))

        self.replace_walls_with_high_cost_tiles()

    def replace_walls_with_high_cost_tiles(self):
        # select only coordinates of walls
        walls = np.where(self.field == enc.WALL)

        n_walls = len(walls[0])

        # replace about a tenth of the walls...
        to_replace = self.rng.randint(0, n_walls, n_walls // 9)

        # ... with space, but very *costly* space (it's trap!)
        for ri in to_replace:
            x, y = walls[0][ri], walls[1][ri]
            self.field[x, y] = enc.SPACE
            self.costs[x, y] = 9

    def random_walk(self):
        frontier = list()

        sx, sy = self.start
        self.field[sx, sy] = enc.SPACE
        frontier.extend(self.get_walls(self.start))

        while len(frontier) > 0:
            current, opposing = frontier[self.rng.randint(len(frontier))]

            cx, cy = current
            ox, oy = opposing
            if self.field[ox, oy] == enc.WALL:
                self.field[cx, cy] = enc.SPACE
                self.field[ox, oy] = enc.SPACE
                frontier.extend(self.get_walls(opposing))
            else:
                frontier.remove((current, opposing))

    def in_bounds(self, position):
        x, y = position
        return x >= 0 and y >= 0 and x < self.size and y < self.size

    def get_walls(self, position):
        walls = []
        px, py = position
        for dx, dy in self.deltas:
            cx = px + dx
            cy = py + dy
            current = (cx, cy)

            ox = px + 2 * dx
            oy = py + 2 * dy
            opposing = (ox, oy)

            if (self.in_bounds(current) and self.field[cx, cy] == enc.WALL and self.in_bounds(opposing)):
                walls.append((current, opposing))
        return walls

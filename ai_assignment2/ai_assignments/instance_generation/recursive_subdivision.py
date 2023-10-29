import numpy as np
from . import enc


class Level(object):
    def __init__(self, rng, size):
        self.rng = rng
        self.size = size

        self.field = np.full((size, size), enc.SPACE, dtype=np.int8)
        self.costs = np.ones_like(self.field, dtype=np.float32)

        k = 1
        self.subdivide(self.field.view(), self.costs.view(), k, 0, 0)

        # such a *crutch*!
        # this 'repairs' dead ends. horrible stuff.
        for x in range(1, size - 1):
            for y in range(1, size - 1):
                s = 0
                s += self.field[x - 1, y]
                s += self.field[x + 1, y]
                s += self.field[x, y - 1]
                s += self.field[x, y + 1]
                if self.field[x, y] == enc.SPACE and s >= 3:
                    self.field[x - 1, y] = enc.SPACE
                    self.field[x + 1, y] = enc.SPACE
                    self.field[x, y - 1] = enc.SPACE
                    self.field[x, y + 1] = enc.SPACE

        spaces = np.where(self.field == enc.SPACE)
        n_spaces = len(spaces[0])

        n_danger = self.rng.randint(3, 7)
        dangers = self.rng.choice(range(n_spaces), n_danger, replace=False)
        for di in dangers:
            rx, ry = np.unravel_index(di, (size, size))
            const = max(1., self.rng.randint(size // 5, size // 2))
            for x in range(size):
                for y in range(size):
                    distance = np.sqrt((rx - x) ** 2 + (ry - y) ** 2)
                    self.costs[x, y] = self.costs[x, y] + (1. / (const + distance))

        self.costs = self.costs - self.costs.min()
        self.costs = self.costs / self.costs.max()
        self.costs = self.costs * 9
        self.costs = self.costs + 1
        self.costs = self.costs.astype(int)

        start_choice = 0
        end_choice = -1

        self.start = (int(spaces[0][start_choice]), int(spaces[1][start_choice]))
        self.end = (int(spaces[0][end_choice]), int(spaces[1][end_choice]))

        if self.start == self.end:
            raise RuntimeError('should never happen')

    def subdivide(self, current, costs, k, d, previous_door):
        w, h = current.shape
        random_stop = self.rng.randint(0, 10) == 0 and d > 2
        if w <= 2 * k + 1 or h <= 2 * k + 1 or random_stop:
            return

        split = previous_door
        while split == previous_door:
            split = self.rng.randint(k, w - k)
        current[split, :] = enc.WALL
        door = self.rng.randint(k, h - k)
        current[split, door] = enc.SPACE

        self.subdivide(
            current[:split, :].T,
            costs[:split, :].T,
            k,
            d + 1,
            door
        )
        self.subdivide(
            current[split + 1:, :].T,
            costs[split + 1:, :].T,
            k,
            d + 1,
            door
        )

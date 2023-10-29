from . import gridworld
import numpy as np


def get_problem(rng, size):
    dones, rewards, starts = __generate(rng, size)
    return gridworld.Gridworld(rng.randint(0, 2 ** 31), dones, rewards, starts)


def get_minimum_problem_size():
    return 3


def __generate(rng, size):
    dones = np.full((size, size), False, dtype=bool)
    # rewards = rng.randint(-9, -1, (size, size), dtype=np.int8)
    rewards = np.zeros((size, size), dtype=np.int8) - 1

    # for i in range(0, size):
    #     for j in range(0, size):
    #         rewards[i, j] = - (j * size + i)

    coordinates = []
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            coordinates.append((i, j))
    indices = np.arange(len(coordinates))

    chosen = rng.choice(indices, max(1, len(indices) // 10), replace=False)

    for c in chosen:
        x, y = coordinates[c]
        dones[x, y] = True
        rewards[x, y] = -100

    starts = np.array([[1., 0]])
    dones[-1, -1] = True
    rewards[-1, -1] = 100

    return dones, rewards, starts

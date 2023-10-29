from . import random_walk
from . import simple_2d


def get_problem(rng, size):
    level = random_walk.Level(rng, size)
    return simple_2d.Simple2DProblem(level.field, level.costs, level.start, level.end)


def get_minimum_problem_size():
    return 3

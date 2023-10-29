from . import terrain
from . import simple_2d as sim


def get_problem(rng, size):
    level = terrain.Level(rng, size)
    return sim.Simple2DProblem(level.field, level.costs, level.start, level.end)


def get_minimum_problem_size():
    return 3

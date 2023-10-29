from . import recursive_subdivision
from . import simple_2d


def get_problem(rng, size):
    level = recursive_subdivision.Level(rng, size)
    return simple_2d.Simple2DProblem(level.field, level.costs, level.start, level.end)


def get_minimum_problem_size():
    return 3

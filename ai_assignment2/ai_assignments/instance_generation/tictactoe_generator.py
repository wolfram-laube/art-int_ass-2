from . import tictactoe


def get_problem(rng, depth):
    return tictactoe.TicTacToe(rng, depth)


def get_minimum_problem_size():
    return 0

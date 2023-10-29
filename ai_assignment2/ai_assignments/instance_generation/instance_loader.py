import json
from . simple_2d import Simple2DProblem
from . tictactoe import TicTacToe
from . gridworld import Gridworld
from . training_set import TrainingSet

__DELEGATES = [
    Simple2DProblem,
    TicTacToe,
    Gridworld,
    TrainingSet
]


# this is just for delegating the decoding,
# based on what is written in the problem instance field 'type'
def from_json(jsonstring):
    data = json.loads(jsonstring)
    for delegate in __DELEGATES:
        if data['type'] == delegate.__name__:
            return delegate.from_json(jsonstring)

    # none of the delegates feel responsible?
    raise ValueError('unknown problem instance type')

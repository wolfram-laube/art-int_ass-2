class Game(object):
    def get_number_of_expanded_nodes(self):
        raise NotImplementedError()

    def get_start_node(self):
        raise NotImplementedError()

    def winner(self, node):
        raise NotImplementedError()

    def successors(self, node):
        raise NotImplementedError()

    def get_max_player(self):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    @staticmethod
    def from_json(jsonstring):
        raise NotImplementedError()


class Node(object):
    def __init__(self, parent, state, action, player, depth):
        self.parent = parent
        self.state = state
        self.action = action
        self.player = player
        self.depth = depth

    def key(self):
        # if state is composed of other stuff (dict, set, ...)
        # make it a tuple containing hashable datatypes
        # (this is supposed to be overridden by subclasses)
        return tuple(self.state) + (self.player, )

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if type(self) == type(other):
            return self.key() == other.key()
        raise ValueError('cannot simply compare two different node types')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Node(id:{}, parent:{}, state:{}, action:{}, player:{}, depth:{})'.format(
            id(self),
            id(self.parent),
            self.state,
            self.action,
            self.player,
            self.depth
        )


def get_move_sequence(end: Node):
    if end is None:
        return list()

    current = end
    reverse_sequence = []
    while current.parent is not None:
        reverse_sequence.append((current.player, current.action))
        current = current.parent
    return list(reversed(reverse_sequence))

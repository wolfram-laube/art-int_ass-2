from .. problem import Problem, Node
import numpy as np
import json


def debitify(bits):
    board = []
    for i in range(9):
        board.append(2 ** i & bits > 0)
    return np.array(board).reshape(3, 3).astype(int)


class TTTNode(Node):
    def __repr__(self):
        return 'TTTNode(id:{}, parent:{},\nstate{}:\n{}\n{}\naction:\n{}\n, cost:{}, depth:{})'.format(
            id(self),
            id(self.parent),
            self.state[2],
            debitify(self.state[0]),
            debitify(self.state[1]),
            debitify(self.action),
            self.cost,
            self.depth
        )


class TicTacToe(Problem):
    __WIN_PATTERNS = [
        [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ],
        [
            0, 0, 1,
            0, 1, 0,
            1, 0, 0
        ],
        [
            1, 1, 1,
            0, 0, 0,
            0, 0, 0
        ],
        [
            0, 0, 0,
            1, 1, 1,
            0, 0, 0
        ],
        [
            0, 0, 0,
            0, 0, 0,
            1, 1, 1
        ],
        [
            1, 0, 0,
            1, 0, 0,
            1, 0, 0
        ],
        [
            0, 1, 0,
            0, 1, 0,
            0, 1, 0
        ],
        [
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ]
    ]

    def __init__(self):
        self.n_expands = 0
        self.win_patterns = []
        for win_pattern in self.__WIN_PATTERNS:
            win_pattern_bits = 0
            for i in range(9):
                win_pattern_bits |= win_pattern[i] * 2 ** i
            self.win_patterns.append(win_pattern_bits)

        self.moves = []
        for i in range(9):
            self.moves.append(2 ** i)

        # for move in self.moves:
        #     print('debitify(move)')
        #     print(debitify(move))
        # print('self.win_patterns', self.win_patterns)

        # for wpa, wpb in zip(self.__WIN_PATTERNS, self.win_patterns):
        #     print('------------')
        #     print(np.array(wpa).reshape(3, 3))
        #     print('---')
        #     print(debitify(wpb))
        self.start_state = (0, 0, 0)

    def get_start_node(self):
        return TTTNode(None, self.start_state, None, 0, 0)

    def get_end_node(self):
        raise NotImplementedError()

    def is_end(self, node):
        A, B, c = node.state
        print('-' * 30)
        print('c', c)
        print('A')
        print(debitify(A))
        print('B')
        print(debitify(B))
        print('A | B')
        print(debitify(A | B))
        for state in self.win_patterns:
            if state & A == state:
                print('A won')
                print('state', debitify(state))
                return True
            if state & B == state:
                print('B won')
                print('state', debitify(state))
                return True
        return False

    def action_cost(self, state, action):
        return 0

    def successor(self, node, action):
        A, B, c = node.state
        if A & action > 0 or B & action > 0:
            return None  # already played on this field

        if c == 0:
            successor_state = (A | action, B, (c + 1) % 2)
        else:
            successor_state = (A, B | action, (c + 1) % 2)

        return TTTNode(
            node,
            successor_state,
            action,
            0,
            node.depth + 1
        )

    def get_number_of_expanded_nodes(self):
        return self.n_expands

    def successors(self, node):
        self.n_expands += 1
        successor_nodes = []
        for move in self.moves:
            succ = self.successor(node, move)
            if succ is not None:
                successor_nodes.append(succ)
        return successor_nodes

    def to_json(self):
        return json.dumps(dict(
            type=self.__class__.__name__,
            start_state=self.start_state
        ))

    @staticmethod
    def from_json(jsonstring):
        data = json.loads(jsonstring)

        ttt = TicTacToe()
        ttt.start_state = tuple(data['start_state'])
        return ttt


def main():
    ttt = TicTacToe()
    start = ttt.get_start_node()
    print('ttt.is_end(start)', ttt.is_end(start))

    succs = ttt.successors(start)
    for node in succs:
        print('node', node)
    print('#' * 30)
    for node in ttt.successors(succs[0]):
        print('node', node)


if __name__ == '__main__':
    main()

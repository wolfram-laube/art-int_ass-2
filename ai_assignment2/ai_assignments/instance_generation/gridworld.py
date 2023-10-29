from .. environment import Environment
import numpy as np
import json


def sample(rng, elements):
    csp = np.cumsum([elm[0] for elm in elements])
    idx = np.argmax(csp > rng.uniform(0, 1))
    return elements[idx]


class Gridworld(Environment):
    DELTAS = [
        (-1, 0),
        (+1, 0),
        (0, -1),
        (0, +1)
    ]
    NAMES = [
        'left',
        'right',
        'up',
        'down'
    ]

    def __init__(self, seed, dones, rewards, starts):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.dones = dones
        self.rewards = rewards
        self.starts = starts

        self.__compute_P()

    def reset(self):
        _, self.state = sample(self.rng, self.starts)
        return self.state

    def step(self, action):
        _, self.state, reward, done = sample(self.rng, self.P[self.state][action])
        return self.state, reward, done

    def get_n_actions(self):
        return 4

    def get_n_states(self):
        return np.prod(self.dones.shape)

    def get_gamma(self):
        return 0.99

    def __compute_P(self):
        w, h = self.dones.shape

        def inbounds(i, j):
            return i >= 0 and j >= 0 and i < w and j < h

        self.P = dict()
        for i in range(0, w):
            for j in range(0, h):
                state = j * w + i
                self.P[state] = dict()

                if self.dones[i, j]:
                    for action in range(self.get_n_actions()):
                        # make it absorbing
                        self.P[state][action] = [(1, state, 0, True)]
                else:
                    for action, (dx, dy) in enumerate(self.    DELTAS):
                        ortho_dir_probs = [
                            (0.8, dx, dy),
                            (0.1, dy, dx),
                            (0.1, -dy, -dx)
                        ]
                        # ortho_dir_probs = [
                        #     (1.0, dx, dy),
                        # ]
                        transitions = []
                        for p, di, dj in ortho_dir_probs:
                            ni = i + di
                            nj = j + dj
                            if inbounds(ni, nj):
                                # we move
                                sprime = nj * w + ni
                                done = self.dones[ni, nj]
                                reward = self.rewards[ni, nj]
                                transitions.append((p, sprime, reward, done))
                            else:
                                # stay in the same state, b/c we bounced
                                sprime = state
                                done = self.dones[i, j]
                                reward = self.rewards[i, j]
                                transitions.append((p, sprime, reward, done))

                        self.P[state][action] = transitions

    def to_json(self):
        return json.dumps(dict(
            type=self.__class__.__name__,
            seed=self.seed,
            dones=self.dones.tolist(),
            rewards=self.rewards.tolist(),
            starts=self.starts.tolist()
        ))

    @staticmethod
    def from_json(jsonstring):
        data = json.loads(jsonstring)
        return Gridworld(
            data['seed'],
            np.array(data['dones']),
            np.array(data['rewards']),
            np.array(data['starts']),
        )


def main():
    rng = np.random.RandomState(123)

    for size in range(3, 100):
        gw = Gridworld(rng, size)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2)
        ax = axes[0]
        ax.imshow(gw.dones * 1., cmap='gray_r')
        ax = axes[1]
        ax.imshow(gw.rewards, cmap='RdBu_r', vmin=-15, vmax=15)
        plt.show()


if __name__ == '__main__':
    main()

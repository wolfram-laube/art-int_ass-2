import json


# this is essentially a customized openai-gym interface,
# extended with aptly named convenience methods...
class Environment():
    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def get_n_actions(self):
        raise NotImplementedError()

    def get_n_states(self):
        raise NotImplementedError()


class Outcome():
    def __init__(self, n_episodes, policy, V=dict(), Q=dict()):
        self.n_episodes = n_episodes
        self.policy = policy
        self.V = V
        self.Q = Q

    def get_n_episodes(self):
        return self.n_episodes

    def to_json(self):
        return json.dumps(dict(
            type=self.__class__.__name__,
            n_episodes=self.n_episodes,
            policy=self.policy,
            V=self.V,
            Q=self.Q,
        ))

    @staticmethod
    def from_json(jsonstring):
        data = json.loads(jsonstring)
        return Outcome(
            data['n_episodes'],
            int_keys(data['policy']),
            int_keys(data['V']),
            int_keys(data['Q'])
        )


def int_keys(d):
    nd = dict()
    if d is not None:
        for k, v in d.items():
            if isinstance(v, dict):
                nd[int(k)] = int_keys(v)
            else:
                nd[int(k)] = v

    return nd


def get_flat_policy(env, policy):
    flat_policy = []
    for state in range(env.get_n_states()):
        for action in range(env.get_n_actions()):
            flat_policy.append((state, action, policy[state][action]))
    return flat_policy

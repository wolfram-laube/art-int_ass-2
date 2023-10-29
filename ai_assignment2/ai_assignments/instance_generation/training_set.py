import json
import numpy as np


class TrainingSet():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def to_json(self):
        return json.dumps(dict(
            type=self.__class__.__name__,
            X=self.X.tolist(),
            y=self.y.tolist()
        ))

    @staticmethod
    def from_json(jsonstring):
        data = json.loads(jsonstring)
        return TrainingSet(
            np.array(data['X']),
            np.array(data['y'])
        )


from sklearn.datasets import make_classification
from ai_assignments.instance_generation.training_set import TrainingSet

# X, y = make_classification(n_samples=100,
#                            n_features=2,
#                            n_informative=2,
#                            n_redundant=0,
#                            n_clusters_per_class=1)
#
# data = TrainingSet(X, y)
#
# with open('/home/verena/Repos/ai/problem_instance_generator/id3_data/test/data.json', 'w') as fh:
#     fh.write(data.to_json())

# data.to_json()

def get_problem(rng, size):
    X, y = make_classification(n_samples=size,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_clusters_per_class=1)  # TODO: maybe scale
    return TrainingSet(X,  y)


def get_minimum_problem_size():
    return 0
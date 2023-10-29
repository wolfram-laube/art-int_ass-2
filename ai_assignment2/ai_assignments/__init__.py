from . instance_generation import maze_generator
from . instance_generation import terrain_generator
from . instance_generation import room_generator
from . instance_generation import tictactoe_generator
from . instance_generation import gridworld_generator
from . instance_generation import instance_loader
from . instance_generation import training_set_generator
import importlib
import os

__GENERATOR_MAPPING = dict(
    maze=maze_generator,
    terrain=terrain_generator,
    rooms=room_generator,
    tictactoe=tictactoe_generator,
    gridworld=gridworld_generator,
    trainset=training_set_generator
)

__SOLVER_MAPPING = dict()
__ADVERSARIAL_SEARCH_MAPPING = dict()
__REINFORCEMENT_LEARNER_MAPPING = dict()
__DECISION_TREE_LEARNER_MAPPING = dict()

# FIXME, WS2021, this is the way to go; solvers
# def register_solver(mapping):
#     __SOLVER_MAPPING.update(mapping)


# FIXME, WS2021, this is the way to go; solvers
# and adv search methods should call a register method
# to register themselves with the rest of the framework.
def register_adversarial_search_method(method_name, method_class):
    __ADVERSARIAL_SEARCH_MAPPING[method_name] = method_class


def register_reinforcement_learning_method(method_name, method_class):
    __REINFORCEMENT_LEARNER_MAPPING[method_name] = method_class


def register_decision_tree_learning_method(method_name, method_class):
    __DECISION_TREE_LEARNER_MAPPING[method_name] = method_class

# FIXME, WS2021, make this the way that solvers register
# themselves with the current module as well
try:
    if os.path.exists('ai_assignments/search/adversarial'):
        import ai_assignments.search.adversarial
except Exception as se:
    # print all other exceptions (syntax errors, etc ...)
    print('#' * 40)
    import traceback
    traceback.print_exc()
    print('#' * 40)


try:
    if os.path.exists('ai_assignments/reinforcement_learning'):
        import ai_assignments.reinforcement_learning
except Exception as se:
    # print all other exceptions (syntax errors, etc ...)
    print('#' * 40)
    import traceback
    traceback.print_exc()
    print('#' * 40)


try:
    if os.path.exists('ai_assignments/decision_tree'):
        import ai_assignments.decision_tree
except Exception as se:
    # print all other exceptions (syntax errors, etc ...)
    print('#' * 40)
    import traceback
    traceback.print_exc()
    print('#' * 40)

# FIXME, WS2021, ... we'll not get around the meta stuff with importlib, are we?
# having multiple import statements like:
#
# import ai_assignments.search.adversarial
# import ai_assignments.reference_implementations.adversarial
#
# only works if the previous ones are found! it's not really such a big problem,
# as currently there are only two modules (student impls, and reference impls ...)
try:
    if os.path.exists('ai_assignments/reference_implementations/adversarial'):
        import ai_assignments.reference_implementations.adversarial
except Exception as se:
    # print all other exceptions (syntax errors, etc ...)
    print('#' * 40)
    import traceback
    traceback.print_exc()
    print('#' * 40)


try:
    if os.path.exists('ai_assignments/reference_implementations/reinforcement_learning'):
        import ai_assignments.reference_implementations.reinforcement_learning
except Exception as se:
    # print all other exceptions (syntax errors, etc ...)
    print('#' * 40)
    import traceback
    traceback.print_exc()
    print('#' * 40)


try:
    if os.path.exists('ai_assignments/reference_implementations/decision_tree'):
        import ai_assignments.reference_implementations.decision_tree
except Exception as se:
    # print all other exceptions (syntax errors, etc ...)
    print('#' * 40)
    import traceback
    traceback.print_exc()
    print('#' * 40)


# FIXME, WS2021, all this machinery is really for the sausages :/
# see much neater way above!
def try_and_get_solver(module_path):
    solver = None
    try:
        # if the reference_implementations do not exist, try to import
        # the implementations that need to be done during an assignment
        solver = importlib.import_module('ai_assignments.' + module_path)
    except Exception as se:
        print('#' * 40)
        print('Could not load problem solver module ({})!'.format(module_path))
        import traceback
        traceback.print_exc()
        print('#' * 40)

    return solver


def get_solvers(path):
    module_names = []
    module_base = os.path.split(path)[-1]

    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith('.py'):
                if not filename.startswith('__'):
                    basename = os.path.splitext(filename)[0]
                    module_names.append('.'.join([module_base, basename]))

    mapping = dict()
    for module_name in module_names:
        solver = try_and_get_solver(module_name)

        if solver is not None:
            # try to get the solver mapping
            try:
                mapping.update(solver.get_solver_mapping())
            except Exception as me:
                print('#' * 40)
                print('Unable to import solver for ({})'.format(module_name))
                import traceback
                traceback.print_exc()
                print('#' * 40)
    return mapping


def register_solver_modules():
    path = os.path.split(__file__)[0]

    reference_solver_path = os.path.join(path, 'reference_implementations')
    default_solver_path = os.path.join(path, 'search')

    for path in [reference_solver_path, default_solver_path]:
        solvers = get_solvers(path)
        __SOLVER_MAPPING.update(solvers)


register_solver_modules()


def get_problem_generators():
    return list(__GENERATOR_MAPPING.keys())


def get_solution_methods():
    return list(__SOLVER_MAPPING.keys())


def get_solution_method(name):
    return __SOLVER_MAPPING[name]


def get_problem_generator(name):
    return __GENERATOR_MAPPING[name]


def load_problem_instance(path):
    with open(path, 'r') as fh:
        return instance_loader.from_json(fh.read())


def get_adversarial_search_methods():
    return list(__ADVERSARIAL_SEARCH_MAPPING.keys())


def get_adversarial_search_method(name):
    return __ADVERSARIAL_SEARCH_MAPPING[name]


def get_reinforcement_learning_methods():
    return __REINFORCEMENT_LEARNER_MAPPING.keys()


def get_reinforcement_learning_method(name):
    return __REINFORCEMENT_LEARNER_MAPPING[name]


def get_decision_tree_learning_methods():
    return __DECISION_TREE_LEARNER_MAPPING.keys()


def get_decision_tree_learning_method(name):
    return __DECISION_TREE_LEARNER_MAPPING[name]

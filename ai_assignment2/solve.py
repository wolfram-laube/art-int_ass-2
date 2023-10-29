import ai_assignments
import argparse
import textwrap
import hashlib
import time
import copy
import os


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
        this script reads in a problem instance in JSON format, and
        applies one or more solution methods that compute solution paths
        between the start state and the end state of the problem instance.
        depending on the solver used, the paths have various different properties.
        '''),
        epilog=textwrap.dedent('''
        example usage:

        $ python solve.py test/board.json rs
        this loads the problem instance stored in 'test/board.json' and computes
        a solution path with the solver named 'rs', stored in 'test/rs.path'
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'problem_instance_name',
        type=str,
        help='the name of the file that stores the problem instance'
    )
    parser.add_argument(
        'solution_methods',
        nargs='+',
        choices=ai_assignments.get_solution_methods(),
        help='''choose one or more solution methods / search methods to solve the problem instance.

        solutions will be written to a text file in the same directory
        as the problem_instance with the name <method_name>.path.

        solution hashes will be written to a text file in the same directory
        as the problem_instance with the name <method_name>.hash'''
    )
    args = parser.parse_args()

    if not os.path.exists(args.problem_instance_name):
        print('the problem instance ({}) does not exist'.format(args.problem_instance_name))
        exit()

    problem_instance = ai_assignments.load_problem_instance(args.problem_instance_name)
    problem_instance_directory, _ = os.path.split(args.problem_instance_name)

    for solution_method_name in args.solution_methods:
        # the problem instance is copied, so that each of the solution methods
        # gets a fresh copy to solve, and all counts are reset

        current_problem_instance, solution_node, solution_path, t_diff = compute_solution(problem_instance,
                                                                                          solution_method_name)

        solution_path_as_str = ','.join(map(str, solution_path))
        solution_hash_as_str = hashlib.sha256(solution_path_as_str.encode('UTF-8')).hexdigest()


        solution_cost = float('NaN')
        if solution_node is not None:
            solution_cost = solution_node.cost

        solution_path_filename = os.path.join(
            problem_instance_directory,
            '{}.path'.format(solution_method_name)
        )
        with open(solution_path_filename, 'w') as fh:
            fh.write(solution_path_as_str)

        solution_hash_filename = os.path.join(
            problem_instance_directory,
            '{}.hash'.format(solution_method_name)
        )
        with open(solution_hash_filename, 'w') as fh:
            fh.write(solution_hash_as_str)

        # output some info, for convenience
        print('### search method: {:>8s} ############################'.format(solution_method_name))
        print('search took    {:4.2f} seconds'.format(t_diff))
        print('nodes expanded {}'.format(
            current_problem_instance.get_number_of_expanded_nodes()))
        print('path length    {}'.format(len(solution_path)))

        print('path cost      {}'.format(solution_cost))
        print('solution hash  {}'.format(
            solution_hash_as_str
        ))


def compute_solution(problem_instance, solution_method_name):
    solution_method = ai_assignments.get_solution_method(solution_method_name)()
    current_problem_instance = copy.deepcopy(problem_instance)
    t_start = time.time()
    solution_node = solution_method.solve(current_problem_instance)
    t_end = time.time()
    t_diff = t_end - t_start
    solution_path = ai_assignments.problem.get_action_sequence(solution_node)
    return current_problem_instance, solution_node, solution_path, t_diff


if __name__ == '__main__':
    main()

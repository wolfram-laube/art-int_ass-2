import ai_assignments
import numpy as np
import textwrap
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
        this script can be used to generate random problem instances.
        you can use these for testing purposes, to debug your algorithms.
        '''),
        epilog=textwrap.dedent('''
        example usage:

        $ python generate.py maze 10 test/board.json
        this generates a random 10x10 board in the shape of a maze,
        and writes the problem instance into the file 'test/board.json'.

        $ python generate.py terrain 50 test/board.json
        this generates a random 50x50 board that tries to simulate terrain
        with different movement costs per state, and writes the problem instance
        into the file 'test/board.json'

        $ python generate.py rooms 20 test/board.json
        this generates a random 20x20 board that consists of various
        rooms with doors, and writes the problem instance into
        the file 'test/board.json'
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'problem_generator',
        choices=ai_assignments.get_problem_generators(),
        help='choose a problem generator from the provided options'
    )
    parser.add_argument(
        'size',
        type=int,
        help='the size of the problem instance (minimum size depends on the problem type)'
    )
    parser.add_argument(
        'problem_instance_name',
        type=str,
        help='''the path to the file that will store the
        problem instance. (directories will be created,
        if they do not exist yet)'''
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=155853,
        help='''set this to a number in the range [0, 2**31],
        to get a different pseudo-random number generator
        '''
    )
    parser.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='''overwrite <problem_instance_name>'''
    )

    args = parser.parse_args()

    if os.path.exists(args.problem_instance_name) and not args.force:
        print('problem instance {} already exists'.format(args.problem_instance_name))
        exit(-1)


    generation_method = ai_assignments.get_problem_generator(args.problem_generator)
    minimum_size = generation_method.get_minimum_problem_size()
    if args.size < minimum_size:
        print('problem size is too small ({} < {})'.format(args.size, minimum_size))
        exit()
    problem_instance = generation_method.get_problem(
        np.random.RandomState(int(np.abs(args.seed))), args.size
    )

    path_to_file, filename = os.path.split(args.problem_instance_name)
    if path_to_file != '':
        os.makedirs(path_to_file, exist_ok=True)

    with open(args.problem_instance_name, 'w') as fh:
        fh.write(problem_instance.to_json())


if __name__ == '__main__':
    main()

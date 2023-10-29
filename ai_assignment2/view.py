import ai_assignments
from ai_assignments.utils.visualization import plot_field_and_costs
import argparse
import textwrap
import os


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
        this script lets you view a problem instance in JSON format,
        and will optionally display solution paths superimposed on
        the problem instance. the simple 2D environment for most of
        these assignments consists of two boards that encode

            SPACES in white color
            WALLS in black color

        in the upper display, and the costs for 'transitioning to a new state',
        which means stepping onto a new position on the board, in the lower
        display.
        '''),
        epilog=textwrap.dedent('''
        example usage:

        $ python view.py test/board.json test/bfs.path
        this will view the problem instance 'test/board.json', and superimpose
        the path 'test/bfs.path' on top of both views.

        $ python view.py test/board.json test/*.path
        this will view the problem instance 'test/board.json', and superimpose
        ALL the files in the directory 'test', that end in '.path'
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('problem_instance_name', type=str)
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--coords', default=False, action='store_true')
    parser.add_argument('--grid', default=False, action='store_true')
    args = parser.parse_args()

    problem = ai_assignments.load_problem_instance(args.problem_instance_name)

    sequences = []
    for path_filename in args.paths:
        name = os.path.splitext(os.path.split(path_filename)[-1])[0]
        with open(path_filename, 'r') as fh:
            sequence_string = fh.read()
            if sequence_string == '':
                print('path file {} is empty'.format(path_filename))
            else:
                sequence = sequence_string.split(',')
                sequences.append(
                    (name, problem.get_start_node(), sequence)
                )

    start_and_end = [
        ('start', 'o', [problem.get_start_node()]),
        ('end', 'o', [problem.get_end_node()])
    ]

    plot_field_and_costs(
        problem,
        sequences=sequences,
        nodes=start_and_end,
        show_coordinates=args.coords,
        show_grid=args.grid
    )


if __name__ == '__main__':
    main()

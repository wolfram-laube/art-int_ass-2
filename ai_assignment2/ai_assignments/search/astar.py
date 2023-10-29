import math
from abc import abstractmethod

from .. problem import Problem
from .. datastructures.priority_queue import PriorityQueue


# please ignore this
def get_solver_mapping():
    return dict(
        astar_ec=ASTAR_Euclidean,
        astar_mh=ASTAR_Manhattan
    )


class ASTAR(object):
    @abstractmethod
    def heuristic(self, current, goal):
        pass

    # TODO, Exercise 2:
    # implement A* search (ASTAR)
    # - use the provided PriorityQueue where appropriate
    # - to put items into the PriorityQueue, use 'pq.put(<priority>, <item>)'
    # - to get items out of the PriorityQueue, use 'pq.get()'
    # - use a 'set()' to store nodes that were already visited
    def solve(self, problem: Problem):
        start_node = problem.get_start_node()
        end_node = problem.get_end_node()

        pq = PriorityQueue()  # Priority queue for the fringe
        pq.put(0, start_node)  # Start node has 0 cost

        visited = set()  # Set to store already visited nodes
        g_values = {start_node.state: 0}  # Cost to reach each state

        while pq.has_elements():
            current_node = pq.get()

            if problem.is_end(current_node):
                return current_node

            visited.add(current_node.state)

            for successor in problem.successors(current_node):
                if successor.state not in visited:
                    tentative_g_value = g_values[current_node.state] + successor.cost
                    if successor.state not in g_values or tentative_g_value < g_values[successor.state]:
                        g_values[successor.state] = tentative_g_value
                        f_value = tentative_g_value + self.heuristic(successor, end_node)
                        pq.put(f_value, successor)

        return None


# please note that in an ideal world, the heuristics should actually be part
# of the problem definition, as it assumes domain knowledge about the structure
# of the problem, and defines a distance to the goal state


# this is the ASTAR variant with the euclidean distance as a heuristic
# it is registered as a solver with the name 'astar_ec'
class ASTAR_Euclidean(ASTAR):
    def heuristic(self, current, goal):
        cy, cx = current.state
        gy, gx = goal.state
        return math.sqrt((cy - gy) ** 2 + (cx - gx) ** 2)


# this is the ASTAR variant with the manhattan distance as a heuristic
# it is registered as a solver with the name 'astar_mh'
class ASTAR_Manhattan(ASTAR):
    def heuristic(self, current, goal):
        cy, cx = current.state
        gy, gx = goal.state
        return math.fabs((cy - gy)) + math.fabs(cx - gx)

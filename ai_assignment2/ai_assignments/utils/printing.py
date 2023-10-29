from ai_assignments.datastructures.queue import Queue

def get_height(node):
    if node is None:
        return 0
    return max(get_height(node.left_child), get_height(node.right_child)) + 1


class DummyNode():
    def __init__(self):
        self.left_child  = None
        self.right_child = None
        self.label = None
        self.split_feature = None
        self.split_point = None

def print_decision_tree(tree):
    height = get_height(tree.root)
    visited = set()
    frontier = Queue()

    lines = ['']

    previous_level = 1
    frontier.put((tree.root, 1))

    while frontier.has_elements():
        current, level = frontier.get()
        if level > previous_level:
            lines.append('')
            previous_level = level
        lines[-1] += print_node(current, height, level)
        if current not in visited:
            visited.add(current)
            if current.left_child is not None:
                frontier.put((current.left_child, level + 1))
            else:
                if level < height: frontier.put((DummyNode(), level + 1))
            if current.right_child is not None:
                frontier.put((current.right_child, level + 1))
            else:
                if level < height: frontier.put((DummyNode(), level + 1))

    for line in lines:
        print(line)
    return None


def print_node(node, height, level=1):
    node_width = 10
    n_spaces = 2 ** (height - level - 1) * node_width - node_width // 2
    if n_spaces > 0:
        text = " " * n_spaces
    else:
        text = ""

    if isinstance(node, DummyNode):
        return text + "          " + text

    if node.label is not None:
        text = text + "(    " + str(node.label) + "   )" + text
    elif node.split_feature is not None:
        text_snippet = "(x" + str(node.split_feature) + ":" + str(round(node.split_point, 2)) + ")"
        if len(text_snippet) != node_width:
            text_snippet = " " + text_snippet
        text = text + text_snippet + text

    return text

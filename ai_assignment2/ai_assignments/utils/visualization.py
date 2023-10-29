from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
from .. environment import Environment
from .. problem import Problem
from .. game import Game
from typing import Iterable
from mpl_toolkits.axes_grid1 import make_axes_locatable


# this is a convenience function that displays a simple 2D problem
# optionally it can take a list of names with start nodes and action sequences
#
# sequences = [
#     (name_a, start_node_a, [action_x, action_y, ...]),
#     (name_b, start_node_b, [action_u, action_v, ...]),
#     .
#     .
#     .
# ]
#
# optionally it can take a list of names with lists containing nodes to highlight
# nodes = [
#     (name_a, 'x', [node_x, node_y, ...]),
#     (name_b, 'o', [node_u, node_v, ...]),
#     .
#     .
#     .
# ]
def plot_field_and_costs(problem: Problem,
                         sequences: Iterable=None,
                         nodes: Iterable=None,
                         show_coordinates=False,
                         show_grid=False,
                         plot_filename=None):
    fig = plt.figure()
    field_ax, costs_ax = plot_field_and_costs_aux(fig, problem, show_coordinates, show_grid)
    if sequences is not None and len(sequences) > 0:
        plot_sequences(fig, field_ax, problem, sequences)
        plot_sequences(fig, costs_ax, problem, sequences)

    if nodes is not None and len(nodes) > 0:
        plot_nodes(fig, field_ax, nodes)

    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close(fig)
    else:
        plt.show()


def plot_sequences(fig, ax, problem, sequences):
    for (name, start_node, action_sequence), color in zip(sequences, TABLEAU_COLORS):
        draw_path(fig, ax, problem, name, start_node, action_sequence, color)

    ax.legend(
        bbox_to_anchor=(-1, 0),
        loc='upper left',
    ).set_draggable(True)


def plot_nodes(fig, ax, nodes):
    if len(nodes) > 0:
        if len(nodes[0]) == 3:
            for (name, marker, node_collection), color in zip(nodes, TABLEAU_COLORS):
                if len(node_collection) > 0:
                    draw_nodes(fig, ax, name, node_collection, color, marker)
        else:
            for name, marker, node_collection, color in nodes:
                if len(node_collection) > 0:
                    draw_nodes(fig, ax, name, node_collection, color, marker)

        ax.legend(
            bbox_to_anchor=(-1, 0),
            loc='lower left',
        ).set_draggable(True)


def draw_nodes(fig, ax, name, node_collection, color, marker):
    states = np.array([node.state for node in node_collection])
    if len(states) > 0:
        ax.scatter(states[:, 0], states[:, 1], color=color, label=name, marker=marker)


def plot_field_and_costs_aux(fig, problem, show_coordinates, show_grid,
                             field_ax=None, costs_ax=None):

    if field_ax is None:
        ax = field_ax = plt.subplot(211)
    else:
        ax = field_ax

    ax.set_title('The field')
    im = ax.imshow(problem.board.T, cmap='gray_r')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])

    if costs_ax is None:
        ax = costs_ax = plt.subplot(212, sharex=ax, sharey=ax)
    else:
        ax = costs_ax

    ax.set_title('The costs (for stepping on a tile)')
    im = ax.imshow(problem.costs.T, cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    ticks = np.arange(problem.costs.min(), problem.costs.max() + 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)

    for ax in [field_ax, costs_ax]:
        ax.tick_params(
            top=show_coordinates,
            left=show_coordinates,
            labelleft=show_coordinates,
            labeltop=show_coordinates,
            right=False,
            bottom=False,
            labelbottom=False
        )

        # Major ticks
        s = problem.board.shape[0]
        ax.set_xticks(np.arange(0, s, 1))
        ax.set_yticks(np.arange(0, s, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, s, 1), minor=True)
        ax.set_yticks(np.arange(-.5, s, 1), minor=True)

    if show_grid:
        for color, ax in zip(['m', 'w'], [field_ax, costs_ax]):
            # Gridlines based on minor ticks
            ax.grid(which='minor', color=color, linestyle='-', linewidth=1)

    return field_ax, costs_ax


def plot_environment_and_policy(env: Environment,
                                policy=None,
                                V=None,
                                Q=None,
                                show_coordinates=False,
                                show_grid=False,
                                plot_filename=None,
                                debug_info=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    dones_ax = axes[0, 0]
    rewards_ax = axes[0, 1]
    V_ax = axes[1, 0]
    Q_ax = axes[1, 1]

    dones_ax.set_title('Terminal States and Policy')
    dones_ax.imshow(env.dones.T, cmap='gray_r', vmin=0, vmax=4)

    rewards_ax.set_title('Immediate Rewards')
    rewards_ax.imshow(env.rewards.T, cmap='RdBu_r', vmin=-25, vmax=25)

    if len(policy) > 0:
        plot_policy(env, dones_ax, policy)

    w, h = env.dones.shape
    V_array = np.zeros(env.dones.shape)
    for s, v in V.items():
        sy, sx = divmod(s, w)
        V_array[sx, sy] = v
    V_ax.set_title('State Value Function $V(s)$')
    r = max(1e-13, np.max(np.abs(V_array)))
    V_ax.imshow(V_array.T, cmap='RdBu_r', vmin=-r, vmax=r)

    if debug_info:
        for s, v in V.items():
            sy, sx = divmod(s, w)
            V_ax.text(sx, sy, f'{sx},{sy}:{s}',
                      color='w', fontdict=dict(size=6),
                      horizontalalignment='center', verticalalignment='center')

    Q_ax.set_title('State Action Value Function $Q(s, a)$')
    poly_patches_q_values = draw_Q(Q_ax, Q, env, debug_info)

    def format_coord(x, y):
        for poly_patch, q_value in poly_patches_q_values:
            if poly_patch.contains_point(Q_ax.transData.transform((x, y))):
                # print('poly_patch', poly_patch)
                # print('q_value', q_value)
                return f'x:{x:4.2f} y:{y:4.2f} {q_value}'
        return f'x:{x:4.2f} y:{y:4.2f}'

    Q_ax.format_coord = format_coord

    for ax in [dones_ax, rewards_ax, V_ax, Q_ax]:
        ax.tick_params(
            top=show_coordinates,
            left=show_coordinates,
            labelleft=show_coordinates,
            labeltop=show_coordinates,
            right=False,
            bottom=False,
            labelbottom=False
        )

        # Major ticks
        s = env.dones.shape[0]
        ax.set_xticks(np.arange(0, s, 1))
        ax.set_yticks(np.arange(0, s, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, s, 1), minor=True)
        ax.set_yticks(np.arange(-.5, s, 1), minor=True)

    if show_grid:
        for color, ax in zip(['m', 'w', 'w'], [dones_ax, rewards_ax, V_ax]):
            # Gridlines based on minor ticks
            ax.grid(which='minor', color=color, linestyle='-', linewidth=1)

    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close(fig)
    else:
        plt.show()


def plot_policy(env, ax, policy):
    w, h = env.dones.shape
    xs = np.arange(w)
    ys = np.arange(h)
    xx, yy = np.meshgrid(xs, ys)

    # we need a quiver for each of the four action
    quivers = list()
    for a in range(env.get_n_actions()):
        quivers.append(list())

    # we parse the textual description of the lake
    for s in range(env.get_n_states()):
        y, x = divmod(s, w)
        if env.dones[x, y]:
            for a in range(env.get_n_actions()):
                quivers[a].append((0., 0.))
        else:
            for a in range(env.get_n_actions()):
                wdx, wdy = env.DELTAS[a]
                corrected = np.array([wdx, -wdy])
                quivers[a].append(corrected * policy[s][a])

    # plot each quiver
    for quiver in quivers:
        q = np.array(quiver)
        ax.quiver(xx, yy, q[:, 0], q[:, 1], units='xy', scale=1.5)


def draw_Q(ax, Q, env, debug_info):
    pattern = np.zeros(env.dones.shape)
    ax.imshow(pattern, cmap='gray_r')
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle, Polygon
    w, h = env.dones.shape

    r = 0
    for state, qs in Q.items():
        for a, q in qs.items():
            r = max(r, abs(q))

    r = max(1e-13, r)
    norm = Normalize(vmin=-r, vmax=r)
    cmap = plt.get_cmap('RdBu_r')
    sm = ScalarMappable(norm, cmap)

    hover_polygons = []
    for state, qs in Q.items():
        # print('qs', qs)
        y, x = divmod(state, w)
        if env.dones[x, y]:
            continue
        y += 0.5
        x += 0.5

        dx = 1
        dy = 1

        ulx = (x - 1) * dx
        uly = (y - 1) * dy

        rect = Rectangle(
            xy=(ulx, uly),
            width=dx,
            height=dy,
            edgecolor='k',
            facecolor='none'
        )
        ax.add_artist(rect)

        mx = (x - 1) * dx + dx / 2.
        my = (y - 1) * dy + dy / 2.

        ul = ulx, uly
        ur = ulx + dx, uly
        ll = ulx, uly + dy
        lr = ulx + dx, uly + dy
        m = mx, my

        up = [ul, m, ur]
        left = [ul, m, ll]
        right = [ur, m, lr]
        down = [ll, m, lr]
        action_polys = [left, right, up, down]
        for a, poly in enumerate(action_polys):
            poly_patch = Polygon(
                poly,
                edgecolor='k',
                linewidth=0.1,
                facecolor=sm.to_rgba(qs[a])
            )
            if debug_info:
                mmx = np.mean([x for x, y in poly])
                mmy = np.mean([y for x, y in poly])
                sss = '\n'.join(map(str, env.P[state][a]))
                ax.text(mmx, mmy, f'{env.NAMES[a][0]}:{sss}',
                        fontdict=dict(size=5), horizontalalignment='center',
                        verticalalignment='center')

            hover_polygons.append((poly_patch, f'{env.NAMES[a]}:{qs[a]:4.2f}'))
            ax.add_artist(poly_patch)
    return hover_polygons


def draw_path(fig, ax, problem: Problem, name, start_node, action_sequence, color):
    current = start_node
    xs = [current.state[0]]
    ys = [current.state[1]]
    us = [0]
    vs = [0]

    length = len(action_sequence)
    cost = 0
    costs = [0] * length
    for i, action in enumerate(action_sequence):
        costs[i] = current.cost
        xs.append(current.state[0])
        ys.append(current.state[1])
        current = problem.successor(current, action)
        dx, dy = problem.ACTIONS_DELTA[action]
        us.append(dx)
        vs.append(-dy)
        cost = current.cost

    quiv = ax.quiver(
        xs, ys, us, vs,
        color=color,
        label='{} l:{} c:{}'.format(name, length, cost),
        scale_units='xy',
        units='xy',
        scale=1,
        headwidth=1,
        headlength=1,
        linewidth=1,
        picker=5
    )
    return quiv


def export_equivalent_graph_to_yed_graphml(problem: Problem):
    import pyyed
    from .. instance_generation import enc
    from .. problem import Node

    g = pyyed.Graph()

    board = problem.board
    size = board.shape[0]

    scale = 160
    for x in range(size):
        for y in range(size):
            if board[x, y] == enc.SPACE:
                g.add_node(
                    '{},{}'.format(x, y),
                    x=str(x * scale),
                    y=str(y * scale),
                    width='50',
                    height='50',
                    shape='ellipse',
                    shape_fill='#FFFFFF'
                )

    all_actions = set(problem.ACTIONS_DELTA.keys())
    for x in range(size):
        for y in range(size):
            if board[x, y] == enc.SPACE:
                state = (x, y)
                actions = set()
                # if cost = 0, we automatically get the
                # cost for the step!
                current = Node(None, state, None, 0, 0)
                for succ in problem.successors(current):
                    actions.add(succ.action)
                    sx, sy = succ.state

                    g.add_edge(
                        '{},{}'.format(x, y),
                        '{},{}'.format(sx, sy),
                        label='{} (c:{})'.format(succ.action, succ.cost)
                    )

                actions_str = ','.join(map(str, sorted(list(all_actions - actions))))
                # add self edge with actions that lead to staying in the state
                g.add_edge(
                    '{},{}'.format(x, y),
                    '{},{}'.format(x, y),
                    label='{} (c:{})'.format(actions_str, problem.costs[x, y])
                )

    return g.get_graph()


def plot_search_tree(problem: Problem, fringe, closed, current, successors, tree_ax, fringe_ax):
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    G = nx.DiGraph(ordering='out')
    node_labels = dict()
    edge_labels = dict()

    def sort_key(node):
        for order, key in enumerate(problem.ACTIONS_DELTA.keys()):
            if key == node.action:
                return order

        return -1

    node_lists_ordered = [fringe, closed, current, successors]

    all_nodes = []
    for node_list in node_lists_ordered:
        all_nodes.extend(node_list)

    all_nodes = sorted(all_nodes, key=sort_key)

    for node in all_nodes:
        G.add_node(id(node), search_node=node)
        node_labels[id(node)] = 's:{},{}\nd:{}\nc:{}'.format(
            *node.state,
            node.depth,
            node.cost
        )

    # display fringe in a different ax
    fG = nx.Graph(ordering='out')
    fG_node_labels = dict()
    fringe_color = 'tab:blue'
    if len(fringe) > 0:
        for node in fringe:
            fG.add_node(id(node))
            fG_node_labels[id(node)] = 's:{},{}\nd:{}\nc:{}'.format(
                *node.state,
                node.depth,
                node.cost
            )
    else:
        fG.add_node('empty')
        fG_node_labels['empty'] = 'the fringe is empty'
        fringe_color = 'white'

    for node in all_nodes:
        if node.parent is not None:
            edge = id(node.parent), id(node)
            G.add_edge(*edge, parent_node=node.parent)
            action_cost = problem.action_cost(node.parent.state, node.action)
            edge_labels[edge] = '{}\nc:{}'.format(node.action, action_cost)

    pos = graphviz_layout(G, prog='dot')
    fG_pos = graphviz_layout(fG, prog='dot')

    node_size = 1000
    for color, node_list in zip(TABLEAU_COLORS, node_lists_ordered):
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[id(node) for node in node_list],
            node_size=node_size,
            ax=tree_ax,
            node_color=color
        )
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=tree_ax)

    nx.draw_networkx_nodes(
        fG, fG_pos,
        node_size=node_size,
        ax=fringe_ax,
        node_color=fringe_color
    )
    nx.draw_networkx_labels(fG, fG_pos, fG_node_labels, font_size=8, ax=fringe_ax)

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, node_size=node_size, ax=tree_ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=tree_ax)


def plot_game_tree(title, game: Game, nodes, annotations=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(title)
    try:
        networkx_plot_game_tree(game, nodes, ax, highlight=annotations)
    except FileNotFoundError as fnfe:
        text = ''
        text += 'program "dot" was not found.\n'
        text += 'in a shell, in your activated environment, type:\n'
        text += '$ conda install graphviz\n'
        print(text)
        ax.set_title(text)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def networkx_plot_game_tree(game: Game, nodes, tree_ax, highlight=None):
    # TODO: this needs some serious refactoring
    # use visitors for styling, for example, instead of cumbersome dicts
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox, HPacker, VPacker, TextArea

    G = nx.DiGraph(ordering='out')
    nodes_extra = dict()
    edges_extra = dict()

    def sort_key(node):
        if node.action is None:
            return (-1, -1)
        return node.action

    for node in sorted(nodes, key=sort_key):
        G.add_node(id(node), search_node=node)
        terminal, winner = game.outcome(node)
        nodes_extra[id(node)] = dict(
            board=node.state,
            player=node.player,
            depth=node.depth,
            terminal=terminal,
            winner=winner
        )

    for node in nodes:
        if node.parent is not None:
            edge = id(node.parent), id(node)
            G.add_edge(*edge, parent_node=node.parent)
            edges_extra[edge] = dict(
                label='{}'.format(node.action),
                parent_player=node.parent.player
            )

    node_size = 1000
    positions = graphviz_layout(G, prog='dot')

    from matplotlib.colors import Normalize, LinearSegmentedColormap

    blue_orange = LinearSegmentedColormap.from_list(
        'blue_orange',
        ['tab:blue', 'lightgray', 'tab:orange']
    )

    inf = float('Inf')
    x_range = [inf, -inf]
    y_range = [inf, -inf]
    for id_node, pos in positions.items():
        x, y = pos
        x_range = [min(x, x_range[0]), max(x, x_range[1])]
        y_range = [min(y, y_range[0]), max(y, y_range[1])]

        player = nodes_extra[id_node]['player']
        text_player = 'p:{}'.format(player)
        text_depth = 'd:{}'.format(nodes_extra[id_node]['depth'])
        color_player = 'tab:blue' if player == -1 else 'tab:orange'

        frameon = False
        bboxprops = None
        if nodes_extra[id_node]['terminal']:
            winner = nodes_extra[id_node]['winner']
            frameon = True
            if winner is None:
                edgecolor = 'tab:purple'
            else:
                edgecolor = 'tab:blue' if winner == -1 else 'tab:orange'
            bboxprops = dict(
                facecolor='none',
                edgecolor=edgecolor
            )
            color_player = 'k'
            text_player = 'w:{}'.format(winner)
            if winner is None:
                text_player = ''

        # needs to be transposed b/c image coordinates etc ...
        board = nodes_extra[id_node]['board'].T
        textbox_player = TextArea(text_player, textprops=dict(size=6, color=color_player))
        textbox_depth = TextArea(text_depth, textprops=dict(size=6))

        textbox_children = [textbox_player, textbox_depth]

        if highlight is not None:
            if id_node in highlight:
                if nodes_extra[id_node]['terminal']:
                    frameon = True
                    if nodes_extra[id_node]['winner'] is None:
                        edgecolor = 'tab:purple'
                    else:
                        edgecolor = 'tab:blue' if winner == -1 else 'tab:orange'

                    bboxprops = dict(
                        facecolor='none',
                        edgecolor=edgecolor
                    )

                if len(highlight[id_node]) > 0:
                    for key, value in highlight[id_node].items():
                        textbox_children.append(
                            TextArea('{}:{}'.format(key, value), textprops=dict(size=6))
                        )

        imagebox = OffsetImage(board, zoom=5, cmap=blue_orange, norm=Normalize(vmin=-1, vmax=1))
        packed = HPacker(
            align='center',
            children=[
                imagebox,
                VPacker(
                    align='center',
                    children=textbox_children,
                    sep=0.1, pad=0.1
                )
            ],
            sep=0.1, pad=0.1
        )

        ab = AnnotationBbox(packed, pos, xycoords='data', frameon=frameon, bboxprops=bboxprops)
        tree_ax.add_artist(ab)

    def min_dist(a, b):
        if a == b:
            return [a - 1, b + 1]
        else:
            return [a - 0.9 * abs(a), b + 0.1 * abs(b)]

    x_range = min_dist(*x_range)
    y_range = min_dist(*y_range)
    tree_ax.set_xlim(x_range)
    tree_ax.set_ylim(y_range)

    orange_edges = []
    blue_edges = []

    for edge, extra in edges_extra.items():
        if extra['parent_player'] == -1:
            blue_edges.append(edge)
        else:
            orange_edges.append(edge)

    for color, edgelist in [('tab:orange', orange_edges), ('tab:blue', blue_edges)]:
        nx.draw_networkx_edges(
            G, positions,
            edgelist=edgelist,
            edge_color=color,
            arrowstyle='-|>',
            arrowsize=10,
            node_size=node_size,
            ax=tree_ax
        )
    edge_labels = {edge_id: edge['label'] for edge_id, edge in edges_extra.items()}
    nx.draw_networkx_edge_labels(G, positions, edge_labels, ax=tree_ax, font_size=6)


def standalone_plot_game_tree(game: Game, nodes, tree_ax, highlight=None):
    # this standalone tree layouter is not really prime-time ready
    # needs a min-distance of nodes hacked in or something
    from .tree_layout import tree_layout_nodes
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox, HPacker, VPacker, TextArea

    positions = tree_layout_nodes(nodes)

    nodes_extra = dict()
    for node in nodes:
        terminal, winner = game.outcome(node)
        nodes_extra[id(node)] = dict(
            board=node.state,
            player=node.player,
            depth=node.depth,
            terminal=terminal,
            winner=winner
        )

    # for node in nodes:
    #     if node.parent is not None:
    #         edge = id(node.parent), id(node)
    #         G.add_edge(*edge, parent_node=node.parent)
    #         edges_extra[edge] = dict(
    #             label='{}'.format(node.action),
    #             parent_player=node.parent.player
    #         )

    # node_size = 1000
    # positions = graphviz_layout(G, prog='dot')

    from matplotlib.colors import Normalize, LinearSegmentedColormap

    blue_orange = LinearSegmentedColormap.from_list(
        'blue_orange',
        ['tab:blue', 'lightgray', 'tab:orange']
    )

    x_range = [0, 0]
    y_range = [0, 0]
    for id_node, pos in positions.items():
        x, y = pos
        x_range = [min(x, x_range[0]), max(x, x_range[1])]
        y_range = [min(y, y_range[0]), max(y, y_range[1])]

        player = nodes_extra[id_node]['player']
        text_player = 'p:{}'.format(player)
        text_depth = 'd:{}'.format(nodes_extra[id_node]['depth'])
        color_player = 'tab:blue' if player == -1 else 'tab:orange'

        frameon = False
        bboxprops = None
        if nodes_extra[id_node]['terminal']:
            winner = nodes_extra[id_node]['winner']
            frameon = True
            bboxprops = dict(
                facecolor='none',
                edgecolor='tab:blue' if winner == -1 else 'tab:orange'
            )
            color_player = 'k'
            text_player = 'w:{}'.format(winner)

        if highlight is not None:
            if id_node in highlight:
                bboxprops = dict(
                    facecolor='none',
                    edgecolor='tab:purple',
                )

        # needs to be transposed b/c image coordinates etc ...
        board = nodes_extra[id_node]['board'].T
        textbox_player = TextArea(text_player, textprops=dict(size=6, color=color_player))
        textbox_depth = TextArea(text_depth, textprops=dict(size=6))

        imagebox = OffsetImage(board, zoom=5, cmap=blue_orange, norm=Normalize(vmin=-1, vmax=1))
        packed = HPacker(
            align='center',
            children=[
                imagebox,
                VPacker(
                    align='center',
                    children=[
                        textbox_player,
                        textbox_depth
                    ],
                    sep=0.1, pad=0.1
                )
            ],
            sep=0.1, pad=0.1
        )
        ab = AnnotationBbox(packed, pos, xycoords='data', frameon=frameon, bboxprops=bboxprops)
        tree_ax.add_artist(ab)

    print('x_range', x_range)
    print('y_range', y_range)
    tree_ax.set_xlim(np.array(x_range) * 1.1)
    tree_ax.set_ylim(np.array(y_range) * 1.1)

    # orange_edges = []
    # blue_edges = []

    # for edge, extra in edges_extra.items():
    #     if extra['parent_player'] == -1:
    #         blue_edges.append(edge)
    #     else:
    #         orange_edges.append(edge)

    # for color, edgelist in [('tab:orange', orange_edges), ('tab:blue', blue_edges)]:
    #     nx.draw_networkx_edges(
    #         G, positions,
    #         edgelist=edgelist,
    #         edge_color=color,
    #         arrowstyle='-|>',
    #         arrowsize=10,
    #         node_size=node_size,
    #         ax=tree_ax
    #     )
    # edge_labels = {edge_id: edge['label'] for edge_id, edge in edges_extra.items()}
    # nx.draw_networkx_edge_labels(G, positions, edge_labels, ax=tree_ax, font_size=6)


def main():
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    G = nx.DiGraph(ordering='out')

    G.add_node(4)
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_edge(0, 2)
    G.add_edge(0, 1)
    G.add_edge(1, 3)
    G.add_edge(1, 4)

    pos = graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)
    plt.show()


if __name__ == '__main__':
    main()

from itertools import permutations
import numpy as np
import copy


class MST_node:

    def __init__(self, p, r, t):
        self.parent = p
        self.tour = t
        self.remaining = r
        self.child = []
        self.action = self.tour[-1]
        self.visit = 0
        self.score = 0
        self.child_list = []
        self.avg_score = 0


# given a tsp problem, return a root_node with:
# 1.all other nodes attached in the remaining list 
# 2. always starts at node 0
def root_node(tsp):
    r = [i + 1 for i in range(tsp.n - 1)]
    return MST_node(p=None, r=r, t=[0])


# Main function for solving tsp problem 
# start with root node, run mcts iteratively unitl hitting a leaf
def mcts_solver(Tsp, node, computation_power=3000):
    while not is_leaf(node):
        node = monte_carlo_tree_search(Tsp, node, computation_power)
    best_tour = node.tour
    mcts_score = Tsp.tour_value(best_tour)
    best_tour.append(best_tour[0])

    return best_tour, mcts_score


# Monte carlo tree search contains: 
# 1. traverse from the root node until hit leaf or a not fully expended node
# 2. run simulation for the returned leaf from step 1 until a complete tour is obtained 
# 3. backup the value of that complete tour from the leaf upward until no parents
# 4. repeat step 1-3 until computation power runs out
def monte_carlo_tree_search(Tsp, root, computation_power):
    for _ in range(computation_power):
        leaf = traverse(root)
        simulation_res = rollout(leaf, Tsp)
        backup(leaf, simulation_res)

    return best_child(root)


def traverse(node):
    while not is_leaf(node):
        if is_fully_expend(node):
            node = best_child_uct(node)
        else:
            node = expend(node)
    return node


def expend(node):
    p = next_child_prob(node)
    next_node = np.random.choice(node.remaining, p=p)
    next_tour = copy.copy(node.tour)
    next_remaining = copy.copy(node.remaining)
    next_tour.append(next_node)
    next_remaining.remove(next_node)
    child = MST_node(p=node, r=next_remaining, t=next_tour)
    node.child.append(child)
    node.child_list.append(child.action)
    return child


def rollout(node, Tsp):
    np.random.shuffle(node.remaining)

    # [0] means go back to the origin
    rand_tour = node.tour + node.remaining + [0]

    return Tsp.tour_value(rand_tour)


def backup(node, simulation_res):
    node.visit += 1
    node.score += simulation_res
    node.avg_score = node.score / node.visit
    if node.parent is not None:
        backup(node.parent, simulation_res)


def best_child(root):
    # pick the child with the highest avg score
    return max(root.child, key=lambda child: child.avg_score)


def next_child_prob(node):
    # return 1/n prob for every node if child list is empty
    if len(node.child) == 0:
        return np.ones(len(node.remaining), ) / len(node.remaining)
    else:
        p = np.ones(len(node.remaining), )
        j = 0

        # loop through child list, set prob for choosing each child to zero
        for j, i in enumerate(node.child_list):
            visited_child_pos = np.where(node.remaining == i)[0][0]
            p[visited_child_pos] = 0

        # adjust prob for other unvisited child propotionally
        return p / (len(node.remaining) - j - 1)


def is_fully_expend(node):
    return len(node.remaining) == len(node.child)


def is_leaf(node):
    if len(node.remaining) == 0 and len(node.child) == 0:
        return True
    else:
        return False


def best_child_uct(node):
    c = 1  # exploration constant
    k = np.log(node.visit)
    return max(node.child, key=lambda child: child.avg_score + c * np.sqrt(2 * k / child.visit))


# sanity check for optimal solution using full permutation
# only good for small number of nodes
def full_permutation(tsp, node_lis):
    permu = list(permutations(node_lis))
    length = []
    for i in permu:
        temp = list(i)
        temp.append(temp[0])
        tour_len = tsp.tour_length(temp)
        length.append(tour_len)
    return np.min(length)
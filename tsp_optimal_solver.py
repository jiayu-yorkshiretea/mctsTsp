from tsp_solver.greedy import solve_tsp
from mcts_solver import MST_node, mcts_solver, root_node, full_permutation
import tsp
import numpy as np

node_lis = [0, 1, 2, 3, 4]
tsp1 = tsp.TSP(20, 2)
root1 = root_node(tsp1)
best_tour1, mcts_score1 = mcts_solver(tsp1, root1, computation_power=3000)
tour_len1 = tsp1.tour_length(best_tour1)

D = tsp1.dist_mat()
# D = [
#     [],
#     [10],
#     [15, 35],
#     [20, 25, 30]
# ]

path = solve_tsp(D)
path.append(path[0])
print('The tour length found by mcts is:')
print(tour_len1)
print('The best tour found by mcts is:')
print((np.array([best_tour1]) + 1)[0])
print('The tour length found by optimal solver is:')
print(tsp1.tour_length(path))
print('The best tour found by optimal solver is:')
print((np.array([path]) + 1)[0])
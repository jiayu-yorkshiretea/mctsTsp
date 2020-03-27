import numpy as np

print('hello')


class TSP:

    def __init__(self, n, d=2):
        self.n, self.d = n, d
        # self.seed = seed
        self.points = random_tsp(n, d)

    def tour_length(self, tour):
        tours = self.points[tour]
        diffs = np.diff(tours, axis=0)
        tour_len = np.linalg.norm(diffs, axis=1, ord=2).sum()
        return tour_len

    def tour_value(self, tour):
        return ((2 * self.n) - self.tour_length(tour)) / (2 * self.n)

    def dist_mat(self):
        adj_mat = np.zeros((self.n, self.n))

        for count_i, i in enumerate(self.points):
            count_j = count_i + 1
            for j in self.points[1 + count_i:]:
                d = np.sqrt((i - j)[0] ** 2 + (i - j)[1] ** 2)
                adj_mat[count_j, count_i] = d
                count_j += 1

        distance_mat = []

        for i in range(adj_mat.shape[0]):
            distance_mat.append(adj_mat[i][0:i].tolist())

        return distance_mat


def random_tsp(n, d):
    return np.random.rand(n, d)

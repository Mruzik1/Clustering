import matplotlib.pyplot as plt
import numpy as np


class Cluster2d:
    # points - numpy array, axis 0 length is 2 (x and y)
    # clusters_count - integer value, count of clusters
    def __init__(self, points: np.ndarray, cluters_count: int):
        self.points = points
        self.clusters = np.random.uniform((0, 0), np.amax(points, axis=0), (cluters_count, 2))

    # draw a graph (if animated is True, you should call this method in a cycle; the blue lines are visible if display_nearest is True)
    def draw(self, animated=False, display_nearest=False):
        plt.scatter(self.points.T[0], self.points.T[1], color='r')
        plt.scatter(self.clusters.T[0], self.clusters.T[1], color='b')

        if display_nearest:
            nearest_idx = self.nearest_points().T[1]
            for i, e in enumerate(self.points):
                plt.plot((self.clusters[int(nearest_idx[i])][0], e[0]), (self.clusters[int(nearest_idx[i])][1], e[1]), color='b', linestyle='--')

        if animated:
            plt.draw()
            plt.pause(0.3)
            plt.clf()
        else:
            plt.show()

    # find the nearest cluster points
    # returns a list of elements: first one is ditance, the second one is a cluster's index
    def nearest_points(self) -> np.ndarray:
        result = []
        for p in self.points:
            result.append(min([(np.sqrt((p[0]-e[0])**2 + (p[1]-e[1])**2), i) for i, e in enumerate(self.clusters)]))
            
        return np.array(result)

    # replace cluster points with respect to the nearest points
    def replace_clusters(self):
        nearest_idx = self.nearest_points().T[1]

        for i in range(len(self.clusters)):
            tmp_nearest = [k for j, k in enumerate(self.points) if nearest_idx[j] == i]
            self.clusters[i] = np.sum(tmp_nearest, axis=0) / len(tmp_nearest) if len(tmp_nearest) != 0 else self.clusters[i]


# testing
if __name__ == '__main__':

    # input data (random points, some kind of clusters)
    points = np.random.uniform(0, 12, (100, 2))
    points = np.concatenate((points, np.random.uniform(16, 40, (500, 2))))
    points = np.concatenate((points, np.random.uniform(24, 60, (250, 2))))
    test_cluster = Cluster2d(points, 5)

    # training the model 
    for _ in range(15):
        test_cluster.draw(animated=True, display_nearest=True)
        test_cluster.replace_clusters()

    print('Finished!')
    test_cluster.draw(display_nearest=True)
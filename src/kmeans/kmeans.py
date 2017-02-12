import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def create_fig(width=10, height=10, dpi=100):
    fig = plt.figure()
    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)
    return fig


def create_centroids(num_centroids=8, num_dimensions=2, min_coord=-10, max_coord=10, seed=0,
                     black_list=None):
    rand = random.Random()
    rand.seed(seed)
    centroids = {}
    while (len(centroids) != num_centroids):
        c = [rand.uniform(min_coord, max_coord) for x in range(num_dimensions)]
        if black_list is not None and c not in black_list:
            centroids.append(c)
    return centroids


def create_cluster(center=(0, 0), num_dimensions=2, num_elements=50, radius=8, seed=0,
                   black_list=None):
    rand = random.Random()
    rand.seed(seed)
    cluster = set()
    while (len(cluster) != num_elements):
        c = tuple([rand.uniform(center[d] - radius / 2, center[d] + radius / 2)
                   for d in range(num_dimensions)])
        if black_list is not None and c not in black_list:
            cluster.add(c)
    return cluster


def get_samples_from_list_of_clusters(list_of_clusters):
    samples = []
    for cluster in list_of_clusters:
        for point in cluster:
            samples.append(point)
            break
    return samples


def calculate_score(clusters, actual):
    '''expected is a dict(point -> cluster_id).
    actual is a list of set of points, each set is considered a cluster.
        '''
    expected_clusters = set(expected.values())
    actual_clusters = {}
    return


if __name__ == '__main__':
    # Dem configs
    num_clusters = 3
    num_dimensions = 2
    cluster_radius = 3
    num_elements_per_cluster = 8
    seed = 0

    # Here we be creating all points
    # Sets are used to prevent duplication
    all_points = set()
    centroids = {(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)}
    all_points.update(centroids)
    clusters = []
    for c in centroids:
        cluster = create_cluster(center=c, num_dimensions=num_dimensions, num_elements=num_elements_per_cluster,
                                 radius=cluster_radius, seed=seed, black_list=all_points)
        seed += 1
        all_points.update(cluster)
    all_points = list(all_points)
    all_points = np.asarray(all_points)

    # Plot all dem points
    x = [c[0] for c in all_points]
    y = [c[1] for c in all_points]
    fig = create_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    plt.savefig("all_points.png", bbox_inches="tight")

    # Plot points colored accordin to kmeans clusters
    x = [c[0] for c in all_points]
    y = [c[1] for c in all_points]
    for cluster_count in range(2, 9):
        kmeans = KMeans(n_clusters=cluster_count, random_state=0)
        kmeans.fit(all_points)
        # Plot all dem points, coloring with kmeans colors
        c = kmeans.labels_
        fig = create_fig()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, c=c)
        plt.savefig(f"kmeans{cluster_count}.png", bbox_inches="tight")

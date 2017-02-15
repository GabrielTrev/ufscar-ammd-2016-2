import random
import itertools
from scipy import linalg
import numpy as np
from sklearn.mixture import GaussianMixture as EM
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'red', 'black'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.savefig(title + ".png")


def create_fig(width=10, height=10, dpi=100):
    """
    Creates a matplotlib figure.
    """
    fig = plt.figure()
    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)
    return fig


def create_cluster(center=(0, 0), num_dimensions=2, num_elements=50, radius=8, seed=0,
                   black_list=None):
    """
    black_list is meant to be a set of points.
    The created cluster is guaranteed to not contains points that are in black_list.
    """
    assert (num_dimensions > 0)
    assert (num_elements >= 0)
    assert (radius > 0)
    rand = random.Random()
    rand.seed(seed)
    cluster = set()
    while (len(cluster) != num_elements):
        c = tuple([rand.uniform(center[d] - radius / 2, center[d] + radius / 2)
                   for d in range(num_dimensions)])
        if black_list is not None and c not in black_list:
            cluster.add(c)
    return cluster


def calculate_accuracy(correct_clusters, ordered_points, ordered_labels):
    """
    :param correct_clusters: A dict (point -> cluster_id).
    :param ordered_points: The points used by KMeans to create the clusters.
    :param ordered_labels: The labels, created by KMeans, associated to each each point in ordered points.
    """
    assert (len(correct_clusters) > 0)
    assert (len(correct_clusters) == len(ordered_points))
    assert (len(correct_clusters) == len(ordered_labels))

    # Create a mapping of the labels I created and the ones KMeans created
    kmeans_to_mine = dict()
    mine_to_kmeans = dict()
    for i in range(len(ordered_points)):
        pt = tuple(ordered_points[i])
        kmeans_label = ordered_labels[i]
        my_label = correct_clusters[pt]
        mine_to_kmeans[my_label] = kmeans_label
        kmeans_to_mine[kmeans_label] = my_label

    # Calculate the score
    matches = 0
    total = len(correct_clusters)
    for i in range(len(ordered_points)):
        pt = tuple(ordered_points[i])
        kmeans_label = ordered_labels[i]
        my_label = correct_clusters[pt]
        mtk = mine_to_kmeans[my_label]
        ktm = kmeans_to_mine[kmeans_label]
        if my_label == ktm and kmeans_label == mtk:
            matches += 1
    return matches / total


if __name__ == '__main__':
    # Dem configs
    centroids = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)]
    number_of_clusters_to_try = [3, 4, 5, 6, 7]
    # The number of dimensions we'are playing with
    num_dimensions = len(centroids[0])
    # When creating a cluster, points are created in a radius this big around a centroid
    cluster_radius = 3
    num_elements_per_cluster = 15
    seed = 0
    centroids = set(centroids)

    # Sanity check
    assert (all(map(lambda x: len(x) == num_dimensions, centroids)))

    # Create the clusters and assign a id to each one.
    all_points = set(centroids)
    current_cluster_id = 0
    point_to_cluster = dict()
    for center in centroids:
        # Create the cluster around the centroid
        cluster = create_cluster(center=center, num_dimensions=num_dimensions, num_elements=num_elements_per_cluster,
                                 radius=cluster_radius, seed=seed, black_list=all_points)
        # Update the point_to_cluster dict
        point_to_cluster[center] = current_cluster_id
        for pt in cluster:
            point_to_cluster[pt] = current_cluster_id
        # Update stuff for the next iteration
        seed += 1
        current_cluster_id += 1
        all_points.update(cluster)

    # Workaround to get a ndarray from the set of all points
    all_points = list(all_points)
    all_points = np.asarray(all_points)

    # If we're working with 2d, plot all dem points
    x = [c[0] for c in all_points]
    y = [c[1] for c in all_points]
    if num_dimensions == 2:
        fig = create_fig()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y)
        plt.savefig("all_points.png", bbox_inches="tight")

    # Use EM to find dem clusters
    for cluster_count in number_of_clusters_to_try:
        em = EM(n_components=cluster_count, covariance_type='full', init_params='kmeans', random_state=0, max_iter=100)
        em.fit(X=all_points)
        label = em.predict(all_points)
        # for p in blargh:
        #    print(p)

        # Create a dict (point -> guessed_cluster)
        labels = list(label)
        accuracy = calculate_accuracy(point_to_cluster, all_points, labels)
        print(f"Accuracy of EM using {cluster_count} clusters:{accuracy}")

        # Again, if we're workin in 2d:
        # Plot all dem points, coloring with kmeans colors
        if num_dimensions == 2:
            fig = create_fig()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x, y, c=labels)
            plot_results(all_points, label, em.means_, em.covariances_, 0, f"Nunvezinhas{cluster_count}")
            # plt.savefig(f"em{cluster_count}.png", bbox_inches="tight")

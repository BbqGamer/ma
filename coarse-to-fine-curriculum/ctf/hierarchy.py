from __future__ import annotations

import logging
from typing import Iterable

import numpy as np


def _find_closest_cluster_min(
    dist_matrix: np.ndarray,
    clusters: list[list[int]],
    node_to_cluster: np.ndarray,
) -> tuple[list[int | None], list[float]]:
    n = dist_matrix.shape[0]
    closest_cluster: list[int | None] = []
    closest_cluster_dists: list[float] = []

    for cluster in clusters:
        cluster_nodes = np.asarray(cluster)
        cluster_mask = np.zeros((n,), dtype=bool)
        cluster_mask[cluster_nodes] = True
        non_cluster_mask = np.logical_not(cluster_mask)
        non_cluster_nodes = np.where(non_cluster_mask)[0]
        dists = dist_matrix[cluster_mask][:, non_cluster_mask]
        if dists.size == 0:
            closest_cluster.append(None)
            closest_cluster_dists.append(float("inf"))
            continue
        if dists.ndim < 2:
            if len(cluster_nodes) == 1:
                dists = dists[None]
            else:
                dists = dists[:, None]
        min_dists = np.amin(dists, axis=1)
        cluster_node_id = int(np.argmin(min_dists))
        non_cluster_node_id = int(np.argmin(dists[cluster_node_id]))
        smallest_dist = float(dists[cluster_node_id][non_cluster_node_id])
        non_cluster_node_id = int(non_cluster_nodes[non_cluster_node_id])
        closest_cluster.append(int(node_to_cluster[non_cluster_node_id]))
        closest_cluster_dists.append(smallest_dist)

    return closest_cluster, closest_cluster_dists


def find_closest_cluster(
    dist_matrix: np.ndarray,
    clusters: list[list[int]],
) -> tuple[list[int | None], list[float]]:
    n = dist_matrix.shape[0]
    node_to_cluster = np.zeros((n,), dtype=np.int64)
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_id
    return _find_closest_cluster_min(dist_matrix, clusters, node_to_cluster)


def merge_clusters(
    clusters: list[list[int]],
    closest_cluster: list[int | None],
    num_clusters: int,
) -> list[list[int]]:
    color = ["not_visited" for _ in range(num_clusters)]
    cluster_to_set = np.zeros((num_clusters,), dtype=np.int64)
    cluster_sets: list[set[int]] = []

    for i in range(num_clusters):
        if color[i] != "not_visited":
            continue
        current_set = {i}
        color[i] = "in_progress"
        j = i
        while (
            closest_cluster[j] is not None
            and color[int(closest_cluster[j])] == "not_visited"
        ):
            next_cluster = int(closest_cluster[j])
            current_set.add(next_cluster)
            color[next_cluster] = "in_progress"
            j = next_cluster

        if closest_cluster[j] is None or color[int(closest_cluster[j])] == "in_progress":
            set_index = len(cluster_sets)
            cluster_sets.append(current_set)
        else:
            set_index = int(cluster_to_set[int(closest_cluster[j])])
            cluster_sets[set_index].update(current_set)

        for elem in current_set:
            color[elem] = "visited"
            cluster_to_set[elem] = set_index

    return [
        [node for cluster_index in cluster_set for node in clusters[cluster_index]]
        for cluster_set in cluster_sets
    ]


def affinity_clustering(dist_matrix: np.ndarray, eps: float = 1e-7) -> list[list[list[int]]]:
    num_elem = dist_matrix.shape[0]
    dist_matrix = dist_matrix + np.random.rand(num_elem, num_elem) * eps

    clusters = [[i] for i in range(num_elem)]
    clusters_per_level: list[list[list[int]]] = [clusters]
    level = 0

    while len(clusters) > 1:
        closest_cluster, closest_cluster_dists = find_closest_cluster(dist_matrix, clusters)
        new_clusters = merge_clusters(clusters, closest_cluster, len(clusters))

        if level == 0:
            while len(new_clusters) == 1:
                logging.info(
                    "All nodes merged in the first iteration; removing the most expensive edge"
                )
                idx = int(np.argmax(closest_cluster_dists))
                closest_cluster[idx] = None
                closest_cluster_dists[idx] = -np.inf
                new_clusters = merge_clusters(clusters, closest_cluster, len(clusters))

        clusters = new_clusters
        clusters_per_level.append(clusters)
        level += 1

    return clusters_per_level


def compute_hierarchy(
    dist_matrix: np.ndarray,
    make_symmetric: bool = True,
    eps: float = 1e-7,
) -> list[list[list[int]]]:
    working = np.array(dist_matrix, copy=True)
    if make_symmetric:
        working = working + working.T
    logging.info("Computing hierarchy from distance matrix with shape %s", working.shape)
    clusters_per_level = affinity_clustering(working, eps=eps)
    clusters_per_level = clusters_per_level[::-1]
    return clusters_per_level[1:]


def singleton_clusters(num_classes: int) -> list[list[int]]:
    return [[i] for i in range(num_classes)]


def clusters_to_label_map(
    clusters: Iterable[Iterable[int]],
    num_classes: int,
) -> np.ndarray:
    mapping = np.zeros((num_classes,), dtype=np.int64)
    for new_label, cluster in enumerate(clusters):
        for old_label in cluster:
            mapping[int(old_label)] = new_label
    return mapping

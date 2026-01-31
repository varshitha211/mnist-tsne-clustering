import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score

def cluster_stats(clusters, labels):
    stats = {}
    for c in set(clusters):
        labels_in_cluster = labels[clusters == c]
        most_common = Counter(labels_in_cluster).most_common(1)[0]
        stats[int(c)] = {
            "count": len(labels_in_cluster),
            "most_common_digit": most_common[0],
            "frequency": most_common[1]
        }
    return stats

def clustering_accuracy(clusters, labels):
    correct = 0
    for c in set(clusters):
        labels_in_cluster = labels[clusters == c]
        majority = Counter(labels_in_cluster).most_common(1)[0][0]
        correct += (labels_in_cluster == majority).sum()
    return correct / len(labels)

def silhouette(X_2d, clusters):
    return silhouette_score(X_2d, clusters)

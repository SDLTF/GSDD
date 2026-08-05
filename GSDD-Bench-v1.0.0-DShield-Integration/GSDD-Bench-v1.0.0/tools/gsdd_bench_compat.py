"""Small Python-3.13 compatibility helpers injected into DShield-Official.

The official repository imports sklearn-extra KMedoids at module import time.
scikit-learn-extra currently lacks a reliable CPython 3.13 Windows wheel, so
this module provides a deterministic PAM-compatible KMedoids class with the
subset of the API used by DShield.
"""
from __future__ import annotations
import types
import numpy as np
from sklearn.metrics import pairwise_distances

class KMedoids:
    def __init__(self, n_clusters: int, method: str = "pam", max_iter: int = 100, random_state: int = 1):
        self.n_clusters = int(n_clusters)
        self.method = method
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.medoid_indices_ = None
        self.cluster_centers_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < self.n_clusters:
            raise ValueError("KMedoids requires a 2-D array with n_samples >= n_clusters")
        distances = pairwise_distances(x, metric="euclidean")
        rng = np.random.RandomState(self.random_state)
        # Deterministic farthest-first initialization after one seeded point.
        medoids = [int(rng.randint(len(x)))]
        while len(medoids) < self.n_clusters:
            nearest = distances[:, medoids].min(axis=1)
            nearest[medoids] = -1
            medoids.append(int(np.argmax(nearest)))
        medoids = np.asarray(medoids, dtype=np.int64)
        labels = np.zeros(len(x), dtype=np.int64)
        for _ in range(self.max_iter):
            labels = np.argmin(distances[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for cluster_id in range(self.n_clusters):
                members = np.flatnonzero(labels == cluster_id)
                if len(members) == 0:
                    candidates = np.setdiff1d(np.arange(len(x)), new_medoids, assume_unique=False)
                    if len(candidates):
                        nearest = distances[candidates][:, new_medoids].min(axis=1)
                        new_medoids[cluster_id] = candidates[int(np.argmax(nearest))]
                    continue
                intra = distances[np.ix_(members, members)].sum(axis=1)
                new_medoids[cluster_id] = members[int(np.argmin(intra))]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
        self.medoid_indices_ = medoids
        self.cluster_centers_ = x[medoids]
        return self

    def predict(self, x):
        if self.cluster_centers_ is None:
            raise RuntimeError("KMedoids.fit must be called before predict")
        return np.argmin(pairwise_distances(np.asarray(x), self.cluster_centers_), axis=1)

cluster = types.SimpleNamespace(KMedoids=KMedoids)

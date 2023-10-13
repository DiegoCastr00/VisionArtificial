import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
def K_Means(data, columns, num_clusters):
    X = data[columns].values
    inertia_values = []

    def calculate_distances(X, centroids):
        num_points = X.shape[0]
        num_clusters = centroids.shape[0]
        distances = np.zeros((num_points, num_clusters))
        for i in range(num_clusters):
            centroid = centroids[i]
            squared_distances = np.sum((X - centroid) ** 2, axis=1)
            distances[:, i] = np.sqrt(squared_distances)  
        return distances
    def update_centroids(X, labels, num_clusters):
        new_centroids = np.zeros((num_clusters, X.shape[1]))
        for i in range(num_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        return new_centroids
    def calculate_inertia(X, labels, centroids):
        inertia = 0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            squared_distances = np.sum((cluster_points - centroids[i]) ** 2)
            inertia += squared_distances
        return inertia
    
    fig = plt.figure(figsize=(12, 22))
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], num_clusters, replace=False)]
    max_iters = 100
    inertia_values = []  # Debes inicializar inertia_values antes de usarlo
    max_clusters = num_clusters
    for num_clusters in range(1, max_clusters + 1):
        np.random.seed(0)
        centroids = X[np.random.choice(X.shape[0], num_clusters, replace=False)]
        max_iters = 100

        for _ in range(max_iters):
            distances = calculate_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = update_centroids(X, labels, num_clusters)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        inertia = calculate_inertia(X, labels, centroids)
        inertia_values.append(inertia)  # Agrega el valor de inercia a la lista

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='rainbow')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='black', label='Centroides')
        ax.set_title(f'Clusters: {num_clusters}\nInercia: {inertia:.2f}')
        ax.legend()
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2])


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_clusters + 1), inertia_values, marker='o', linestyle='-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title(f'Método del Codo para Encontrar el Número Óptimo de Clusters ({columns[0]}, {columns[1]}, {columns[2]})')
    plt.grid(True)
    plt.show()

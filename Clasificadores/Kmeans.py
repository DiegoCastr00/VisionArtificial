import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importar Axes3D de mpl_toolkits.mplot3d
import pandas as pd

def K_Means(data, columns, num_clusters):
    num_clusters = num_clusters + 1;
    X_scaled = data[columns].values
    inertia_values = []

    def calculate_inertia(X, labels, centroids):
        inertia = 0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            squared_distances = np.sum((cluster_points - centroids[i]) ** 2)
            inertia += squared_distances
        return inertia

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

    np.random.seed(0)
    centroids = X_scaled[np.random.choice(X_scaled.shape[0], num_clusters, replace=False)]
    max_iters = 1000

    for _ in range(max_iters):
        distances = calculate_distances(X_scaled, centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = update_centroids(X_scaled, labels, num_clusters)

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    inertia = calculate_inertia(X_scaled, labels, centroids)
    inertia_values.append(inertia)

    # Visualización de los clusters en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Crear un gráfico 3D

    # Graficar los datos con puntos
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap='rainbow', marker='o', label='Datos')

    # Graficar los centroides con una "x"
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=100, c='black', marker='x', label='Centroides')

    ax.set_title(f'Clusters: {num_clusters-1}\nInercia: {inertia:.2f}')
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    plt.legend()
    plt.show()



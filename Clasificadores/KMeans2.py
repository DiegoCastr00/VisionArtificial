import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def disEuclidian(point1, point2):
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def centroidsRand(data, k):
    centroids = random.sample(data, k)
    return centroids

def AsignCentroids(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    
    for point in data:
        distances = [disEuclidian(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    
    return clusters

def newCentroids(clusters):
    centroids = []
    for cluster in clusters:
        cluster_center = [sum(point[i] for point in cluster) / len(cluster) for i in range(len(cluster[0]))]
        centroids.append(cluster_center)
    return centroids

def covergencia(old_centroids, new_centroids, tol=1e-4):
    return all(disEuclidian(old, new) < tol for old, new in zip(old_centroids, new_centroids))

def k_means(data, k):
    centroids = centroidsRand(data, k)
    converged = False
    while not converged:
        clusters = AsignCentroids(data, centroids)
        new_centroids = newCentroids(clusters)
        converged = covergencia(centroids, new_centroids)
        centroids = new_centroids
        
    labeled_data = []
    for i, cluster in enumerate(clusters):
        for point in cluster:
            index = data.index(point) 
            labeled_data.append((point, i, index)) 

    return centroids, clusters, labeled_data

def plot_clusters(centroids, clusters):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for i, cluster in enumerate(clusters):
        x, y = zip(*cluster)
        ax1.scatter(x, y, label=f'Cluster {i + 1}')

    centroids_x, centroids_y = zip(*centroids)
    ax1.scatter(centroids_x, centroids_y, color='black', marker='x', s=100, label='Centroides')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('K-Means Clustering')
    ax1.legend()
    ax1.grid(True)

def plot_clusters_3d(centroids, clusters):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111, projection='3d')

    for i, cluster in enumerate(clusters):
        x, y, z = zip(*cluster)
        ax1.scatter(x, y, z, label=f'Cluster {i + 1}')

    centroids_x, centroids_y, centroids_z = zip(*centroids)
    ax1.scatter(centroids_x, centroids_y, centroids_z, color='black', marker='x', s=100, label='Centroides')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('K-Means Clustering in 3D')
    ax1.legend()

    plt.show()
    

def calculate_inertia(data, clusters, centroids):
    total_inertia = 0
    for i in range(len(centroids)):
        cluster_points = clusters[i]
        centroid = centroids[i]
        inertia_cluster = sum(disEuclidian(centroid, point) ** 2 for point in cluster_points)
        total_inertia += inertia_cluster
    return total_inertia

def elbow_method(data, max_clusters):
    inertia_values = []
    for k in range(1, max_clusters + 1):
        centroids, clusters, _ = k_means(data, k)
        inertia = calculate_inertia(data, clusters, centroids)
        inertia_values.append(inertia)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Encontrar el Número Óptimo de Clusters')
    plt.grid(True)
    plt.show()

def KmeansComplete(datos, k):
    data_list = datos.tolist()
    centroids, clusters, newdata = k_means(data_list, k)    
    plot_clusters_3d(centroids, clusters)
  
    for i, centroid in enumerate(centroids):
        print(f"Centroide {i + 1}: {centroid}")
        print(f"Puntos en el cluster {i + 1}: {clusters[i]}")

    return newdata

def kMeansCompleteIMG(IMG, k): 
    R, G, B = IMG[:, :, 0].reshape(-1, 1) , IMG[:, :, 1].reshape(-1, 1) , IMG[:, :, 2].reshape(-1, 1)
    nueva_matriz = np.column_stack((R, G, B))
    data_list = nueva_matriz.tolist()
    centroids, clusters = k_means(data_list, k)    
    
    inertia = []
    for k in range(1, len(clusters) + 1):
        total_inertia = calculate_inertia(k, clusters, centroids)
        inertia.append(total_inertia)

    plot_clusters_3d(centroids, clusters)
  
    for i, centroid in enumerate(centroids):
        print(f"Centroide {i + 1}: {centroid}")
        print(f"Puntos en el cluster {i + 1}: {clusters[i]}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(clusters) + 1), inertia, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Determinar k')
    plt.grid(True)
    plt.show()
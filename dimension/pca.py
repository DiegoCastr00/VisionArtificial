def PCA(X,num_components):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    # Estandarizar los datos)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calcular la matriz de covarianza
    cov_matrix = np.cov(X_scaled, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    num_components = num_components
    top_eigenvectors = eigenvectors[:, :num_components]
    X_pca = X_scaled.dot(top_eigenvectors)

    # if num_components == 3:
    #     plt.figure(figsize=(15, 15))
    #     ax = plt.subplot(1, 2, 1, projection='3d')

    #     for target, color in zip(np.unique(y), ['r', 'g', 'b']):
    #         indices_to_keep = y == target
    #         ax.scatter(X[indices_to_keep, 0], X[indices_to_keep, 1],X[indices_to_keep, 2], c=color, label=target)
            
    #     ax.set_xlabel('Característica 1')
    #     ax.set_ylabel('Característica 2')
    #     ax.set_zlabel('Característica 3')
    #     ax.set_title('Conjunto de Datos')
        
    #     class_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    #     ax = plt.subplot(1, 2, 2, projection='3d')
    #     for target in np.unique(y):
    #         indices_to_keep = y == target
    #         ax.scatter(X_pca[indices_to_keep, 0], X_pca[indices_to_keep, 1],X[indices_to_keep, 2], label=class_mapping[target])

    #     ax.set_xlabel('Componente Principal 1')
    #     ax.set_ylabel('Componente Principal 2')
    #     ax.set_zlabel('Componente Principal 3')
    #     ax.set_title('Datos después de PCA')
    #     ax.legend()
    #     plt.tight_layout()
        
        
    # elif num_components == 2:
    #     plt.figure(figsize=(12, 6))
    #     ax = plt.subplot(1, 2, 1)
    #     for target, color in zip(np.unique(y), ['r', 'g', 'b']):
    #         indices_to_keep = y == target
    #         ax.scatter(X[indices_to_keep, 0], X[indices_to_keep, 1], c=color, label=target)
    #     ax.set_xlabel('Característica 1')
    #     ax.set_ylabel('Característica 2')
    #     ax.set_title('Conjunto de Datos')
        
    #     ax = plt.subplot(1, 2, 2)
    #     for target in np.unique(y):
    #         indices_to_keep = y == target
    #         ax.scatter(X_pca[indices_to_keep, 0], X_pca[indices_to_keep, 1])

    #     ax.set_xlabel('Componente Principal 1')
    #     ax.set_ylabel('Componente Principal 2')
    #     ax.set_title('Datos después de PCA')
    #     ax.legend()
    #     plt.tight_layout()
        
    # elif num_components == 1:
    #     plt.figure(figsize=(12, 6))
    #     ax = plt.subplot(1, 2, 1)
    #     for target, color in zip(np.unique(y), ['r', 'g', 'b']):
    #         indices_to_keep = y == target
    #         ax.scatter(X[indices_to_keep, 0], X[indices_to_keep, 1], c=color, label=target)
    #     ax.set_xlabel('Característica 1')
    #     ax.set_ylabel('Característica 2')
    #     ax.set_title('Conjunto de Datos')
        
    #     class_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    #     ax = plt.subplot(1, 2, 2)
    #     for target in np.unique(y):
    #         indices_to_keep = y == target
    #         y_constant = np.zeros(X_pca.shape[0]) 
    #         ax.scatter(X_pca[indices_to_keep, 0], np.zeros(np.sum(indices_to_keep)), label=class_mapping[target])
    #     ax.set_xlabel('Primer Componente Principal')
    #     ax.set_title('PCA del Conjunto de Datos')
    #     ax.legend()
    #     plt.tight_layout()
    # else:
    #     print("\nNo se puede graficar en más de 3 dimensiones")
        
    return X_pca